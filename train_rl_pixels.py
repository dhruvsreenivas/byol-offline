#! /usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import os
import dmcgym
import gym
import jax
import jax.numpy as jnp
import tqdm
from absl import app, flags
from ml_collections import config_flags
import wandb
from typing import List, Mapping, Tuple, Optional
import numpy as np
import random

from byol_offline.models import BYOLLearner
from byol_offline.agents import SACLearner
from byol_offline.data import (
    _preprocess,
    LatentReplayBuffer,
    Batch,
    SequenceBatch,
)
from byol_offline.data.dataset import _stack_dicts, _dict_to_batch, _batch_to_dict
from byol_offline.data.vd4rl_dataset import VD4RLDataset
from byol_offline.wrappers import wrap_pixels

from utils import combine_batches, evaluate_model_based

"""Does model-based RL training offline."""

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "byol-mbrl-pixels", "WandB project name.")
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string(
    "dataset_level", "medium", "Dataset level (e.g. random, medium, expert, etc.)"
)
flags.DEFINE_string("dataset_path", None, "Path to dataset. Defaults to `~/.vd4rl`.")
flags.DEFINE_integer("dataset_size", 500_000, "How many samples to load from the dataset directory.")
flags.DEFINE_boolean(
    "pack_obs_and_next_obs", True, "Whether to pack observation and next observations in batch."
)

flags.DEFINE_integer("seed", 69, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", None, "How often to evaluate. Defaults to not evaluating.")
flags.DEFINE_integer("eval_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("rollout_length", 5, "Number of timesteps to roll out in world model.")
flags.DEFINE_float("reward_penalty_coef", 0.0, "Reward penalty coefficient.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer(
    "start_training", 50_000, "Number of steps before training officially starts."
)

flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 1, "Number of frames to stack.")
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat. If None, uses 2 or PlaNet defaults."
)

flags.DEFINE_boolean("tqdm", True, "Whether to use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Whether to use WandB logging.")
flags.DEFINE_integer("save_interval", 1000, "How often to save.")
flags.DEFINE_boolean(
    "checkpoint_agent", False, "Whether to checkpoint agent."
)
flags.DEFINE_boolean(
    "resume_from_checkpoint", False, "Whether to resume from the latest checkpoint."
)
flags.DEFINE_integer("max_checkpoints", 10, "Maximum number of checkpoints to save.")
flags.DEFINE_boolean("load_buffer", False, "Whether to load buffer in.")

config_flags.DEFINE_config_file(
    "rl_config",
    "configs/agents/sac_config.py",
    "File path to the RL hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "byol_config",
    "configs/models/byol_config.py",
    "File path to the world model hyperparameter configuration. Used only for initialization.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spin-v0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}


def process_dataset_to_latents(
    model: BYOLLearner, ds: VD4RLDataset, num_episodes: Optional[int] = None,
) -> List[Mapping[str, np.ndarray]]:
    """Process all observation data to latents, and return the resulting trajectories."""
    
    latent_trajectories = []
    episodes = random.choices(ds._episodes, k=num_episodes) if num_episodes is not None else ds._episodes
    for episode in episodes:
        
        # first preprocess the episode
        episode = _preprocess(episode)
        
        # expand it out to get batch dimension in
        episode = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=1), episode
        )
        
        model._state, features = model._process_to_latent(
            model._state, episode.observations, episode.actions
        ) # [T, 1, feature_dim]
        
        latent_trajectory = dict(
            observations=features[:-1],
            actions=episode.actions[1:],
            rewards=episode.rewards[1:],
            next_observations=features[1:],
            dones=episode.dones[1:],
            masks=episode.masks[1:]
        )
        latent_trajectory = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x), latent_trajectory
        ) # [T, feature_dim]
        
        latent_trajectories.append(latent_trajectory)
        
    return latent_trajectories


def latent_rollout(
    agent: SACLearner,
    model: BYOLLearner,
    batch: Batch,
    sequence_length: int,
    reward_penalty_coef: float = 0.0,
) -> SequenceBatch:
    """Rolls out actions selected by the learner in the world model."""
    
    # first encode into embeddings
    model._state, embs = model._encode(model._state, batch.observations)
    B = embs.shape[0]
    
    # start with initial state
    model._state, initial_state = model._initialize_state(model._state, B)
    
    # first step is a posterior step
    model._state, (state, features, _) = model._observe(
        model._state, embs, batch.actions, initial_state
    )
    masks = jnp.expand_dims(batch.masks, -1)
    features *= masks
    
    # now because we cannot do a scan (assigning state is not good when JIT compiling), we have to python for loop. REEE
    trajectory = []
    for _ in range(sequence_length):
        
        # get action given features
        agent._state, actions = agent._act(
            agent._state, features, 0, eval=False
        )
        
        # roll out in model
        model._state, (next_state, next_features, rewards) = model._imagine(
            model._state, actions, state, sample=False
        )
        rewards = np.squeeze(rewards)
        
        # add dict to the list
        time_step = dict(
            observations=features,
            actions=actions,
            rewards=rewards,
            next_observations=next_features,
            dones=np.zeros_like(rewards),
            masks=np.ones_like(rewards),
        )
        trajectory.append(time_step)
        
        # update
        state = next_state
        features = next_features
        
    trajectory = _stack_dicts(trajectory, axis=0)
    batch = _dict_to_batch(trajectory, is_sequence=False)
    
    # now add reward penalty coefficient
    if reward_penalty_coef > 0.0:
        model._state, reconstructed_observations = model._decode(
            model._state, trajectory["observations"]
        )
        
        pseudo_batch = batch._replace(
            observations=reconstructed_observations
        )
        model._state, penalties = model._compute_uncertainty(
            model._state, pseudo_batch
        )
        assert batch.rewards.shape == penalties.shape
        
        batch = batch._replace(
            rewards=batch.rewards - reward_penalty_coef * penalties
        )

    return batch


def main(_):
    
    # first initialize wandb project
    group = "-".join([FLAGS.env_name, FLAGS.dataset_level])
    wandb.init(
        project=FLAGS.project_name, entity="dhruv_sreenivas",
        mode="disabled" if not FLAGS.wandb else None, group=group
    )
    wandb.config.update(FLAGS)
    
    if FLAGS.checkpoint_agent:
        chkpt_dir = os.path.join(
            "checkpoints", "rl", FLAGS.env_name, FLAGS.dataset_level
        )
        os.makedirs(chkpt_dir, exist_ok=True)
        
    action_repeat = FLAGS.action_repeat or PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)
    
    def wrap(env: gym.Env) -> Tuple[gym.Env, Tuple[str, ...]]:
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )
    
    # create train + eval envs
    env = gym.make(FLAGS.env_name)
    env, pixel_keys = wrap(env)
    env.seed(FLAGS.seed)
    
    eval_env = gym.make(FLAGS.env_name)
    eval_env, _ = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 69)
    
    ds = VD4RLDataset(
        env,
        FLAGS.dataset_level,
        FLAGS.image_size,
        pixel_keys=pixel_keys,
        capacity=FLAGS.dataset_size,
        dataset_path=FLAGS.dataset_path,
    )
    ds_iterator = ds.get_iterator(
        sample_args=dict(
            batch_size=FLAGS.batch_size,
            pack_obs_and_next_obs=FLAGS.pack_obs_and_next_obs
        )
    )
    ds.seed(FLAGS.seed)
    
    # instantiate model, and load it in
    byol_config = FLAGS.byol_config
    byol_config.pmap = jax.local_device_count() > 1
    model = BYOLLearner(
        byol_config, FLAGS.seed, env.observation_space, env.action_space
    )
    
    # load final model checkpoint
    checkpoint = os.path.join(
        "checkpoints", "byol", FLAGS.env_name, FLAGS.dataset_level, "final_ckpt.pkl"
    )
    model.load(checkpoint)
    
    # define latent replay buffer
    rssm_config = byol_config.pixel.rssm
    representation_dim = rssm_config.gru_hidden_size + rssm_config.stoch_discrete_dim * rssm_config.stoch_dim
    replay_buffer = LatentReplayBuffer(
        representation_dim,
        env.action_space,
        FLAGS.seed,
    )
    
    # process all real data and add to replay buffer if needed
    buffer_dir = os.path.join(
        "latent_buffers", FLAGS.env_name, FLAGS.dataset_level
    )
    
    buffer_chkpt = os.path.join(buffer_dir, "populated.pkl")
    populated_buffer_exists = os.path.exists(buffer_chkpt)
    if not populated_buffer_exists:
        buffer_chkpt = os.path.join(buffer_dir, f"base.pkl")
    
    if FLAGS.load_buffer:
        replay_buffer.load(buffer_chkpt)
    else:
        latent_trajectories = process_dataset_to_latents(model, ds)
        for trajectory in latent_trajectories:
            replay_buffer.insert_trajectory(trajectory, real=True)
        
        # save buffer for future loading
        os.makedirs(buffer_dir, exist_ok=True)
        replay_buffer.save(buffer_chkpt)
    
    # initialize agent
    agent_config = FLAGS.rl_config
    agent_config.pmap = jax.local_device_count() > 1
    agent_config.observation_repr_dim = representation_dim
    agent = SACLearner(
        agent_config, FLAGS.seed, env.observation_space, env.action_space
    )
    
    # if we are resuming from a checkpoint, we load in the latest one
    if FLAGS.resume_from_checkpoint:
        checkpoints = [
            os.path.join(chkpt_dir, fn) for fn in os.listdir(chkpt_dir)
        ]
        if len(checkpoints) > 0:
            latest_chkpt = max(checkpoints, key=os.path.getctime)
            agent.load(latest_chkpt)
            
            checkpoint_num = int(latest_chkpt.split("_")[-1][:-4])
            print(f"*** Starting from checkpoint {checkpoint_num}. ***")
    else:
        checkpoint_num = 0
    
    # populate buffer with random data
    if not populated_buffer_exists:
        for _ in tqdm.tqdm(
            range(FLAGS.start_training // FLAGS.rollout_length),
            smoothing=0.1,
            disable=not FLAGS.tqdm,
            desc="Buffer populating"
        ):
            batch = next(ds_iterator)
            rollouts = latent_rollout(
                agent, model, batch, FLAGS.rollout_length,
                reward_penalty_coef=FLAGS.reward_penalty_coef
            )
            
            rollout_dict = _batch_to_dict(rollouts)
            replay_buffer.insert_batch_of_trajectories(rollout_dict, real=False)
        
        # now save the buffer
        populated_buffer_chkpt = os.path.join(buffer_dir, "populated.pkl")
        replay_buffer.save(populated_buffer_chkpt)
    
    # now train the agent
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1 - checkpoint_num),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
        desc="Offline RL",
    ):
        
        # --- sample from buffer ---
        
        real_batch = replay_buffer.sample(
            FLAGS.batch_size, from_real=True
        )
        mb_batch = replay_buffer.sample(
            FLAGS.batch_size, from_real=False
        )
        combined_batch = combine_batches(real_batch, mb_batch)
        
        # --- update the agent ---
        
        agent._state, metrics = agent._update(
            agent._state, combined_batch, step=i
        )
        
        # --- resample data and add it into the buffer ---
        
        # real
        latent_trajectories = process_dataset_to_latents(model, ds, num_episodes=1)
        for trajectory in latent_trajectories:
            replay_buffer.insert_trajectory(trajectory, real=True)
        
        # model-based
        batch = next(ds_iterator)
        rollouts = latent_rollout(
            agent, model, batch, FLAGS.rollout_length,
            reward_penalty_coef=FLAGS.reward_penalty_coef
        )
        rollout_dict = _batch_to_dict(rollouts)
        replay_buffer.insert_batch_of_trajectories(rollout_dict, real=False)
        
        # --- if logging time, we log ---
        
        if i % FLAGS.log_interval == 0:
            for k, v in metrics.items():
                wandb.log({f"train/{k}": v}, step=i + checkpoint_num)
        
        # --- evaluate every so often ---
        
        if FLAGS.eval_interval is not None and i % FLAGS.eval_interval == 0:
            eval_metrics = evaluate_model_based(
                eval_env, agent, model, FLAGS.eval_episodes,
            )
            
            for k, v in eval_metrics.items():
                wandb.log({f"evaluation/{k}": v}, step=i + checkpoint_num)
        
        # --- optionally save ---
        
        if FLAGS.checkpoint_agent and i % FLAGS.save_interval == 0:
            checkpoints = [
                os.path.join(chkpt_dir, fn) for fn in os.listdir(chkpt_dir)
            ]
            
            # remove oldest one if needed
            if len(checkpoints) == FLAGS.max_checkpoints:
                oldest_chkpt = min(checkpoints, key=os.path.getctime)
                os.remove(oldest_chkpt)
                
            checkpoint_path = os.path.join(chkpt_dir, f"ckpt_{i + checkpoint_num}.pkl")
            agent.save(checkpoint_path)
    
        
    # --- save final checkpoint ---
    if FLAGS.checkpoint_agent:
        final_checkpoint_path = os.path.join(chkpt_dir, "final_ckpt.pkl")
        agent.save(final_checkpoint_path)
        

if __name__ == "__main__":
    app.run(main)