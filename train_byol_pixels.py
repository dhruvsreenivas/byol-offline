#! /usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import os
import dmcgym
import gym
import jax
import tqdm
from absl import app, flags
from ml_collections import config_flags
import wandb
from typing import Tuple
from PIL import Image
import numpy as np

from byol_offline.models import BYOLLearner
from byol_offline.data.vd4rl_dataset import VD4RLDataset
from byol_offline.wrappers import wrap_pixels

"""Trains BYOL-Explore world model offline."""

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "byol_offline_pixels", "WandB project name.")
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
flags.DEFINE_integer("log_interval", 50, "Logging interval.")
flags.DEFINE_integer("eval_interval", None, "How often to evaluate. Defaults to not evaluating.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("sequence_length", 5, "Sequence length.")
flags.DEFINE_integer("max_steps", 3000, "Number of training steps.")

flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Number of frames to stack.")
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat. If None, uses 2 or PlaNet defaults."
)

flags.DEFINE_boolean("tqdm", True, "Whether to use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Whether to use WandB logging.")
flags.DEFINE_integer("save_interval", 1000, "How often to save.")
flags.DEFINE_boolean(
    "checkpoint_model", False, "Whether to checkpoint model."
)
flags.DEFINE_integer("max_checkpoints", 10, "Maximum number of checkpoints to save.")

config_flags.DEFINE_config_file(
    "config",
    "configs/models/byol_config.py",
    "File path to the training hyperparameter configuration.",
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


def main(_):
    
    # first initialize wandb project
    group = "-".join([FLAGS.env_name, FLAGS.dataset_level])
    wandb.init(
        project=FLAGS.project_name, entity="dhruv_sreenivas",
        mode="disabled" if not FLAGS.wandb else None, group=group
    )
    wandb.config.update(FLAGS)
    
    # set up checkpointing
    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(
            "checkpoints", "byol", FLAGS.env_name, FLAGS.dataset_level
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
        
    env = gym.make(FLAGS.env_name)
    env, pixel_keys = wrap(env)
    env.seed(FLAGS.seed)
    
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
            sequence_length=FLAGS.sequence_length,
            batch_size=FLAGS.batch_size,
            pack_obs_and_next_obs=FLAGS.pack_obs_and_next_obs
        )
    )
    
    # instantiate learner
    config = FLAGS.config
    config.pmap = jax.local_device_count() > 1
    learner = BYOLLearner(
        config, FLAGS.seed, env.observation_space, env.action_space
    )
    
    # now start training
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        # grab a batch of sequences
        batch = next(ds_iterator)
        
        learner._state, metrics = learner._update(
            learner._state, batch, step=i
        )
        
        if i % FLAGS.log_interval == 0:
            for k, v in metrics.items():
                wandb.log({f"train/{k}": v}, step=i)
                
        # evaluate every so often
        if FLAGS.eval_interval is not None and i % FLAGS.eval_interval == 0:
            learner._state, posterior_images, prior_images = learner._eval(learner._state, batch)
            actual_images = batch.observations[:, 0, ...][:, :, :, :3]
            
            actual_images = [
                Image.fromarray(np.asarray(image), mode="RGB")
                for image in actual_images
            ]
            posterior_images = [
                Image.fromarray(np.asarray(image), mode="RGB")
                for image in posterior_images
            ]
            prior_images = [
                Image.fromarray(np.asarray(image), mode="RGB")
                for image in prior_images
            ]
            
            for actual, posterior, prior in zip(actual_images, posterior_images, prior_images):
                wandb.log(
                    {
                        "actual": wandb.Image(actual, "Actual"),
                        "posterior": wandb.Image(posterior, "Posterior"),
                        "prior": wandb.Image(prior, "Prior")
                    }
                )
                
        # optionally save
        if FLAGS.checkpoint_model and i % FLAGS.save_interval == 0:
            checkpoints = [
                os.path.join(chkpt_dir, fn) for fn in os.listdir(chkpt_dir)
            ]
            
            # remove oldest one if needed
            if len(checkpoints) == FLAGS.max_checkpoints:
                oldest_chkpt = min(checkpoints, key=os.path.getctime)
                os.remove(oldest_chkpt)
                
            checkpoint_path = os.path.join(chkpt_dir, f"ckpt_{i}.pkl")
            learner.save(checkpoint_path)
            
    # save final checkpoint
    if FLAGS.checkpoint_model:
        final_checkpoint_path = os.path.join(chkpt_dir, "final_ckpt.pkl")
        learner.save(final_checkpoint_path)
        

if __name__ == "__main__":
    app.run(main)