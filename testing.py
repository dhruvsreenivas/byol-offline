import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import hydra
import gym
from pathlib import Path
from tqdm import trange
import wandb

from byol_offline.models import *
from byol_offline.agents.td3 import TD3
from memory.replay_buffer import *
from utils import get_gym_dataset, make_gym_env

'''Various testing functions to make sure core machinery works.'''

@hydra.main(config_path='cfgs', config_name='config')
def test_world_model(cfg):
    # Make sure print statements are enabled in WM __call__ function to print out state when running this method
    key = jax.random.PRNGKey(42)
    init_key, key1, key2 = jax.random.split(key, 3)

    # dummy input creation
    dummy_obs = jax.random.normal(key1, shape=(20, 10, 111))
    dummy_actions = jax.random.normal(key2, shape=(20, 10, 7))
    
    # world model creation
    wm_fn = lambda o, a: MLPLatentWorldModel(cfg.byol.d4rl)(o, a)
    wm = hk.without_apply_rng(hk.transform(wm_fn))
    params = wm.init(init_key, dummy_obs, dummy_actions)

    # sliding window mask
    mask = jnp.arange(20)
    mask = jnp.expand_dims(mask, axis=range(1, jnp.ndim(dummy_obs))) # (20, 1, 1), can use same mask for dummy action
    mask = jnp.logical_or(mask < 7, mask >= 7 + 8)

    obs = jnp.where(mask, 0, dummy_obs)
    actions = jnp.where(mask, 0, dummy_actions)
    obs = jnp.roll(obs, -7)
    actions = jnp.roll(actions, -7)

    _ = wm.apply(params, obs, actions)

def test_sampler_dataloading(d4rl=True, byol=True):
    '''Testing dataloading across epochs.'''
    if d4rl:
        if byol:
            buffer = D4RLSequenceReplayBuffer('hopper', 'medium', 25)
            dataloader = byol_sampling_dataloader(buffer, max_steps=200, batch_size=20)
        else:
            buffer = D4RLTransitionReplayBuffer('hopper', 'medium')
            dataloader = rnd_sampling_dataloader(buffer, max_steps=200, batch_size=20)
    else:
        data_path = Path('./offline_data/cheetah_run/med_exp')
        if byol:
            buffer = VD4RLSequenceReplayBuffer(data_path, 25)
            dataloader = byol_sampling_dataloader(buffer, max_steps=200, batch_size=20)
        else:
            buffer = VD4RLTransitionReplayBuffer(data_path)
            dataloader = rnd_sampling_dataloader(buffer, max_steps=200, batch_size=20)
    
    for epoch in range(10):
        print(f'starting epoch {epoch + 1}')
        print('*' * 50)
        for batch in dataloader:
            obs, action, reward, next_obs, done = batch
            print(f'obs info: shape: {obs.shape}, dtype: {obs.dtype}')
            print(f'action info: shape: {action.shape}, dtype: {action.dtype}')
            print(f'reward info: shape: {reward.shape}, dtype: {reward.dtype}')
            print(f'next obs info: shape: {next_obs.shape}, dtype: {next_obs.dtype}')
            print(f'done info: shape: {done.shape}, dtype: {done.dtype}')
            print('*' * 50)
            
def batch_eq(batch1, batch2):
    obs1, act1, rew1, nobs1, d1 = batch1
    obs2, act2, rew2, nobs2, d2 = batch2
    
    assert obs1.shape == obs2.shape
    assert act1.shape == act2.shape
    assert rew1.shape == rew2.shape
    assert nobs1.shape == nobs2.shape
    assert d1.shape == d2.shape
    
    obs_eq = np.all(obs1 == obs2)
    act_eq = np.all(act1 == act2)
    rew_eq = np.all(rew1 == rew2)
    nobs_eq = np.all(nobs1 == nobs2)
    d_eq = np.all(d1 == d2)
    
    assert obs_eq == act_eq == rew_eq == nobs_eq == d_eq
    return obs_eq and act_eq and rew_eq and nobs_eq and d_eq
            
def test_iterative_dataloading():
    name = 'hopper'
    capability = 'medium'
    loader, _ = rnd_iterative_dataloader(name, capability, batch_size=1024, normalize=True)
    
    batches = []
    for epoch in range(2):
        count = 0
        for batch in loader:
            print('*' * 50)
            obs, action, reward, next_obs, done = batch
            
            if epoch == 0:
                print(f'obs info: shape: {obs.shape}, dtype: {obs.dtype}')
                print(f'action info: shape: {action.shape}, dtype: {action.dtype}')
                print(f'reward info: shape: {reward.shape}, dtype: {reward.dtype}')
                print(f'next obs info: shape: {next_obs.shape}, dtype: {next_obs.dtype}')
                print(f'done info: shape: {done.shape}, dtype: {done.dtype}')
            
            if epoch == 0:
                batches.append(batch)
            else:
                # assert we're doing in different order
                eq = batch_eq(batch, batches[count])
                assert not eq, f"matched at count {count} -- if 0 this is not good haha"
            
            count += 1

@hydra.main(config_path='cfgs', config_name='config')
def test_bonus(cfg, byol=False):
    '''Test that bonus on non-medium dataset is lower than bonus on expert or random dataset.'''
    env = make_gym_env(cfg.task, cfg.level)
    cfg.obs_shape = env.observation_space.shape
    cfg.action_shape = env.action_space.shape
    
    if byol:
        trainer = WorldModelTrainer(cfg.byol)
    else:
        trainer = RNDModelTrainer(cfg.rnd)
    
    action_dir = 'actions' if cfg.rnd.cat_actions else 'no_actions'
    model_path = f'/home/ds844/byol-offline/pretrained_models/{"byol" if byol else "rnd"}/hopper/medium/{action_dir}/{"byol" if byol else "rnd"}_1000.pkl'
    trainer.load(model_path)
    
    # grabbing data
    random_dataset = get_gym_dataset(cfg.task, 'random')
    medium_dataset = get_gym_dataset(cfg.task, 'medium')
    expert_dataset = get_gym_dataset(cfg.task, 'expert')
    
    random_obs = random_dataset['observations']
    random_actions = random_dataset['actions']
    medium_obs = medium_dataset['observations']
    medium_actions = medium_dataset['actions']
    expert_obs = expert_dataset['observations']
    expert_actions = expert_dataset['actions']
    
    # normalize everything
    random_stats = get_mujoco_dataset_transformations(random_dataset)
    medium_stats = get_mujoco_dataset_transformations(medium_dataset)
    expert_stats = get_mujoco_dataset_transformations(expert_dataset)
    
    random_obs, random_actions = normalize_sa(random_obs, random_actions, random_stats)
    medium_obs, medium_actions = normalize_sa(medium_obs, medium_actions, medium_stats)
    expert_obs, expert_actions = normalize_sa(expert_obs, expert_actions, expert_stats)
    
    # uncertainties
    random_uncertainties = trainer._compute_uncertainty(random_obs, random_actions, 0)
    medium_uncertainties = trainer._compute_uncertainty(medium_obs, medium_actions, 0)
    expert_uncertainties = trainer._compute_uncertainty(expert_obs, expert_actions, 0)
    print(f'shapes: \n random: {random_uncertainties.shape} \n medium: {medium_uncertainties.shape} \n expert: {expert_uncertainties.shape}')
    
    print(f'stats for random {"byol" if byol else "rnd"} model: {np.mean(random_uncertainties)}, {np.std(random_uncertainties)}')
    print(f'stats for medium {"byol" if byol else "rnd"} model: {np.mean(medium_uncertainties)}, {np.std(medium_uncertainties)}')
    print(f'stats for expert {"byol" if byol else "rnd"} model: {np.mean(expert_uncertainties)}, {np.std(expert_uncertainties)}')
    
@hydra.main(config_path='cfgs', config_name='config')
def test_reward(cfg):
    '''Checking average reward in datasets.'''
    random_dataset = get_gym_dataset(cfg.task, 'random')
    medium_dataset = get_gym_dataset(cfg.task, 'medium')
    expert_dataset = get_gym_dataset(cfg.task, 'expert')
    
    random_rewards = random_dataset['rewards']
    medium_rewards = medium_dataset['rewards']
    expert_rewards = expert_dataset['rewards']
    
    print(f'stats for random dataset: {np.mean(random_rewards)}, {np.std(random_rewards)}')
    print(f'stats for medium dataset: {np.mean(medium_rewards)}, {np.std(medium_rewards)}')
    print(f'stats for expert dataset: {np.mean(expert_rewards)}, {np.std(expert_rewards)}')
    
@hydra.main(config_path='cfgs', config_name='config')
def test_rl_algo(cfg):
    '''Testing undelying RL algos for sanity checking purposes.'''
    assert cfg.reward_aug == 'none', 'Not doing any pessimism when testing online RL algo implementation.'
    
    # env + seed
    train_env = gym.make('Hopper-v2')
    eval_env = gym.make('Hopper-v2')
    train_env.seed(cfg.seed)
    train_env.action_space.seed(cfg.seed)
    eval_env.seed(cfg.seed + 100)
    np.random.seed(cfg.seed)
    
    # cfg update, exploration noise + buffer
    cfg.obs_shape = train_env.observation_space.shape
    cfg.action_shape = train_env.action_space.shape
    cfg.max_action = float(train_env.action_space.high[0])
    cfg.reward_min = -1.0
    cfg.reward_max = 1.0
    expl_noise = 0.1
    buffer = ReplayBuffer(int(1e6), cfg.obs_shape[0], cfg.action_shape[0])
    
    # rng key for sampling actions online
    rng = jax.random.PRNGKey(cfg.seed + 42)
    
    # init project
    entity = 'dhruv_sreenivas'
    wandb.init(project=cfg.project_name, entity=entity, name=f'td3 hopperv2 seed {cfg.seed}')
    
    # agent setup
    agent = TD3(cfg)
    
    def eval_policy(n_episodes=10):
        ep_rews = []
        ep_lens = []
        for _ in range(n_episodes):
            ob = eval_env.reset()
            done = False
            
            ep_rew = 0.0
            ep_len = 0
            while not done:
                action = agent._act(ob, 0, True)
                ob, rew, done, _ = eval_env.step(action)
                ep_rew += rew
                ep_len += 1
            
            ep_rews.append(ep_rew)
            ep_lens.append(ep_len)
            
        mean_rew, std_rew = np.mean(ep_rews), np.std(ep_rews)
        mean_len, std_len = np.mean(ep_lens), np.std(ep_lens)
        metric_dict = {
            'mean_rew': mean_rew,
            'std_rew': std_rew,
            'mean_len': mean_len,
            'std_len': std_len
        }
        wandb.log(metric_dict)
    
    # training loop
    ob, done = train_env.reset(), False
    episode_timesteps = 0
    episode_reward = 0.0
    for it in trange(int(1e6)):
        
        # do exploration if not collected enough data yet
        if it < 25000: # hardcoded for TD3
            action = train_env.action_space.sample()
        else:
            agent_action = agent._act(ob, 0, False)
            
            rng, sub_rng = jax.random.split(rng)
            noise = jax.random.normal(sub_rng, shape=agent_action.shape) * cfg.max_action * expl_noise # can also go to np.random.normal if needed
            action = jnp.clip(agent_action + noise, -cfg.max_action, cfg.max_action)
            
        # step env
        n_ob, rew, done, _ = train_env.step(action)
        episode_timesteps += 1
        done_bool = done if episode_timesteps < train_env._max_episode_steps else False
        
        # add to replay buffer
        buffer.add(ob, action, rew, n_ob, done_bool)
        
        ob = n_ob
        episode_reward += rew
        
        # train when ready
        if it >= 25000:
            transitions = buffer.sample(256)
            new_train_state, train_metrics = agent._update(agent.train_state, transitions, it)
            agent.train_state = new_train_state
            
            logged_metrics = {}
            for k, v in train_metrics.items():
                if v < jnp.inf:
                    logged_metrics[k] = v
                    
            wandb.log(logged_metrics)
            
        # reset episode measures when done
        if done:
            # print(f'Total iters: {it + 1}, episode timesteps: {episode_timesteps}, episode reward: {episode_reward}')
            ob, done = train_env.reset(), False
            episode_timesteps = 0
            episode_reward = 0.0
            
        if (it + 1) % 5000 == 0:
            eval_policy()
            
    wandb.finish()
    
if __name__ == '__main__':
    test_rl_algo()