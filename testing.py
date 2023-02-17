import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import hydra
import gym
from pathlib import Path
from hydra.utils import to_absolute_path
from tqdm import trange
import wandb
import time

from byol_offline.networks.encoder import *
from byol_offline.networks.decoder import *
from byol_offline.networks.rnn import *
from byol_offline.models import *
from byol_offline.agents.td3 import TD3
from memory.replay_buffer import *
from utils import get_gym_dataset, make_gym_env, print_dict, get_test_traj, to_seq_np

'''Various testing functions to make sure core machinery works.'''

def test_get_test_traj():
    data_dir = Path(to_absolute_path('offline_data')) / 'dmc' / 'cheetah_run' / 'med_exp'
    
    obs, actions, first_frames = get_test_traj(data_dir, frame_stack=3, seq_len=10)
    print(f'image shape: {obs.shape}')
    print(f'actions shape: {actions.shape}')
    print(f'first frames shape: {first_frames.shape}')

def test_encoding_decoding(dreamer=True):
    shape = (64, 64, 9)
    if dreamer:
        enc_fn = lambda x: DreamerEncoder(32)(x)
        dec_fn = lambda x: DreamerDecoder(shape[-1], 32)(x)
    else:
        enc_fn = lambda x: DrQv2Encoder()(x)
        dec_fn = lambda x: DrQv2Decoder(shape[-1])(x)
    
    enc = hk.without_apply_rng(hk.transform(enc_fn))
    dec = hk.without_apply_rng(hk.transform(dec_fn))
    
    key = jax.random.PRNGKey(42)
    enc_key, dec_key, key = jax.random.split(key, 3)
    enc_params = enc.init(enc_key, jnp.zeros((1,) + shape))
    dec_params = dec.init(dec_key, jnp.zeros((1, 1024 if dreamer else 20000)))
    
    ob_key, rep_key = jax.random.split(key)
    rand_inp = jax.random.normal(ob_key, shape=(10,) + shape)
    rand_rep = jax.random.normal(rep_key, shape=(10, 1024 if dreamer else 20000))
    
    out = enc.apply(enc_params, rand_inp)
    print(f'representation shape: {out.shape}') # should be rand rep shape
    assert out.shape == rand_rep.shape
    
    out_rec = dec.apply(dec_params, rand_rep)
    print(f'reconstruction shape: {out_rec.shape}')
    
@hydra.main(config_path='cfgs', config_name='config')
def test_rssm(cfg):
    rssm_fn = lambda e, a, s: RSSM(cfg.byol.vd4rl)(e, a, s) # don't think I need to multitransform yet, this is saved for WM
    rssm = hk.transform(rssm_fn)
    
    key = jax.random.PRNGKey(42)
    embed_key, action_key, init_key, apply_key = jax.random.split(key, 4)
    rand_embeds = jax.random.normal(embed_key, shape=(20, 10, 100))
    rand_actions = jax.random.normal(action_key, shape=(20, 10, 6))
    rand_state = None
    
    params = rssm.init(init_key, rand_embeds, rand_actions, rand_state)
    priors, posts, post_features, prior_features = rssm.apply(params, apply_key, rand_embeds, rand_actions, rand_state)
    print(f'prior shape: {priors.shape}')
    print(f'post shape: {posts.shape}')
    print(f'post features shape: {post_features.shape}')
    print(f'prior features shape: {prior_features.shape}')
    
@hydra.main(config_path='cfgs', config_name='config')
def test_world_model_module(cfg):
    '''Testing if the world model module apply method works.'''
    # Make sure print statements are enabled in WM __call__ function to print out state when running this method
    key = jax.random.PRNGKey(42)
    init_key, dreamer_key, byol_key, key1, key2 = jax.random.split(key, 5)
    
    cfg.obs_shape = (64, 64, 9)
    cfg.action_shape = (6,)

    # dummy input creation
    dummy_obs = jax.random.normal(key1, shape=(10, 50, 64, 64, 9))
    dummy_actions = jax.random.normal(key2, shape=(10, 50, 6))
    
    # world model creation
    def wm_fn():
        wm = ConvWorldModel(cfg.byol.vd4rl)
        
        def init(o, a):
            # same as standard forward pass
            return wm(o, a)
        
        def dreamer_forward(o, a):
            return wm._dreamer_forward(o, a)
        
        def byol_forward(o, a):
            return wm._byol_forward(o, a)
        
        def imagine_fn(a, s):
            return wm._onestep_imagine(a, s)
        
        return init, (dreamer_forward, byol_forward, imagine_fn)
    
    wm = hk.multi_transform(wm_fn)
    params = wm.init(init_key, dummy_obs, dummy_actions)

    dreamer_forward, byol_forward, _ = wm.apply
    pred_latents, embeds = byol_forward(params, dreamer_key, dummy_obs, dummy_actions)
    dreamer_stuff = dreamer_forward(params, byol_key, dummy_obs, dummy_actions)
    
    # initial checks
    print('=== shape checks (byol) ===')
    print(pred_latents.shape)
    print(embeds.shape)
    print('=== shape checks (dreamer) ===')
    for e in dreamer_stuff:
        print(e.shape)

@hydra.main(config_path='cfgs', config_name='config')
def test_world_model_update(cfg):
    '''Testing if the world model loss function/update scheme works.'''
    cfg.obs_shape = (64, 64, 9)
    cfg.action_shape = (6,)
    # cfg.pmap = True # for now just test pmapping abilities
    
    wm_trainer = WorldModelTrainer(cfg.byol)
    
    # rand inputs for updating
    key = jax.random.PRNGKey(42)
    first_key, second_key = jax.random.split(key)
    
    obs_key, act_key, rew_key = jax.random.split(first_key, 3)
    rand_obs = jax.random.normal(obs_key, (10, 50, 64, 64, 9))
    rand_act = jax.random.normal(act_key, (10, 50, 6))
    rand_rew = jax.random.normal(rew_key, (10, 50))
    print('=' * 20 + ' created first set of rand inputs ' + '=' * 20)
    
    start_time = time.time()
    _, metrics = wm_trainer._update(wm_trainer.train_state, rand_obs, rand_act, rand_rew, 0)
    end_time = time.time()
    print_dict(metrics)
    
    secs = end_time - start_time
    mins = int(secs // 60)
    secs = secs % 60
    print(f'total amount of time taken for jit compile + update: {mins} mins & {secs} secs.')
    
    obs_key_2, act_key_2, rew_key_2 = jax.random.split(second_key, 3)
    rand_obs_2 = jax.random.normal(obs_key_2, (10, 50, 64, 64, 9))
    rand_act_2 = jax.random.normal(act_key_2, (10, 50, 6))
    rand_rew_2 = jax.random.normal(rew_key_2, (10, 50))
    print('=' * 20 + ' created second set of rand inputs ' + '=' * 20)
    
    start_time_2 = time.time()
    _, metrics = wm_trainer._update(wm_trainer.train_state, rand_obs_2, rand_act_2, rand_rew_2, 0)
    end_time_2 = time.time()
    
    secs_2 = end_time_2 - start_time_2
    mins_2 = int(secs_2 // 60)
    secs_2 = secs_2 % 60
    print(f'total amount of time for 1 update after jit compilation: {mins_2} mins & {secs_2} secs.')

def test_sampler_dataloading_tf(d4rl=True, byol=True):
    '''Testing dataloading across epochs.'''
    if d4rl:
        if byol:
            buffer = D4RLSequenceReplayBuffer('hopper', 'medium', 10)
            dataloader = byol_sampling_dataloader_tf(buffer, max_steps=10000, batch_size=50)
        else:
            buffer = D4RLTransitionReplayBuffer('hopper', 'medium')
            dataloader = rnd_sampling_dataloader_tf(buffer, max_steps=200, batch_size=20)
    else:
        data_path = Path('./offline_data/dmc/cheetah_run/med_exp')
        if byol:
            buffer = VD4RLSequenceReplayBuffer(data_path, 10, frame_stack=3)
            dataloader = byol_sampling_dataloader_tf(buffer, max_steps=10000, batch_size=50)
        else:
            buffer = VD4RLTransitionReplayBuffer(data_path, frame_stack=3)
            dataloader = rnd_sampling_dataloader_tf(buffer, max_steps=200, batch_size=20)
    
    for epoch in range(1000):
        count = 0
        print(f'starting epoch {epoch + 1}')
        print('*' * 50)
        o_shape, a_shape, r_shape, no_shape, d_shape = None, None, None, None, None
        for batch in dataloader:
            obs, action, reward, next_obs, done = batch
            
            if count == 0:
                print(f'obs info: shape: {obs.shape}, dtype: {obs.dtype}')
                print(f'action info: shape: {action.shape}, dtype: {action.dtype}')
                print(f'reward info: shape: {reward.shape}, dtype: {reward.dtype}')
                print(f'next obs info: shape: {next_obs.shape}, dtype: {next_obs.dtype}')
                print(f'done info: shape: {done.shape}, dtype: {done.dtype}')
                print('*' * 50)
                
                o_shape = obs.shape
                a_shape = action.shape
                r_shape = reward.shape
                no_shape = next_obs.shape
                d_shape = done.shape
            else:
                assert obs.shape == o_shape
                assert action.shape == a_shape
                assert reward.shape == r_shape
                assert next_obs.shape == no_shape
                assert done.shape == d_shape
            
            count += 1
            
def test_byol_fn_dataloading():
    data_dir = Path('./offline_data/dmc/cheetah_run/med_exp')
    dset = byol_fn_dataloader_tf(data_dir, max_steps=10000, batch_size=50, seq_len=10)
    
    o_shape, a_shape, r_shape, no_shape, d_shape = None, None, None, None, None

    count = 0
    for batch in dset:
        o, a, r, no, d = batch
        if count == 0:
            print(o.shape)
            print(a.shape)
            print(r.shape)
            print(no.shape)
            print(d.shape)
            
            o_shape = o.shape
            a_shape = a.shape
            r_shape = r.shape
            no_shape = no.shape
            d_shape = d.shape
        else:
            assert o.shape == o_shape
            assert a.shape == a_shape
            assert r.shape == r_shape
            assert no.shape == no_shape
            assert d.shape == d_shape
        
        count += 1
    
    print(f'total number of batches: {count}')

def test_sampler_dataloading_torch():
    print('=== Testing dataloading with PyTorch ===')
    data_path = Path('./offline_data/dmc/cheetah_run/med_exp')
    dloader = byol_sampling_dataloader_torch(data_path, seq_len=10, max_steps=10000, batch_size=50, frame_stack=3)
    
    print('*' * 30)
    for _ in range(10):
        batch = next(iter(dloader))
        
        obs, action, reward, next_obs, done = batch
        print(f'obs shape: {obs.shape}')
        print(f'action shape: {action.shape}')
        print(f'reward shape: {reward.shape}')
        print(f'next_obs shape: {next_obs.shape}')
        print(f'done shape: {done.shape}')
        print('*' * 30)
        
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
    loader, _ = d4rl_rnd_iterative_dataloader(name, capability, batch_size=1024, normalize=True)
    
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
    model_path = f'/home/ds844/byol-offline/pretrained_models/{"byol" if byol else "rnd"}/hopper/medium/{action_dir + "/" if not byol else ""}{"byol" if byol else "rnd"}_1000.pkl'
    print(f'Model path: {model_path}')
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
    if byol:
        random_obs, random_actions = jnp.expand_dims(random_obs, 0), jnp.expand_dims(random_actions, 0) # (1, N, dim)
        medium_obs, medium_actions = jnp.expand_dims(medium_obs, 0), jnp.expand_dims(medium_actions, 0)
        expert_obs, expert_actions = jnp.expand_dims(expert_obs, 0), jnp.expand_dims(expert_actions, 0)
    
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
    # test_world_model_module()
    # test_world_model_update()
    test_rssm()
    # =========================
    # test_get_test_traj()
    # =========================
    # test_sampler_dataloading_tf(d4rl=False, byol=True)
    # test_sampler_dataloading_torch()