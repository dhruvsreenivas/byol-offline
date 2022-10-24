import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import haiku as hk
import hydra
from pathlib import Path

from byol_offline.models import *
from memory.replay_buffer import *

'''Various testing to make sure core machinery works.'''

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
            
def test_iterative_dataloading():
    name = 'hopper'
    capability = 'medium'
    loader = rnd_iterative_dataloader(name, capability, batch_size=128)

    for batch in loader:
        print('*' * 50)
        obs, action, reward, next_obs, done = batch
        print(f'obs info: shape: {obs.shape}, dtype: {obs.dtype}')
        print(f'action info: shape: {action.shape}, dtype: {action.dtype}')
        print(f'reward info: shape: {reward.shape}, dtype: {reward.dtype}')
        print(f'next obs info: shape: {next_obs.shape}, dtype: {next_obs.dtype}')
        print(f'done info: shape: {done.shape}, dtype: {done.dtype}')
        
if __name__ == '__main__':
    test_iterative_dataloading()