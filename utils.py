import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import d4rl

import chex
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Union, Mapping

"""Basic utilities."""


def get_random_traj(path):
    traj_fns = list(path.glob('*.npz'))
    traj_fn = np.random.choice(traj_fns)
    
    with traj_fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in ['image', 'action']}
        return episode


def get_test_traj(path, frame_stack=3, seq_len=10):
    episode = get_random_traj(path)
    
    episode_len = episode['image'].shape[0] - 1 # should be 501 - 1 = 500
    start_idx = np.random.randint(0, episode_len - seq_len)
    
    obs = []
    actions = []
    first_frames = []
    for i in range(start_idx, start_idx + seq_len):
        # current obs
        if i < frame_stack - 1:
            repeat = frame_stack - i
            first_frame = episode["image"][0]
            other_frames = [episode['image'][j] for j in range(1, i+1)] # i frames exactly
            ob = np.concatenate([first_frame] * repeat + other_frames, axis=-1).astype(np.float32) # (H, W, C * frame_stack)
        else:
            ob = [episode["image"][i - x] for x in reversed(range(frame_stack))]
            ob = np.concatenate(ob, -1).astype(np.float32)
            first_frame = episode["image"][i - frame_stack + 1]
        
        obs.append(ob)
        first_frames.append(first_frame)
        
        action = episode['action'][i].astype(np.float32)
        actions.append(action)
        
    obs = np.expand_dims(np.stack(obs), 1) # (T, 1, 64, 64, 9)
    actions = np.expand_dims(np.stack(actions), 1) # (T, 1, 6)
    first_frames = np.stack(first_frames).astype(np.uint8) # should be uint8 here, (T, 64, 64, 3)
    return obs, actions, first_frames


def batch_device_put(batch: Tuple) -> Tuple:
    map_fn = partial(jax.device_put, device=jax.devices()[0])
    return (map_fn(elt) for elt in batch)


def flatten_data(transitions):
    '''In case of sequence data, we flatten across sequence length and batch size dimension.'''
    def flatten(arr):
        new_arr = jnp.reshape(arr, (-1,) + arr.shape[2:])
        return new_arr
    
    obs, actions, rewards, next_obs, dones = transitions.obs, transitions.actions, transitions.rewards, transitions.next_obs, transitions.dones
    obs = flatten(obs)
    actions = flatten(actions)
    rewards = flatten(rewards)
    next_obs = flatten(next_obs)
    dones = flatten(dones)

    transitions = transitions._replace(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones
    )
    return transitions


def seq_batched_like(array: chex.Array) -> chex.Array:
    """Returns a sequence array similar to the one provided that is necessary for batching."""
    
    array_expanded = jnp.expand_dims(array, 0)
    array_expanded = jnp.expand_dims(array_expanded, 0)
    
    array_expanded = jnp.tile(array_expanded, (2,) + (1,) * (array_expanded.ndim - 1))
    return array_expanded


def print_dict(log_dict: Mapping) -> None:
    for k, v in log_dict.items():
        print(f'{k}: {v}')

# ================ GYM UTILS ================

def is_pixel_based(observation_space: gym.Space) -> bool:
    """Determines whether an environment is pixel-based from its observation spec."""
    
    return isinstance(observation_space, gym.spaces.Dict) and "pixels" in list(observation_space.keys())