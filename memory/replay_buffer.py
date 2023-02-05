import numpy as np
import traceback
import random
from utils import get_gym_dataset
from collections import namedtuple
from typing import Union

# tensorflow dataset utilities
import tensorflow as tf
import tensorflow_datasets as tfds

Transition = namedtuple('Transition', ['obs', 'actions', 'rewards', 'next_obs', 'dones'])

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1
            
def load_episode(fn, relevant_keys):
    # Loads episode and only grabs relevant info
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in relevant_keys}
        return episode
    
def get_dataset_size(dataset):
    return dataset['observations'].shape[0]

def get_mujoco_dataset_transformations(dataset):
    '''Get normalization constants for the D4RL dataset we are working with.'''
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_observations = dataset["next_observations"]

    state_mean = np.mean(observations, axis=0).astype(np.float32)
    action_mean = np.mean(actions, axis=0).astype(np.float32)
    next_state_mean = np.mean(next_observations, axis=0).astype(np.float32)

    state_scale = np.abs(observations - state_mean).mean(axis=0).astype(np.float32) + 1e-8
    action_scale = np.abs(actions - action_mean).mean(axis=0).astype(np.float32) + 1e-8
    next_state_scale = np.abs(next_observations - next_state_mean).mean(axis=0).astype(np.float32) + 1e-8
    
    # reward max + min
    reward_min = np.min(rewards)
    reward_max = np.max(rewards)
    reward_info = (reward_min, reward_max)

    return state_mean, action_mean, next_state_mean, state_scale, action_scale, next_state_scale, reward_info

def normalize_sa(states, actions, stats):
    state_mean, action_mean, _, state_scale, action_scale, _, _ = stats
    states = (states - state_mean) / state_scale
    actions = (actions - action_mean) / action_scale
    return states, actions

def normalize_rewards_d4rl(dataset):
    '''Returns dataset with normalized rewards.'''
    rewards = dataset['rewards']
    reward_mean, reward_std = np.mean(rewards), np.std(rewards)
    normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-8)
    dataset['rewards'] = normalized_rewards

    return dataset

def clip_actions_d4rl(dataset, eps):
    '''Clips actions in D4RL datasets, as in https://github.com/Div99/XQL/blob/main/offline/dataset_utils.py.'''
    lim = 1 - eps
    dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
    return dataset

def relabel_dones_d4rl(dataset):
    '''Relabels terminals in D4RL dataset, in the case where the next ob and ob differ by a lot.'''
    dones_float = np.zeros_like(dataset['rewards'])
    dones_float[-1] = 1
    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    
    dataset['terminals'] = dones_float
    return dataset

class VD4RLSequenceReplayBuffer:
    '''Replay buffer used to sample sequences of data from.'''
    def __init__(self, data_dir, seq_len, frame_stack=3):
        self._data_dir = data_dir
        self._seq_len = seq_len
        self._data_keys = ['image', 'action', 'reward', 'is_terminal']
        self._frame_stack = frame_stack
        
        # filenames
        self._episode_fns = []
        self._episodes_cache = {}
        
        try:
            self._try_fetch()
        except Exception:
            print('could not fetch all episodes')
            traceback.print_exc()
            
    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes_cache[eps_fn]
    
    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn, self._data_keys)
        except Exception:
            print('could not load episode!')
            return False
        
        self._episode_fns.append(eps_fn)
        self._episodes_cache[eps_fn] = episode
        return True
    
    def _try_fetch(self):
        eps_fns = self._data_dir.glob('*.npz')
        for eps_fn in eps_fns:
            lidx = eps_fn.stem.index('_')
            idx = int(eps_fn.stem[lidx+1:])
            
            if eps_fn in self._episodes_cache.keys():
                continue
            if not self._store_episode(eps_fn):
                print(f'could not load episode {idx}')
                continue
    
    def _sample(self):
        # observations are (T, H, W, C * frame_stack)
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._seq_len)
        
        obs = []
        actions = []
        rewards = []
        next_obs = []
        dones = []
        for i in range(idx, idx + self._seq_len):
            # current obs
            if i < self._frame_stack - 1:
                repeat = self._frame_stack - i
                first_frame = episode["image"][0]
                other_frames = [episode['image'][j] for j in range(1, i+1)] # i frames exactly
                ob = np.concatenate([first_frame] * repeat + other_frames, axis=-1).astype(np.float32) # (H, W, C * frame_stack)
            else:
                ob = [episode["image"][i - x] for x in reversed(range(self._frame_stack))]
                ob = np.concatenate(ob, -1).astype(np.float32)
            
            # current next obs
            if i + 1 < self._frame_stack - 1:
                repeat = self._frame_stack - (i + 1)
                first_frame = episode['image'][0]
                other_next_frames = [episode['image'][j] for j in range(1, i+2)] # i + 1 frames exactly
                next_ob = np.concatenate([first_frame] * repeat + other_next_frames, axis=-1).astype(np.float32)
            else:
                next_ob = [episode["image"][i - x + 1] for x in reversed(range(self._frame_stack))]
                next_ob = np.concatenate(next_ob, -1).astype(np.float32)
            
            obs.append(ob)
            next_obs.append(next_ob)
            
            action = episode['action'][i].astype(np.float32)
            reward = episode['reward'][i].astype(np.float32)
            done = episode['is_terminal'][i].astype(np.float32)
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        obs = np.stack(obs)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_obs = np.stack(next_obs)
        dones = np.stack(dones)

        return obs, actions, rewards, next_obs, dones
    
class VD4RLTransitionReplayBuffer:
    '''Replay buffer used to sample batches of arbitrary transitions.'''
    def __init__(self, data_dir, frame_stack=3):
        self._data_dir = data_dir
        self._data_keys = ['image', 'action', 'reward', 'is_terminal']
        self._frame_stack = frame_stack
        
        # filenames
        self._episode_fns = []
        self._episodes_cache = {}
        
        try:
            self._try_fetch()
        except Exception:
            print('could not fetch all episodes')
            traceback.print_exc()
            
    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes_cache[eps_fn]
    
    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn, self._data_keys)
        except Exception:
            return False
        
        self._episode_fns.append(eps_fn)
        self._episodes_cache[eps_fn] = episode
        return True
    
    def _try_fetch(self):
        eps_fns = self._data_dir.glob('*.npz')
        for eps_fn in eps_fns:
            lidx = eps_fn.stem.index('_')
            idx = int(eps_fn.stem[lidx+1:])
            
            if eps_fn in self._episodes_cache.keys():
                continue
            if not self._store_episode(eps_fn):
                print(f'could not load episode {idx}')
                continue
    
    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - 1)
        if idx < self._frame_stack - 1:
            repeat = self._frame_stack - idx
            first_frame = episode['image'][0]
            curr_frame = episode['image'][idx]
            obs = np.concatenate([first_frame] * repeat + [curr_frame], axis=-1).astype(np.float32)
        else:
            obs = episode['image'][idx - self._frame_stack + 1 : idx + 1].astype(np.float32)
            
        if idx + 1 < self._frame_stack - 1:
            repeat = self._frame_stack - (idx + 1)
            first_frame = episode['image'][0]
            curr_next_frame = episode['image'][idx + 1]
            next_obs = np.concatenate([first_frame] * repeat + [curr_next_frame], axis=-1).astype(np.float32)
        else:
            next_obs = episode['image'][idx - self._frame_stack + 2 : idx + 2].astype(np.float32)
        
        action = episode['action'][idx]
        reward = episode['reward'][idx]
        done = episode['is_terminal'][idx]
        
        return obs, action, reward, next_obs, done
    
class D4RLSequenceReplayBuffer:
    def __init__(self, env_name, capability, seq_len, clip_actions=True, normalize=True, normalize_reward=False):
        self.dataset = get_gym_dataset(env_name, capability, q_learning=True)
        self.n_samples = self.dataset['observations'].shape[0]
        self._seq_len = seq_len
        
        if clip_actions:
            assert not normalize, "don't clip when you are normalizing anyway."
            eps = 1e-5
            self.dataset = clip_actions_d4rl(self.dataset, eps)

        self.normalize = normalize
        if normalize:
            means_scales = get_mujoco_dataset_transformations(self.dataset)
            self.state_mean, self.action_mean, self.next_state_mean, self.state_scale, self.action_scale, self.next_state_scale, reward_info = means_scales
            self.reward_min = reward_info[0]
            self.reward_max = reward_info[1]
        
        if normalize_reward:
            self.dataset = normalize_rewards_d4rl(self.dataset)
        
    def _sample(self):
        # sampling full sequence
        idx = np.random.randint(0, self.n_samples - self._seq_len) # TODO: figure out how to pad for more data
        obs = self.dataset['observations'][idx : idx + self._seq_len]
        action = self.dataset['actions'][idx : idx + self._seq_len]
        reward = self.dataset['rewards'][idx : idx + self._seq_len]
        next_obs = self.dataset['next_observations'][idx : idx + self._seq_len]
        done = self.dataset['terminals'][idx : idx + self._seq_len]

        # normalize if specified
        if self.normalize:
            obs = (obs - self.state_mean) / self.state_scale
            action = (action - self.action_mean) / self.action_scale
            next_obs = (next_obs - self.next_state_mean) / self.next_state_scale
        
        return obs, action, reward, next_obs, done

class D4RLTransitionReplayBuffer:
    def __init__(self, env_name, capability, normalize=True, normalize_reward=False):
        self.dataset = get_gym_dataset(env_name, capability, q_learning=True)
        self.n_samples = self.dataset['observations'].shape[0]

        self.normalize = normalize
        if normalize:
            means_scales = get_mujoco_dataset_transformations(self.dataset)
            self.state_mean, self.action_mean, self.next_state_mean, self.state_scale, self.action_scale, self.next_state_scale, reward_info = means_scales
            self.reward_min = reward_info[0]
            self.reward_max = reward_info[1]
            
        if normalize_reward:
            self.dataset = normalize_rewards_d4rl(self.dataset)
        
    def _sample(self):
        # sampling single item
        idx = np.random.randint(0, self.n_samples)
        obs = self.dataset['observations'][idx]
        action = self.dataset['actions'][idx]
        reward = self.dataset['rewards'][idx]
        next_obs = self.dataset['next_observations'][idx]
        done = self.dataset['terminals'][idx]

        # normalize if specified
        if self.normalize:
            obs = (obs - self.state_mean) / self.state_scale
            action = (action - self.action_mean) / self.action_scale
            next_obs = (next_obs - self.next_state_mean) / self.next_state_scale
        
        return obs, action, reward, next_obs, done, idx
        
def generator_fn(buffer, max_steps=None):
    if max_steps is not None:
        for _ in range(max_steps):
            yield buffer._sample()
    else:
        while True:
            yield buffer._sample()
            
def transpose_fn_img(obs, action, reward, next_obs, done):
    '''Switches from (B, T, ...) to (T, B, ...), with img observations.'''
    obs = tf.transpose(obs, (1, 0, 2, 3, 4))
    action = tf.transpose(action, (1, 0, 2))
    reward = tf.transpose(reward, (1, 0))
    next_obs = tf.transpose(next_obs, (1, 0, 2, 3, 4))
    done = tf.transpose(done, (1, 0))
    return obs, action, reward, next_obs, done

def transpose_fn_state(obs, action, reward, next_obs, done):
    '''Switches from (B, T, ...) to (T, B, ...), with state vector observations.'''
    obs = tf.transpose(obs, (1, 0, 2))
    action = tf.transpose(action, (1, 0, 2))
    reward = tf.transpose(reward, (1, 0))
    next_obs = tf.transpose(next_obs, (1, 0, 2))
    done = tf.transpose(done, (1, 0))
    return obs, action, reward, next_obs, done

def byol_sampling_dataloader(buffer: Union[VD4RLSequenceReplayBuffer, D4RLSequenceReplayBuffer],
                             max_steps: int,
                             batch_size: int,
                             prefetch: bool = True):
    obs, action, reward, next_obs, done = buffer._sample()
    obs_type, action_type, reward_type, next_obs_type, done_type = obs.dtype, action.dtype, reward.dtype, next_obs.dtype, done.dtype
    obs_shape, action_shape, reward_shape, next_obs_shape, done_shape = obs.shape, action.shape, reward.shape, next_obs.shape, done.shape
    
    generator = lambda: generator_fn(buffer, max_steps)
    output_sig = (
        tf.TensorSpec(shape=obs_shape, dtype=obs_type),
        tf.TensorSpec(shape=action_shape, dtype=action_type),
        tf.TensorSpec(shape=reward_shape, dtype=reward_type),
        tf.TensorSpec(shape=next_obs_shape, dtype=next_obs_type),
        tf.TensorSpec(shape=done_shape, dtype=done_type)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if isinstance(buffer, VD4RLSequenceReplayBuffer):
        dataset = dataset.map(transpose_fn_img)
    else:
        dataset = dataset.map(transpose_fn_state)
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(dataset)

def rnd_sampling_dataloader(buffer: Union[VD4RLTransitionReplayBuffer, D4RLTransitionReplayBuffer],
                            max_steps: int,
                            batch_size: int,
                            prefetch: bool = True):
    
    obs, action, reward, next_obs, done = buffer._sample()
    done = np.float32(done)
    obs_type, action_type, reward_type, next_obs_type, done_type = obs.dtype, action.dtype, reward.dtype, next_obs.dtype, done.dtype
    obs_shape, action_shape, reward_shape, next_obs_shape, done_shape = obs.shape, action.shape, reward.shape, next_obs.shape, done.shape
    
    generator = lambda: generator_fn(buffer, max_steps)
    output_sig = (
        tf.TensorSpec(shape=obs_shape, dtype=obs_type),
        tf.TensorSpec(shape=action_shape, dtype=action_type),
        tf.TensorSpec(shape=reward_shape, dtype=reward_type),
        tf.TensorSpec(shape=next_obs_shape, dtype=next_obs_type),
        tf.TensorSpec(shape=done_shape, dtype=done_type),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(dataset)

def d4rl_dict_to_tuple(dict_batch):
    batch = []
    for k in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
        v = dict_batch[k]
        batch.append(tf.cast(v, dtype=tf.float32))
    return tuple(batch)

def d4rl_rnd_iterative_dataloader(dataset_name, dataset_capability, batch_size, normalize=True, state_only=True, prefetch=True):
    dataset = get_gym_dataset(dataset_name, dataset_capability)
    n_examples = get_dataset_size(dataset)
    
    if normalize:
        stats = get_mujoco_dataset_transformations(dataset)
    else:
        stats = None
    
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(d4rl_dict_to_tuple)
    
    if normalize:
        def normalize_map(state, action, reward, next_state, done):
            state_mean, action_mean, next_state_mean, state_scale, action_scale, next_state_scale, _ = stats
            
            normalized_state = (state - state_mean) / state_scale
            if state_only:
                normalized_action = action
            else:
                normalized_action = (action - action_mean) / action_scale
            normalized_next_state = (next_state - next_state_mean) / next_state_scale
            
            return normalized_state, normalized_action, reward, normalized_next_state, done
        dataset = dataset.map(normalize_map)
    
    dataset = dataset.shuffle(buffer_size=n_examples, reshuffle_each_iteration=True) # perfect shuffling
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(dataset), stats

class ReplayBuffer:
    '''Normal replay buffer, used in testing online RL algorithms.'''
    def __init__(self, max_size, state_dim, action_dim):
        self._max_size = max_size
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        self.setup()
        
    def setup(self):
        self._states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._actions = np.empty((self._max_size, self._action_dim), dtype=np.float32)
        self._rewards = np.empty((self._max_size,), dtype=np.float32)
        self._next_states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._dones = np.empty((self._max_size,), dtype=np.float32)
        
        self._n = 0
        self._p = 0
        
    def add(self, state, action, reward, next_state, done):
        self._states[self._p] = state
        self._actions[self._p] = action
        self._rewards[self._p] = reward
        self._next_states[self._p] = next_state
        self._dones[self._p] = np.float32(done)
        
        self._p = (self._p + 1) % self._max_size
        self._n = min(self._n + 1, self._max_size)
        
    def sample(self, batch_size):
        idxs = np.random.randint(0, self._n, size=batch_size)
        
        states = self._states[idxs]
        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        next_states = self._next_states[idxs]
        dones = self._dones[idxs]
        
        return Transition(states, actions, rewards, next_states, dones)
