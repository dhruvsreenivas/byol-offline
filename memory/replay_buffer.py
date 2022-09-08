import numpy as np
import io
import traceback
import random
from utils import get_gym_dataset
from collections import namedtuple

# tensorflow dataset utilities
import tensorflow as tf

Transition = namedtuple('Transition', ['obs', 'actions', 'rewards', 'next_obs', 'dones'])

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def chunk_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0]

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())
            
def load_episode(fn, relevant_keys):
    # Loads episode and only grabs relevant
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in relevant_keys}
        return episode

class VD4RLSequenceReplayBuffer:
    '''Replay buffer used to sample sequences of data from.'''
    def __init__(self, data_dir, seq_len):
        self._data_dir = data_dir
        self._seq_len = seq_len
        self._data_keys = ['image', 'action', 'reward', 'is_terminal']
        
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
        idx = np.random.randint(0, episode_len(episode) - self._seq_len)
        obs = episode["image"][idx : idx + self._seq_len]
        action = episode["action"][idx : idx + self._seq_len]
        reward = episode["reward"][idx : idx + self._seq_len]
        next_obs = episode["image"][idx + 1 : idx + self._seq_len + 1]
        done = episode["is_terminal"][idx : idx + self._seq_len]

        return obs, action, reward, next_obs, done
    
class VD4RLTransitionReplayBuffer:
    '''Replay buffer used to sample batches of arbitrary transitions.'''
    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._data_keys = ['image', 'action', 'reward', 'is_terminal']
        
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
        obs = episode["image"][idx]
        action = episode['action'][idx]
        reward = episode['reward'][idx]
        next_obs = episode['image'][idx + 1]
        done = episode['is_terminal'][idx]
        
        return obs, action, reward, next_obs, done
    
class D4RLSequenceReplayBuffer:
    def __init__(self, env_name, capability, seq_len):
        self.dataset = get_gym_dataset(env_name, capability, q_learning=True)
        self.n_samples = self.dataset['observations'].shape[0]
        self._seq_len = seq_len
        
    def _sample(self):
        # sampling full sequence
        idx = np.random.randint(0, self.n_samples - self._seq_len) # TODO: figure out how to pad for more data
        obs = self.dataset['observations'][idx : idx + self._seq_len]
        action = self.dataset['actions'][idx : idx + self._seq_len]
        reward = self.dataset['rewards'][idx : idx + self._seq_len]
        next_obs = self.dataset['next_observations'][idx : idx + self._seq_len]
        done = self.dataset['terminals'][idx : idx + self._seq_len]
        
        return obs, action, reward, next_obs, done

class D4RLTransitionReplayBuffer:
    def __init__(self, env_name, capability):
        self.dataset = get_gym_dataset(env_name, capability, q_learning=True)
        self.n_samples = self.dataset['observations'].shape[0]
        
    def _sample(self):
        # sampling single item
        idx = np.random.randint(0, self.n_samples) # TODO: figure out how to pad for more data
        obs = self.dataset['observations'][idx]
        action = self.dataset['actions'][idx]
        reward = self.dataset['rewards'][idx]
        next_obs = self.dataset['next_observations'][idx]
        done = self.dataset['terminals'][idx]
        
        return obs, action, reward, next_obs, done
        
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

def byol_dataloader(buffer: VD4RLSequenceReplayBuffer or D4RLSequenceReplayBuffer,
                     max_steps,
                     batch_size,
                     prefetch=True):
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
        dataset = dataset.prefetch(10)
    
    return dataset.as_numpy_iterator()

def rnd_dataloader(buffer: VD4RLTransitionReplayBuffer or D4RLTransitionReplayBuffer,
                   max_steps,
                   batch_size,
                   prefetch=True):
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
        dataset = dataset.prefetch(10)
    
    return dataset.as_numpy_iterator()

if __name__ == '__main__':
    # from pathlib import Path
    # data_dir = Path('../offline_data/cheetah_run/med_exp')
    seq_len = 25
    
    rb = D4RLSequenceReplayBuffer('hopper', 'medium', seq_len)
    print('rb created')
    dataloader = byol_dataloader(rb, max_steps=200, batch_size=20)
    print('dataloader created')
    
    batch = next(dataloader)
    print(type(batch))
    obs, action = batch
    print(obs.shape, type(obs), obs.dtype)
    print(action.shape, type(action), action.dtype)
    exit()