import numpy as np
import io
import traceback
import random
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

class SequenceReplayBuffer:
    '''Replay buffer used to sample sequences of data from.'''
    def __init__(self, data_dir, seq_len):
        self._data_dir = data_dir
        self._seq_len = seq_len
        self._data_keys = ['image', 'action']
        
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
        idx = np.random.randint(0, episode_len(episode) - self._seq_len + 1)
        obs = episode["image"][idx : idx + self._seq_len]
        action = episode["action"][idx : idx + self._seq_len]
        
        return obs, action
        
def generator_fn(buffer: SequenceReplayBuffer, max_steps=None):
    if max_steps is None:
        for _ in range(max_steps):
            yield buffer._sample()
    else:
        while True:
            yield buffer._sample()
            
def transpose_fn(obs, action):
    '''Switches from (B, T, H, W, C) to (T, B, H, W, C).'''
    obs = tf.transpose(obs, (1, 0, 2, 3, 4))
    action = tf.transpose(action, (1, 0, 2))
    return obs, action

def model_dataloader(buffer: SequenceReplayBuffer,
                     max_steps,
                     batch_size,
                     prefetch=True):
    obs, action = buffer._sample()
    obs_type, action_type = obs.dtype, action.dtype
    obs_shape, action_shape = obs.shape, action.shape
    
    generator = lambda: generator_fn(buffer, max_steps)
    output_sig = (tf.TensorSpec(shape=obs_shape, dtype=obs_type), tf.TensorSpec(shape=action_shape, dtype=action_type))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(transpose_fn)
    if prefetch:
        dataset = dataset.prefetch(10)
    
    return dataset.as_numpy_iterator()

class PolicyReplayBuffer:
    '''Replay buffer similar to DrQv2 replay buffer.'''
    def __init__(self, capacity, obs_shape, action_shape):
        self._capacity = capacity
        
        self.obs = np.empty((self._capacity, *obs_shape))
        self.actions = np.empty((self._capacity, *action_shape))
        self.rewards = np.empty((self._capacity, 1))
        self.next_obs = np.empty((self._capacity, *obs_shape))
        self.dones = np.empty((self._capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.n = 0
    
    def add(self, ob, act, rew, n_ob, done):
        self.obs[self.idx] = ob
        self.actions[self.idx] = act
        self.rewards[self.idx] = rew
        self.next_obs[self.idx] = n_ob
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self._capacity
        self.n = min(self.n + 1, self._capacity)
        
    def sample(self, batch_size):
        idxes = np.random.randint(0, self.n, size=batch_size)
        
        obs = self.obs[idxes]
        actions = self.actions[idxes]
        rewards = self.rewards[idxes]
        next_obs = self.next_obs[idxes]
        dones = self.dones[idxes]
        
        return Transition(obs, actions, rewards, next_obs, dones)

if __name__ == '__main__':
    from pathlib import Path
    
    data_dir = Path('../offline_data/cheetah_run/med_exp')
    seq_len = 50
    
    rb = SequenceReplayBuffer(data_dir, seq_len)
    print('rb created')
    dataloader = model_dataloader(rb, max_steps=200, batch_size=20)
    print('dataloader created')
    
    batch = next(dataloader)
    print(type(batch))
    obs, action = batch
    print(obs.shape, type(obs), obs.dtype)
    print(action.shape, type(action), action.dtype)
    exit()