import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import d4rl
import jax.numpy as jnp
import numpy as np

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

class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until

class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False
    
class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)

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

def batched_zeros_like(shape):
    if type(shape) == int:
        return jnp.zeros((1, shape))
    return jnp.zeros((1,) + tuple(shape))

def seq_batched_zeros_like(shape):
    if type(shape) == int:
        return jnp.zeros((2, 1, shape))
    return jnp.zeros((2, 1) + tuple(shape))

def print_dict(log_dict):
    for k, v in log_dict.items():
        print(f'{k}: {v}')

# ================ GYM UTILS ================

MUJOCO_ENVS = {
    'halfcheetah': 'HalfCheetah-v2',
    'ant': 'Ant-v2',
    'hopper': 'Hopper-v2',
    'walker2d': 'Walker2d-v2'
}

ATARI_ENVS = {
    'pong': 'PongNoFrameskip-v4',
    'breakout': 'BreakoutNoFrameskip-v4',
    'beamrider': 'BeamRiderNoFrameskip-v4',
    'qbert': 'QbertNoFrameskip-v4',
    'montezuma': 'MontezumaRevengeNoFrameskip-v4'
}

LEVELS = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']

def make_gym_env(name, capability):
    '''Makes training/eval envs for D4RL training/evaluation.'''
    env_name = name + '-' + capability + '-v2'
    return gym.make(env_name)

def make_atari_env(name, grayscale=True):
    assert name in ATARI_ENVS, "Not an Atari env!"
    name = ATARI_ENVS[name]
    env = gym.make(name)
    # now need to apply atari wrapper there
    env = AtariPreprocessing(env, grayscale_obs=grayscale)
    return env
    
def get_gym_dataset(name, capability, q_learning=True):
    '''Gets dataset associated with env and level.'''
    assert capability in LEVELS, "Not a proper level -- can't load the dataset."
    env_name = name + '-' + capability + '-v2'
    env = gym.make(env_name)
    if q_learning:
        return d4rl.qlearning_dataset(env)
    return env.get_dataset()