import gym
import d4rl
import jax.numpy as jnp

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

LEVELS = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']

def make_gym_env(name, capability):
    '''Makes training/eval envs for D4RL training/evaluation.'''
    env_name = name + '-' + capability + '-v2'
    return gym.make(env_name)

def get_gym_dataset(name, capability, q_learning=True):
    '''Gets dataset associated with env and level.'''
    assert capability in LEVELS, "Not a proper level -- can't load the dataset."
    env_name = name + '-' + capability + '-v2'
    env = gym.make(env_name)
    if q_learning:
        return d4rl.qlearning_dataset(env)
    return env.get_dataset()