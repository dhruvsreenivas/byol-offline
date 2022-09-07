import gym
import d4rl

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

# ================ GYM UTILS ================

MUJOCO_ENVS = {
    'halfcheetah': 'HalfCheetah-v2',
    'ant': 'Ant-v2',
    'hopper': 'Hopper-v2',
    'walker2d': 'Walker2d-v2'
}

LEVELS = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']

def make_gym_env(name):
    env_name = MUJOCO_ENVS[name]
    return gym.make(env_name)

def get_gym_dataset(name, capability, q_learning=True):
    assert capability in LEVELS, "Not a proper level -- can't load the dataset."
    env_name = name + '-' + capability + '-v2'
    env = gym.make(env_name)
    if q_learning:
        return d4rl.qlearning_dataset(env)
    return env.get_dataset()