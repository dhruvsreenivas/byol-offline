import jax
import jax.numpy as jnp
import haiku as hk
import optax

class RMS(object):
    """running mean and std """
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = jnp.zeros(shape)
        self.S = jnp.ones(shape)
        self.n = epsilon

    def __call__(self, x):
        bs = x.shape[0]
        delta = jnp.mean(x, axis=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + jnp.var(x, axis=0) * bs +
                 jnp.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
    
def scale_pen(reward_pen: jnp.ndarray, min_rew: float=-1, max_rew: float=1):
    '''Scale reward penalty to be in [-1, 1].'''
    max_pen, min_pen = jnp.max(reward_pen), jnp.min(reward_pen)
    diff = max_pen - min_pen + 1e-8
    scaled_pen = (max_rew - min_rew) * (reward_pen - min_pen) / diff + min_rew
    return scaled_pen

def q_scaling_factor(q_values: jnp.ndarray, lmbda: float):
    '''Q scaling factor from TD3+BC: https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py#L141'''
    factor = lmbda / jnp.mean(jnp.abs(q_values))
    return jax.lax.stop_gradient(factor)
    
def get_penalized_rewards(rewards: jnp.ndarray, reward_pen: jnp.ndarray, reward_lambda: float, min_rew: float=-1, max_rew: float=1):
    '''Scaling and clipping pessimism bonus.'''
    scaled_pen = scale_pen(reward_pen, min_rew, max_rew)
    unclipped_rewards = rewards - reward_lambda * scaled_pen
    return jnp.clip(unclipped_rewards, min_rew, max_rew)
    
def update_target(params: hk.Params, target_params: hk.Params, ema: float):
    target_params = optax.incremental_update(params, target_params, step_size=ema)
    return target_params