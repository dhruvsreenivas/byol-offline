import jax
import jax.numpy as jnp
import haiku as hk

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
    
def update_target(params: hk.Params, target_params: hk.Params, ema: float):
    target_params = jax.tree_util.tree_map(lambda x, y: ema * x + (1.0 - ema) * y, params, target_params)
    return target_params