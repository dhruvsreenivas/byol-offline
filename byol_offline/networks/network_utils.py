import jax
import jax.numpy as jnp
from jax import lax
import haiku as hk
import distrax

class ClippedNormal(distrax.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale)
        self.low = low
        self.high = high
        self.eps = eps
    
    def _clamp(self, x):
        clamped_x = jnp.clip(x, self.low + self.eps, self.high - self.eps)
        x = x - lax.stop_gradient(x) + lax.stop_gradient(clamped_x)
        return x

    def sample(self, seed, clip=None, sample_shape=()):
        sample_shape = sample_shape + self.batch_shape + self.event_shape
        eps = jax.random.normal(seed, shape=sample_shape)
        eps *= self.scale
        
        if clip is not None:
            eps = jnp.clip(eps, -clip, clip)
        
        x = self.loc + eps
        return self._clamp(x)
    
def squashed_normal_dist(loc, scale):
    # TODO: find out whether to do normal vs multivariate normal diag
    return distrax.Transformed(distrax.Normal(loc, scale), distrax.Block(distrax.Tanh(), ndims=1))

INITIALIZERS = {
    'xavier_uniform': hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    'conv2d_orthogonal': hk.initializers.Orthogonal(scale=jnp.sqrt(2)),
    'linear_orthogonal': hk.initializers.Orthogonal(scale=1.0),
    'zeros': hk.initializers.Constant(0.0)
}
        
        
        