import distrax
import chex
import jax
import jax.numpy as jnp
from typing import Optional

from byol_offline.types import Shape

"""Distributions in JAX."""

class TanhTransformed(distrax.Transformed):
    """Modified version of the tanh transformed distribution from Distrax, to allow for mode computation."""
    
    def __init__(self, distribution: distrax.Distribution) -> None:
        bijector = distrax.Block(distrax.Tanh(), 1)
        super().__init__(distribution, bijector)
        
    def mode(self) -> chex.Array:
        return self.bijector.forward(self.distribution.mode())
    
    
class ClippedNormal(distrax.Normal):
    """Clipped Gaussian distribution."""
    
    def __init__(
        self, loc: chex.Array, scale: chex.Array,
        low: float = -1.0, high: float = 1.0, eps: float = 1e-6
    ):
        super().__init__(loc, scale)
        self.low = low
        self.high = high
        self.eps = eps
    
    def _clamp(self, x: chex.Array):
        clamped_x = jnp.clip(x, self.low + self.eps, self.high - self.eps)
        x = x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped_x)
        return x

    def sample(self, seed: chex.PRNGKey, clip: Optional[float] = None, sample_shape: Shape = ()):
        sample_shape = sample_shape + self.batch_shape + self.event_shape
        eps = jax.random.normal(seed, shape=sample_shape)
        eps *= self.scale
        
        if clip is not None:
            eps = jnp.clip(eps, -clip, clip)
        
        x = self.loc + eps
        return self._clamp(x)