import jax
import jax.numpy as jnp
import haiku as hk

class BYOLPredictor(hk.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
    
    def __call__(self, obs):
        assert jnp.ndim(obs) == 2, 'not the right shape (no sequence should be there when processing).'
        out = hk.Linear(1024)(obs)
        out = jax.nn.relu(out)
        out = hk.Linear(self.repr_dim)(out)
        return out