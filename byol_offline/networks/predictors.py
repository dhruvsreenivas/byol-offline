import chex
import haiku as hk
import jax
import jax.numpy as jnp

"""Various predictor modules for bonus computation."""

class BYOLPredictor(hk.Module):
    """BYOL predictor for visual continuous control."""
    
    def __init__(self, repr_dim: int):
        super().__init__()
        self._repr_dim = repr_dim
    
    def __call__(self, obs: chex.Array) -> chex.Array:
        
        out = hk.Linear(1024)(obs)
        out = jax.nn.relu(out)
        out = hk.Linear(self._repr_dim)(out)
        return out

    
class RNDPredictor(hk.Module):
    """RND predictor for continuous control."""
    
    def __init__(self, hidden_dim: int, repr_dim: int):
        super().__init__()
        
        weight_init = hk.initializers.UniformScaling(0.333)
        
        self._predictor = hk.Sequential([
            hk.Linear(hidden_dim, w_init=weight_init),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jnp.tanh,
            
            hk.nets.MLP(
                [hidden_dim, repr_dim],
                w_init=weight_init,
                activation=jax.nn.elu,
                activate_final=False
            )
        ])
        
    def __call__(self, obs_repr: chex.Array) -> chex.Array:
        prediction = self._predictor(obs_repr)
        return prediction