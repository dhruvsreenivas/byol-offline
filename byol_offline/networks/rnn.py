import jax.numpy as jnp
import haiku as hk

class ClosedLoopGRU(hk.GRU):
    def __init__(self, hidden_size):
        # zero out everything
        super().__init__(
            hidden_size=hidden_size,
            w_i_init=jnp.zeros,
            w_h_init=jnp.zeros,
            b_init=jnp.zeros
        )
        
    def __call__(self, obs, action, state):
        inputs = jnp.concatenate([obs, action], -1)
        return super().__call__(inputs, state)
        
        
        