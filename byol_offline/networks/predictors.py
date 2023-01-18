import jax
import jax.numpy as jnp
import haiku as hk

class BYOLPredictor(hk.Module):
    '''BYOL predictor.'''
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
    
    def __call__(self, obs):
        assert jnp.ndim(obs) == 2, 'not the right shape (no sequence should be there when processing).'
        out = hk.Linear(1024)(obs)
        out = jax.nn.relu(out)
        out = hk.Linear(self.repr_dim)(out)
        return out
    
class RNDPredictor(hk.Module):
    '''RND predictor.'''
    def __init__(self, cfg):
        super().__init__()
        weight_init = hk.initializers.UniformScaling(0.333)
        
        self.predictor = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=weight_init),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jnp.tanh,
            hk.nets.MLP([cfg.hidden_dim, cfg.repr_dim],
                         w_init=weight_init,
                         activation=jax.nn.elu,
                         activate_final=False)
        ])
        
    def __call__(self, obs_repr):
        prediction = self.predictor(obs_repr)
        return prediction