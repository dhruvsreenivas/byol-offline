import jax
import jax.numpy as jnp
import haiku as hk

class RNDPredictor(hk.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        weight_init = hk.initializers.UniformScaling(0.333)
        self.predictor = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=weight_init),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.lax.tanh,
            hk.nets.MLP([cfg.hidden_dim, cfg.repr_dim],
                         w_init=weight_init,
                         activation=jax.nn.elu,
                         activate_final=False)
        ])
        
    def __call__(self, obs_repr):
        prediction = self.predictor(obs_repr)
        return prediction