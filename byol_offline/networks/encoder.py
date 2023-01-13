import jax
import jax.numpy as jnp
import haiku as hk
from byol_offline.networks.network_utils import INITIALIZERS

class DrQv2Encoder(hk.Module):
    '''DeepMind Control Suite encoder, from DrQv2.'''
    def __init__(self):
        super().__init__()
        self.repr_dim = 20000
        
        self.convnet = hk.Sequential([
            hk.Conv2D(32, kernel_shape=3, stride=2, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=3, stride=1, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=3, stride=1, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=3, stride=1, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Flatten()
        ])
        
    def __call__(self, obs: jnp.ndarray):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h
        
class DreamerEncoder(hk.Module):
    '''Dreamer encoder, from DreamerV2.'''
    def __init__(self, depth):
        super().__init__()
        
        self.convnet = hk.Sequential([
            hk.Conv2D(depth, kernel_shape=4, stride=2, w_init=INITIALIZERS['xavier_uniform'], b_init=INITIALIZERS['zeros']),
            jax.nn.elu,
            hk.Conv2D(depth * 2, kernel_shape=4, stride=2, w_init=INITIALIZERS['xavier_uniform'], b_init=INITIALIZERS['zeros']),
            jax.nn.elu,
            hk.Conv2D(depth * 4, kernel_shape=4, stride=2, w_init=INITIALIZERS['xavier_uniform'], b_init=INITIALIZERS['zeros']),
            jax.nn.elu,
            hk.Conv2D(depth * 8, kernel_shape=4, stride=2, w_init=INITIALIZERS['xavier_uniform'], b_init=INITIALIZERS['zeros']),
            jax.nn.elu,
            hk.Flatten()
        ])
        
    def __call__(self, obs: jnp.ndarray):
        obs = obs / 255.0 - 0.5
        out = self.convnet(obs)
        return out

# ====== Atari stuff ======

class AtariEncoder(hk.Module):
    '''Atari encoder for DQN. From acme/jax/networks/atari.py.'''
    def __init__(self):
        super().__init__()
        self.convnet = hk.Sequential([
            hk.Conv2D(32, [8, 8], 4),
            jax.nn.relu,
            hk.Conv2D(64, [4, 4], 2),
            jax.nn.relu,
            hk.Conv2D(64, [3, 3], 1),
            jax.nn.relu,
            hk.Flatten() # this basically does the reshaping at the end
        ])
        
    def __call__(self, x: jnp.ndarray):
        outs = self.convnet(x)
        return outs
    
class DuelingMLP(hk.Module):
    '''Dueling DQN MLP, from acme/jax/networks/atari.py.'''
    def __init__(self, action_dim):
        super().__init__()
        self._value_mlp = hk.Sequential([
            hk.Linear(512),
            jax.nn.relu,
            hk.Linear(1)
        ])
        self._adv_mlp = hk.Sequential([
            hk.Linear(512),
            jax.nn.relu,
            hk.Linear(action_dim)
        ])
    
    def __call__(self, x: jnp.ndarray):
        v = self._value_mlp(x)
        a = self._adv_mlp(x)
        a -= jnp.mean(a, axis=-1, keepdims=True)
        q = v + a
        return q

# ====== for encoder repr dim calculation ======
def out_fn(x):
    return 1 + int((x - 4) / 2)

def dreamer_enc_repr_dim(depth):
    out_size = 64
    for _ in range(4):
        out_size = out_fn(out_size)
    out_dim = depth * 8 * out_size * out_size  # 1536
    return out_dim
        