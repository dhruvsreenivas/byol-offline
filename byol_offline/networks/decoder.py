import jax
import jax.numpy as jnp
import haiku as hk
from byol_offline.networks.network_utils import INITIALIZERS

class DrQv2Decoder(hk.Module):
    '''Reverse of DrQv2 encoder basically.'''
    def __init__(self, in_channel):
        super().__init__()
        
        self.convnet = hk.Sequential([
            hk.Conv2DTranspose(32, kernel_shape=3, stride=1, padding='VALID', w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2DTranspose(32, kernel_shape=3, stride=1, padding='VALID', w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2DTranspose(32, kernel_shape=3, stride=1, padding='VALID', w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2DTranspose(in_channel, kernel_shape=3, stride=2, padding='VALID', w_init=INITIALIZERS['conv2d_orthogonal']),
        ])
    
    def __call__(self, x):
        out_dim = 20000
        x = hk.Linear(out_dim)(x)
        x = jnp.reshape(x, (-1, 25, 25, 32))
        return self.convnet(x)
    
class DreamerDecoder(hk.Module):
    '''Dreamer decoder from DreamerV2.'''
    def __init__(self, in_channel, depth):
        super().__init__()
        self._depth = depth
        
        self.convnet = hk.Sequential([
            hk.Conv2DTranspose(depth * 4, kernel_shape=5, stride=2, padding='VALID', w_init=INITIALIZERS['xavier_uniform']),
            jax.nn.elu,
            hk.Conv2DTranspose(depth * 2, kernel_shape=5, stride=2, padding='VALID', w_init=INITIALIZERS['xavier_uniform']),
            jax.nn.elu,
            hk.Conv2DTranspose(depth, kernel_shape=6, stride=2, padding='VALID', w_init=INITIALIZERS['xavier_uniform']),
            jax.nn.elu,
            hk.Conv2DTranspose(in_channel, kernel_shape=6, stride=2, padding='VALID', w_init=INITIALIZERS['xavier_uniform']),
        ])
        
    def __call__(self, x):
        # (B, feature_dim)
        out_dim = 32 * self._depth
        x = hk.Linear(out_dim)(x)
        x = jnp.reshape(x, (x.shape[0], 1, 1, out_dim))
        mean = self.convnet(x)
        return mean