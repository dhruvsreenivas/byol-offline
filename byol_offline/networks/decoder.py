import jax
import jax.numpy as jnp
import haiku as hk
from byol_offline.networks.network_utils import INITIALIZERS

class DrQv2Decoder(hk.Module):
    '''Reverse of DrQv2 encoder basically.'''
    def __init__(self, in_channel):
        super().__init__()
        
        self.convnet = hk.Sequential([
            hk.Conv2DTranspose(32, kernel_shape=3, stride=1, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2DTranspose(32, kernel_shape=3, stride=1, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2DTranspose(32, kernel_shape=3, stride=1, w_init=INITIALIZERS['conv2d_orthogonal']),
            jax.nn.relu,
            hk.Conv2DTranspose(in_channel, kernel_shape=3, stride=2, w_init=INITIALIZERS['conv2d_orthogonal']),
        ])
    
    def __call__(self, x):
        return self.convnet(x)
    
class DreamerDecoder(hk.Module):
    '''Dreamer decoder from DreamerV2.'''
    def __init__(self, in_channel, depth):
        super().__init__()
        self._depth = depth
        
        self.convnet = hk.Sequential([
            hk.Conv2DTranspose(depth * 4, kernel_shape=5, stride=2, w_init=INITIALIZERS['xavier_uniform']),
            jax.nn.elu,
            hk.Conv2DTranspose(depth * 2, kernel_shape=5, stride=2, w_init=INITIALIZERS['xavier_uniform']),
            jax.nn.elu,
            hk.Conv2DTranspose(depth, kernel_shape=5, stride=2, w_init=INITIALIZERS['xavier_uniform']),
            jax.nn.elu,
            hk.Conv2DTranspose(in_channel, kernel_shape=5, stride=2, w_init=INITIALIZERS['xavier_uniform']),
        ])
        
    def __call__(self, x):
        x = hk.Linear(self._depth * 32)(x)
        x = jnp.reshape(x, x.shape[:-1] + (self._depth * 32, 1, 1))
        mean = self.convnet(x)
        return mean