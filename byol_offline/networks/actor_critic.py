import jax
import jax.numpy as jnp
import haiku as hk
from byol_offline.networks.network_utils import ClippedNormal, INITIALIZERS

class Actor(hk.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()
        
        self.trunk = hk.Sequential([
            hk.Linear(feature_dim, w_init=INITIALIZERS['linear_orthogonal']),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jnp.tanh
        ])
        
        self.policy = hk.Sequential([
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal']),
            jax.nn.relu,
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal']),
            jax.nn.relu,
            hk.Linear(action_shape[0], w_init=INITIALIZERS['linear_orthogonal'])
        ])
    
    def __call__(self, obs, std):
        h = self.trunk(obs)
        
        mu = self.policy(h)
        mu = jnp.tanh(mu)
        std = jnp.ones_like(mu) * std
        
        dist = ClippedNormal(mu, std)
        return dist
    
class Critic(hk.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        
        self.trunk = hk.Sequential([
            hk.Linear(feature_dim, w_init=INITIALIZERS['linear_orthogonal']),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jnp.tanh
        ])
        
        self.q1 = hk.Sequential([
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal']),
            jax.nn.relu,
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal']),
            jax.nn.relu,
            hk.Linear(1, w_init=INITIALIZERS['linear_orthogonal'])
        ])
        
        self.q2 = hk.Sequential([
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal']),
            jax.nn.relu,
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal']),
            jax.nn.relu,
            hk.Linear(1, w_init=INITIALIZERS['linear_orthogonal'])
        ])
        
    def __call__(self, obs, action):
        h = self.trunk(obs)
        h_a = jnp.concatenate([h, action], -1)
        q1 = self.q1(h_a)
        q2 = self.q2(h_a)
        
        return q1, q2