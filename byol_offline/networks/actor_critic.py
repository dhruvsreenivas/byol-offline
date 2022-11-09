import jax
import jax.numpy as jnp
import haiku as hk
from byol_offline.networks.network_utils import *

# ============================== DDPG ==============================

class DDPGActor(hk.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()
        
        self.trunk = hk.Sequential([
            hk.Linear(feature_dim, w_init=INITIALIZERS['linear_orthogonal']),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.lax.tanh
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
        mu = jax.lax.tanh(mu)
        std = jnp.ones_like(mu) * std
        
        dist = ClippedNormal(mu, std)
        return dist
    
class DDPGCritic(hk.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        
        self.trunk = hk.Sequential([
            hk.Linear(feature_dim, w_init=INITIALIZERS['linear_orthogonal']),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.lax.tanh
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
        
        return jnp.squeeze(q1), jnp.squeeze(q2)

# ============================== TD3 ==============================

class TD3Actor(hk.Module):
    '''TD3 actor for MuJoCo envs, from https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py.'''
    def __init__(self, hidden_dim, action_shape, max_action):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._action_dim = action_shape[0]
        self._max_action = max_action
        
        self._weight_init = INITIALIZERS['he_uniform']
    
    def __call__(self, obs):
        x = hk.Linear(self._hidden_dim, w_init=self._weight_init)(obs)
        x = jax.nn.relu(x)
        x = hk.Linear(self._hidden_dim, w_init=self._weight_init)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._action_dim, w_init=self._weight_init)(x)
        
        x = jax.lax.tanh(x)
        return x * self._max_action
    
class TD3Critic(hk.Module):
    '''TD3 critic for MuJoCo envs, from https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py.'''
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.q1 = hk.nets.MLP(
            [hidden_dim, hidden_dim, 1],
            w_init=INITIALIZERS['he_uniform'],
            activation=jax.nn.relu,
            name='q1'
        )
        
        self.q2 = hk.nets.MLP(
            [hidden_dim, hidden_dim, 1],
            w_init=INITIALIZERS['he_uniform'],
            activation=jax.nn.relu,
            name='q2'
        )
        
    def __call__(self, obs, action):
        sa = jnp.concatenate([obs, action], axis=-1)
        
        # twin critic outputs
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return jnp.squeeze(q1), jnp.squeeze(q2)
    
# ============================== SAC ==============================
    
# TODO: do we keep these hardcoded for DMC?
MIN_LOG_STD = -5
MAX_LOG_STD = 2

class SACActor(hk.Module):
    def __init__(self, action_shape, hidden_dim):
        super().__init__()
        
        self.policy = hk.Sequential([
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros']),
            jax.nn.relu,
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros']),
            jax.nn.relu,
            hk.Linear(2 * action_shape[0], w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros'])
        ])
        
    def __call__(self, obs):
        output = self.policy(obs)
        mu, log_std = jnp.split(output, 2, -1)
        log_std = MIN_LOG_STD + 0.5 * (MAX_LOG_STD - MIN_LOG_STD) * (log_std + 1)
        
        std = jnp.exp(log_std)
        dist = squashed_normal_dist(mu, std)
        return dist
    
class SACCritic(hk.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.q1 = hk.Sequential([
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros']),
            jax.nn.relu,
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros']),
            jax.nn.relu,
            hk.Linear(1, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros'])
        ])
        
        self.q2 = hk.Sequential([
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros']),
            jax.nn.relu,
            hk.Linear(hidden_dim, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros']),
            jax.nn.relu,
            hk.Linear(1, w_init=INITIALIZERS['linear_orthogonal'], b_init=INITIALIZERS['zeros'])
        ])
        
    def __call__(self, obs, action):
        obs_action = jnp.concatenate([obs, action], axis=-1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        
        return jnp.squeeze(q1), jnp.squeeze(q2)

# ============================== BC ==============================

class BCActor(hk.Module):
    '''Behavioral cloning network.'''
    def __init__(self, hidden_dim, action_shape):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._action_dim = action_shape[0]
        
    def __call__(self, obs):
        net = hk.nets.MLP(
            [self._hidden_dim, self._hidden_dim, 2 * self._action_dim],
            activation=jax.nn.relu
        )
        out = net(obs)
        mean, log_std = jnp.split(out, 2, -1)
        std = jnp.exp(log_std)
        
        dist = distrax.Normal(mean, std)
        return dist