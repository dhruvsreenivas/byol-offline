import chex
import haiku as hk
import jax
import jax.numpy as jnp
import distrax

from byol_offline.distributions import ClippedNormal, TanhTransformed
from byol_offline.networks.encoder import AtariEncoder, DuelingMLP
from byol_offline.types import DoubleQOutputs

"""Actor critic modules."""

# ============================== DDPG ==============================


class DDPGActor(hk.Module):
    """DDPG actor for visual control."""

    def __init__(self, action_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()

        initializer = hk.initializers.Orthogonal(scale=1.0)

        self._trunk = hk.Sequential(
            [
                hk.Linear(feature_dim, w_init=initializer),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jnp.tanh,
            ]
        )

        self._policy = hk.Sequential(
            [
                hk.Linear(hidden_dim, w_init=initializer),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=initializer),
                jax.nn.relu,
                hk.Linear(action_dim, w_init=initializer),
            ]
        )

    def __call__(self, obs: chex.Array, std: chex.Numeric) -> distrax.Distribution:
        h = self._trunk(obs)

        mu = self._policy(h)
        mu = jnp.tanh(mu)
        std = jnp.ones_like(mu) * std

        distribution = ClippedNormal(mu, std)
        return distribution


class DDPGCritic(hk.Module):
    """DDPG critic for visual control."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()

        initializer = hk.initializers.Orthogonal(scale=1.0)

        self._trunk = hk.Sequential(
            [
                hk.Linear(feature_dim, w_init=initializer),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jnp.tanh,
            ]
        )

        # q function heads
        self._q1 = hk.Sequential(
            [
                hk.Linear(hidden_dim, w_init=initializer),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=initializer),
                jax.nn.relu,
                hk.Linear(1, w_init=initializer),
            ]
        )
        self._q2 = hk.Sequential(
            [
                hk.Linear(hidden_dim, w_init=initializer),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=initializer),
                jax.nn.relu,
                hk.Linear(1, w_init=initializer),
            ]
        )

    def __call__(self, obs: chex.Array, action: chex.Array) -> DoubleQOutputs:
        h = self._trunk(obs)

        ha = jnp.concatenate([h, action], -1)
        q1 = self._q1(ha)
        q2 = self._q2(ha)

        return jnp.squeeze(q1), jnp.squeeze(q2)


# ============================== TD3 ==============================


class TD3Actor(hk.Module):
    """TD3 actor for MuJoCo envs, from https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py."""

    def __init__(self, hidden_dim: int, action_dim: int, max_action: float):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._action_dim = action_dim
        self._max_action = max_action

    def __call__(self, obs: chex.Array) -> chex.Array:
        initializer = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")

        x = hk.Linear(self._hidden_dim, w_init=initializer)(obs)
        x = jax.nn.relu(x)
        x = hk.Linear(self._hidden_dim, w_init=initializer)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._action_dim, w_init=initializer)(x)

        x = jnp.tanh(x)
        return x * self._max_action


class TD3Critic(hk.Module):
    """TD3 critic for MuJoCo envs, from https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        initializer = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")

        self._q1 = hk.nets.MLP(
            [hidden_dim, hidden_dim, 1],
            w_init=initializer,
            activation=jax.nn.relu,
            name="q1",
        )

        self._q2 = hk.nets.MLP(
            [hidden_dim, hidden_dim, 1],
            w_init=initializer,
            activation=jax.nn.relu,
            name="q2",
        )

    def __call__(self, obs: chex.Array, action: chex.Array) -> DoubleQOutputs:
        sa = jnp.concatenate([obs, action], axis=-1)

        # twin critic outputs
        q1 = self._q1(sa)
        q2 = self._q2(sa)

        return jnp.squeeze(q1), jnp.squeeze(q2)


# ============================== SAC ==============================

MIN_LOG_STD = -5
MAX_LOG_STD = 2


class SACActor(hk.Module):
    """SAC actor."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()

        initializer = hk.initializers.Orthogonal(scale=1.0)

        self._policy = hk.Sequential(
            [
                hk.Linear(hidden_dim, w_init=initializer, b_init=jnp.zeros),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=initializer, b_init=jnp.zeros),
                jax.nn.relu,
                hk.Linear(2 * action_dim, w_init=initializer, b_init=jnp.zeros),
            ]
        )

    def __call__(self, observation: chex.Array) -> distrax.Distribution:
        output = self._policy(observation)

        mu, log_std = jnp.split(output, 2, axis=-1)
        log_std = MIN_LOG_STD + 0.5 * (MAX_LOG_STD - MIN_LOG_STD) * (log_std + 1)
        std = jnp.exp(log_std)

        # get distribution and required transform
        base_distribution = distrax.MultivariateNormalDiag(mu, std)
        distribution = TanhTransformed(base_distribution)

        return distribution


class SACCritic(hk.Module):
    """Critic for SAC."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        initializer = hk.initializers.Orthogonal(scale=1.0)

        self._q1 = hk.Sequential(
            [
                hk.Linear(hidden_dim, w_init=initializer, b_init=jnp.zeros),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=initializer, b_init=jnp.zeros),
                jax.nn.relu,
                hk.Linear(1, w_init=initializer, b_init=jnp.zeros),
            ]
        )

        self._q2 = hk.Sequential(
            [
                hk.Linear(hidden_dim, w_init=initializer, b_init=jnp.zeros),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=initializer, b_init=jnp.zeros),
                jax.nn.relu,
                hk.Linear(1, w_init=initializer, b_init=jnp.zeros),
            ]
        )

    def __call__(self, observation: chex.Array, action: chex.Array) -> DoubleQOutputs:
        obs_action = jnp.concatenate([observation, action], axis=-1)

        q1 = self._q1(obs_action)
        q2 = self._q2(obs_action)

        return jnp.squeeze(q1), jnp.squeeze(q2)


# ============================== BC ==============================


class BCActor(hk.Module):
    """Network for behavioral cloning."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._action_dim = action_dim

    def __call__(self, observation: chex.Array) -> distrax.Distribution:
        net = hk.nets.MLP(
            [self._hidden_dim, self._hidden_dim, 2 * self._action_dim],
            activation=jax.nn.relu,
        )
        out = net(observation)
        mean, log_std = jnp.split(out, 2, -1)
        std = jnp.exp(log_std)

        distribution = distrax.Normal(mean, std)
        return distribution


# ============================== DQN ==============================


class DuelingDQN(hk.Module):
    """Dueling DQN network."""

    def __init__(self, action_dim: int):
        super().__init__(name="dueling_dqn")

        self._trunk = AtariEncoder()
        self._mlp = DuelingMLP(action_dim)

    def __call__(self, x: chex.Array) -> chex.Array:
        z = self._trunk(x)
        q = self._mlp(z)
        return q
