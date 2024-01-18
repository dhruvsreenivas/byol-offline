import gym
import chex
import jax
import jax.numpy as jnp
from typing import Any, Mapping

"""Basic utilities."""


def broadcast_to_local_devices(value: Any) -> Any:
    """Broadcasts an object to all local devices. Taken from BYOL repository."""
    
    devices = jax.local_devices()
    
    def _replicate(x: Any) -> Any:
        x = jnp.array(x)
        return jax.device_put_sharded([x] * len(devices), devices)
    
    return jax.tree_util.tree_map(_replicate, value)


def seq_batched_like(array: chex.Array) -> chex.Array:
    """Returns a sequence array similar to the one provided that is necessary for batching."""
    
    array_expanded = jnp.expand_dims(array, 0)
    array_expanded = jnp.expand_dims(array_expanded, 0)
    
    array_expanded = jnp.tile(array_expanded, (2,) + (1,) * (array_expanded.ndim - 1))
    return array_expanded


def print_dict(log_dict: Mapping) -> None:
    for k, v in log_dict.items():
        print(f'{k}: {v}')

# ================ GYM UTILS ================

def is_pixel_based(observation_space: gym.Space) -> bool:
    """Determines whether an environment is pixel-based from its observation spec."""
    
    return isinstance(observation_space, gym.spaces.Dict) and "pixels" in list(observation_space.keys())