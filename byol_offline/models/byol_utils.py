import chex
import jax
import jax.numpy as jnp
from typing import Optional


def l2_normalize(
    x: chex.Array, axis: Optional[int] = None, epsilon: float = 1e-12
) -> chex.Array:
    """Stable L2 normalization from BYOL repo."""

    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm


def sliding_window(arr: chex.Array, idx: int, window_size: int) -> chex.Array:
    """Get the relevant mask for given array, starting at `idx`, with specific window size."""

    ndims = jnp.ndim(arr)
    mask = jnp.arange(arr.shape[0])
    mask = jnp.expand_dims(mask, axis=range(1, ndims))
    mask = jnp.logical_or(mask < idx, mask >= idx + window_size)

    masked_arr = jnp.where(mask, 0, arr)  # zeros out for unacceptable starting idxes
    masked_arr = jnp.roll(masked_arr, shift=-idx, axis=0)
    return masked_arr
