import jax
import jax.numpy as jnp
from typing import Optional
import haiku as hk

def l2_normalize(x: jnp.ndarray, axis: Optional[int] = None, epsilon: float=1e-12) -> jnp.ndarray:
    '''Stable L2 normalization from BYOL repo.'''
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm

def sliding_window(arr, idx, window_size):
    '''Get the relevant mask for given array, starting at index, with specific window size.'''
    ndims = jnp.ndim(arr)
    mask = jnp.arange(arr.shape[0])
    mask = jnp.expand_dims(mask, axis=range(1, ndims))
    mask = jnp.tile(mask, (1,) + arr.shape[1:])
    mask = (mask >= idx) * (mask < idx + window_size) * (idx + window_size < arr.shape[0])
    
    masked_arr = jnp.where(mask, arr, 0)
    return masked_arr

def target_update_fn(params: hk.Params, target_params: hk.Params, ema: float) -> hk.Params:
    new_target_params = jax.tree_util.tree_map(lambda x, y: ema * x + (1.0 - ema) * y, target_params, params)
    return new_target_params