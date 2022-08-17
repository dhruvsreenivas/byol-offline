import jax
import jax.numpy as jnp
from typing import Optional
import haiku as hk
import functools

def sliding_window(arr: jnp.ndarray, window_size: int) -> jnp.ndarray:
    '''Creates a sliding window, with shape 'window_size', that sweeps across the entire array along the first axis.'''
    idx = jnp.arange(arr.shape[0] - window_size + 1)[:, None] + jnp.arange(window_size)[None, :]
    return arr[idx]

def l2_normalize(x: jnp.ndarray, axis: Optional[int] = None, epsilon: float=1e-12) -> jnp.ndarray:
    '''Stable L2 normalization from BYOL repo.'''
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm

def pad_sliding_windows(arr: jnp.ndarray, seq_len: int) -> jnp.ndarray:
    '''Pad sliding windows triangularly with 0s.
    
    :param arr: Input array, should have shape (num_seqs, window_size, B), where window_size < seq_len.
    
    :return properly padded arr, of shape (num_seqs, seq_len, B).
    '''
    arr = jnp.pad(arr, [(0, 0), (0, seq_len - arr.shape[1]), (0, 0)]) # pad with 0s first on right of dim 1
    # now pad
    rows, column_indices = jnp.ogrid[:arr.shape[0], :arr.shape[1]]
    roll_amts = jnp.arange(arr.shape[0])
    column_indices = column_indices - roll_amts[:, jnp.newaxis]
    return arr[rows, column_indices, :]

def target_update_fn(params: hk.Params, target_params: hk.Params, ema: float) -> hk.Params:
    new_target_params = jax.tree_util.tree_map(lambda x, y: ema * x + (1.0 - ema) * y, target_params, params)
    return new_target_params