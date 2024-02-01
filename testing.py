import dmcgym
import gym
import chex
import jax
import jax.numpy as jnp

from byol_offline.data.vd4rl_dataset import VD4RLDataset, default_wrap
from byol_offline.data import _preprocess, Batch, SequenceBatch
from byol_offline.models.byol_utils import sliding_window


def data_tests():
    """Tests dataset ops."""

    env = gym.make("cheetah-run-v0")
    env, pixel_keys = default_wrap(env)

    dataset = VD4RLDataset(env, "expert", pixel_keys=pixel_keys)
    print()

    batch = dataset.sample(100)
    assert isinstance(batch, Batch)
    print(jax.tree_util.tree_map(lambda x: x.shape, batch))
    print(jax.tree_util.tree_map(lambda x: x.dtype, batch))

    print("\n====================== SEQ ===============================\n")

    sequence_batch = dataset.sample_sequences(10, 5)
    print(jax.tree_util.tree_map(lambda x: x.shape, sequence_batch))
    print(jax.tree_util.tree_map(lambda x: x.dtype, sequence_batch))

    print("\n========================================================\n")

    sequence_batch_preprocessed = _preprocess(sequence_batch)
    print(jax.tree_util.tree_map(lambda x: x.dtype, sequence_batch_preprocessed))

    print("\n======================= V2 ===============================\n")

    sequence_batch2 = dataset.sample_sequences(10, 5, pack_obs_and_next_obs=True)
    print(jax.tree_util.tree_map(lambda x: x.shape, sequence_batch2))
    print(jax.tree_util.tree_map(lambda x: x.dtype, sequence_batch2))

    print("\n========================================================\n")

    sequence_batch_preprocessed2 = _preprocess(sequence_batch2)
    assert isinstance(sequence_batch_preprocessed2, SequenceBatch)
    print(jax.tree_util.tree_map(lambda x: x.dtype, sequence_batch_preprocessed2))

    print()


def rolling_window_test():
    """Tests whether our rolling window implementation yields the same result as the one from Acme."""

    def rolling_window_acme(
        x: chex.Array, window_size: int, axis: int = 0
    ) -> chex.Array:
        T = x.shape[axis]
        starts = jnp.arange(T - window_size + 1)
        ends = jnp.arange(window_size)

        idx = starts[:, None] + ends[None, :]
        out = jnp.take(x, idx, axis=axis)
        return out

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape=(23, 10, 64, 64, 3))
    window_size = 5

    acme_window = rolling_window_acme(x, window_size=window_size)
    our_window = sliding_window(
        x, x.shape[0] - window_size, window_size
    )  # this should be the masked array, but with last 5 elements rolled to the front

    # acme window is of shape [T, B, *dims], where last 5 elements are the LAST elements in the list are rolled to the front
    # this should be equal to the LAST element in the acme window

    assert acme_window.shape == (
        x.shape[0] - window_size + 1,
        window_size,
        10,
        64,
        64,
        3,
    )
    assert our_window.shape == x.shape

    last_window = acme_window[-1]  # [5, 10, 64, 64, 3]
    assert (
        last_window == our_window[:window_size]
    ).all(), "WRONG: sliced windows should be the same."


if __name__ == "__main__":
    rolling_window_test()
