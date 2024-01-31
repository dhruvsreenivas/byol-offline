import dmcgym
import gym
import jax

from byol_offline.data.vd4rl_dataset import VD4RLDataset, default_wrap
from byol_offline.data import _preprocess, Batch, SequenceBatch

if __name__ == "__main__":
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
