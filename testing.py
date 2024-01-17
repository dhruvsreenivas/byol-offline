import dmcgym
import gym
import jax

from byol_offline.data.vd4rl_dataset import VD4RLDataset, default_wrap

if __name__ == "__main__":
    env = gym.make("cheetah-run-v0")
    env, pixel_keys = default_wrap(env)
    
    dataset = VD4RLDataset(env, "expert", pixel_keys=pixel_keys)
    print()
    
    batch = dataset.sample(100)
    print(jax.tree_util.tree_map(lambda x: x.shape, batch))
    
    print("\n========================================================\n")
    
    sequence_batch = dataset.sample_sequences(10, 5)
    print(jax.tree_util.tree_map(lambda x: x.shape, sequence_batch))
    
    print("\n========================================================\n")
    
    sequence_batch2 = dataset.sample_sequences(10, 5, pack_obs_and_next_obs=True)
    print(jax.tree_util.tree_map(lambda x: x.shape, sequence_batch2))
    print()