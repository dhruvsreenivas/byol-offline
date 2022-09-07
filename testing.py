import jax
import jax.numpy as jnp
import haiku as hk
import hydra

from byol_offline.models import *

@hydra.main(config_path='cfgs', config_name='config')
def test_world_model(cfg):
    # Make sure print statements are enabled in WM __call__ function to print out state when running this method
    key = jax.random.PRNGKey(42)
    init_key, key1, key2 = jax.random.split(key, 3)

    # dummy input creation
    dummy_obs = jax.random.normal(key1, shape=(20, 10, 111))
    dummy_actions = jax.random.normal(key2, shape=(20, 10, 7))
    
    # world model creation
    wm_fn = lambda o, a: MLPLatentWorldModel(cfg.byol.d4rl)(o, a)
    wm = hk.without_apply_rng(hk.transform(wm_fn))
    params = wm.init(init_key, dummy_obs, dummy_actions)

    # sliding window mask
    mask = jnp.arange(20)
    mask = jnp.expand_dims(mask, axis=range(1, jnp.ndim(dummy_obs))) # (20, 1, 1), can use same mask for dummy action
    mask = jnp.logical_or(mask < 7, mask >= 7 + 8)

    obs = jnp.where(mask, 0, dummy_obs)
    actions = jnp.where(mask, 0, dummy_actions)
    obs = jnp.roll(obs, -7)
    actions = jnp.roll(actions, -7)

    _ = wm.apply(params, obs, actions)

if __name__ == '__main__':
    test_world_model()