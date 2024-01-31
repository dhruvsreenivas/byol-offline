import gym
import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Union, Mapping

from byol_offline.base_learner import Learner, ReinforcementLearner
from byol_offline.data import Batch, SequenceBatch

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


def combine_batches(
    batch1: Union[Batch, SequenceBatch], batch2: Union[Batch, SequenceBatch]
) -> Union[Batch, SequenceBatch]:
    return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0), batch1, batch2
    )


# ================ GYM UTILS ================


def is_pixel_based(observation_space: gym.Space) -> bool:
    """Determines whether an environment is pixel-based from its observation spec."""

    return isinstance(observation_space, gym.spaces.Dict) and "pixels" in list(
        observation_space.keys()
    )


# ================ EVALUATION UTILS ================


def evaluate_model_based(
    env: gym.Env, agent: ReinforcementLearner, model: Learner, num_episodes: int = 10
) -> Mapping[str, float]:
    """Evaluates the agent trained in a model-based fashion."""

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    def to_array(
        observation: Union[np.ndarray, Mapping[str, np.ndarray]]
    ) -> np.ndarray:
        if isinstance(observation, Mapping):
            return observation["pixels"] / 255.0 - 0.5

        return observation

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        observation = to_array(observation).squeeze()
        observation = np.expand_dims(
            observation, axis=0
        )  # to add batch dim, [1, 64, 64, 3]

        # get initial state of the model
        model._state, state = model._initialize_state(model._state, 1)
        dummy_action = np.zeros_like(env.action_space.sample())
        dummy_action = jnp.expand_dims(dummy_action, axis=0)  # [1, action_dim]

        while not done:
            # first embed the observation
            model._state, embed = model._encode(model._state, observation)

            # now get the feature
            model._state, (state, feature, _) = model._observe(
                model._state, embed, dummy_action, state
            )

            # now run the policy
            agent._state, action = agent._act(agent._state, feature, step=0, eval=True)
            action = jnp.squeeze(action)

            # run the action in the real environment
            observation, _, done, _ = env.step(action)
            observation = to_array(observation).squeeze()
            observation = np.expand_dims(
                observation, axis=0
            )  # to add batch dim, [1, 64, 64, 3]

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
