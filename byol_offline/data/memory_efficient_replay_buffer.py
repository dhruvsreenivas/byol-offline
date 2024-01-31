import copy
from typing import Iterable, Optional, Tuple, Mapping
import random

import gym
import numpy as np
from gym.spaces import Box

from byol_offline.data.dataset import (
    ReplayBuffer,
    DatasetDict,
    _sample,
    _dict_to_batch,
    _batch_to_dict,
    _stack_dicts,
    Batch,
    SequenceBatch,
)


class MemoryEfficientReplayBuffer(ReplayBuffer):
    """Memory efficient replay buffer for image observations."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
    ):
        assert len(pixel_keys) <= 1, "Not implemented for multiple pixel keys."
        self._pixel_keys = pixel_keys

        observation_space = copy.deepcopy(observation_space)
        self._num_stack = None
        for pixel_key in self._pixel_keys:
            pixel_obs_space = observation_space.spaces[pixel_key]
            if self._num_stack is None:
                self._num_stack = pixel_obs_space.shape[-1]
            else:
                assert self._num_stack == pixel_obs_space.shape[-1]

            self._unstacked_dim_size = pixel_obs_space.shape[-2]
            low = pixel_obs_space.low[..., 0]
            high = pixel_obs_space.high[..., 0]
            unstacked_pixel_obs_space = Box(
                low=low, high=high, dtype=pixel_obs_space.dtype
            )
            observation_space.spaces[pixel_key] = unstacked_pixel_obs_space

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        for pixel_key in self._pixel_keys:
            next_observation_space_dict.pop(pixel_key)
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            observation_key=pixel_keys[0],
        )

    def insert(self, data_dict: DatasetDict) -> None:
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        obs_pixels = {}
        next_obs_pixels = {}
        for pixel_key in self._pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        if self._first:
            for i in range(self._num_stack):
                for pixel_key in self._pixel_keys:
                    data_dict["observations"][pixel_key] = obs_pixels[pixel_key][..., i]

                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        for pixel_key in self._pixel_keys:
            data_dict["observations"][pixel_key] = next_obs_pixels[pixel_key][..., -1]

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> Batch:
        """Samples from the replay buffer.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one image.
                It's useful when they have overlapping frames.

        Returns:
            A batch.
        """

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

            # verify correctness here
            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    if hasattr(self.np_random, "integers"):
                        indx[i] = self.np_random.integers(len(self))
                    else:
                        indx[i] = self.np_random.randint(len(self))
        else:
            raise NotImplementedError("Not implemented for non-None indices.")

        if keys is None:
            keys = self._dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")
        keys.remove("next_observations")

        batch = super().sample(batch_size, keys, indx)

        obs_keys = self._dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self._pixel_keys:
            obs_keys.remove(pixel_key)

        # convert all non-observation stuff back to dict
        batch = _batch_to_dict(batch, self._observation_key)

        # add batch observations in
        batch["observations"] = {}
        batch["next_observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(
                self._dataset_dict["observations"][k], indx
            )

        for pixel_key in self._pixel_keys:
            obs_pixels = self._dataset_dict["observations"][pixel_key]
            obs_pixels = np.lib.stride_tricks.sliding_window_view(
                obs_pixels, self._num_stack + 1, axis=0
            )
            obs_pixels = obs_pixels[indx - self._num_stack]

            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[..., :-1]
                batch["next_observations"][pixel_key] = obs_pixels[..., 1:]

        batch = _dict_to_batch(batch, self._observation_key)
        return batch

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        keys: Optional[Iterable[str]] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> SequenceBatch:
        if self._trajectory_info is None:
            self._trajectory_info = self._trajectory_boundaries_and_returns()

        episode_starts, episode_ends, _ = self._trajectory_info

        # now we have to find sequences which are long enough
        good_trajectory_indices = [
            (s, e)
            for s, e in zip(episode_starts, episode_ends)
            if e - s >= sequence_length
        ]

        # now we choose good starting indices for each datapoint
        starting_indices = []
        for _ in range(batch_size):
            s, e = random.choice(
                good_trajectory_indices
            )  # e is exclusive, so max is e - sequence_length
            L = e - s
            offset = (
                np.random.randint(0, L - sequence_length) if L > sequence_length else 0
            )
            assert s + offset + sequence_length < e

            starting_indices.append(s + offset)

        if keys is None:
            keys = self._dataset_dict.keys()
        else:
            assert "observations" in keys

        # ----- now sample all non-observation attrs -----

        keys = list(keys)
        keys.remove("observations")
        keys.remove("next_observations")

        obs_keys = self._dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self._pixel_keys:
            obs_keys.remove(pixel_key)

        sequence_batch_dicts = []
        for starting_index in starting_indices:
            indx = np.arange(starting_index, starting_index + sequence_length)

            batch_dict = dict()
            for k in keys:
                if isinstance(self._dataset_dict[k], Mapping):
                    batch_dict[k] = _sample(self._dataset_dict[k], indx)
                else:
                    batch_dict[k] = self._dataset_dict[k][indx]

            # now we have to add the observations back in
            batch_dict["observations"] = {}
            batch_dict["next_observations"] = {}
            for k in obs_keys:
                batch_dict["observations"][k] = _sample(
                    self._dataset_dict["observations"][k], indx
                )

            for pixel_key in self._pixel_keys:
                obs_pixels = self._dataset_dict["observations"][pixel_key]
                obs_pixels = np.lib.stride_tricks.sliding_window_view(
                    obs_pixels, self._num_stack + 1, axis=0
                )
                obs_pixels = obs_pixels[indx - self._num_stack]

                if pack_obs_and_next_obs:
                    batch_dict["observations"][pixel_key] = obs_pixels
                else:
                    batch_dict["observations"][pixel_key] = obs_pixels[..., :-1]
                    batch_dict["next_observations"][pixel_key] = obs_pixels[..., 1:]

            sequence_batch_dicts.append(batch_dict)

        sequence_batch_dict = _stack_dicts(sequence_batch_dicts, axis=1)
        del sequence_batch_dicts

        return _dict_to_batch(
            sequence_batch_dict, self._observation_key, is_sequence=True
        )
