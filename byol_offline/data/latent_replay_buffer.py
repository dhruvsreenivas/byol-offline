import gym
import jax
import numpy as np
import collections
from typing import Optional, Mapping, Generator, Iterable
import dill

from byol_offline.data import Dataset, Batch
from byol_offline.data.dataset import _insert_recursively, _sample, _dict_to_batch
from byol_offline.types import DatasetDict


class LatentReplayBuffer(Dataset):
    """Latent replay buffer for storing both real and model-based rollouts."""

    def __init__(
        self,
        latent_observation_dim: int,
        action_space: gym.Space,
        capacity: int,
        seed: Optional[int] = None,
        observation_key: Optional[str] = None,
    ):
        real_dataset_dict = dict(
            observations=np.empty((capacity, latent_observation_dim), dtype=np.float32),
            next_observations=np.empty(
                (capacity, latent_observation_dim), dtype=np.float32
            ),
            actions=np.empty((capacity, *action_space.shape), dtype=np.float32),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=np.float32),
        )
        mb_dataset_dict = dict(
            observations=np.empty((capacity, latent_observation_dim), dtype=np.float32),
            next_observations=np.empty(
                (capacity, latent_observation_dim), dtype=np.float32
            ),
            actions=np.empty((capacity, *action_space.shape), dtype=np.float32),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=np.float32),
        )

        dataset_dict = dict(real=real_dataset_dict, mb=mb_dataset_dict)
        super().__init__(dataset_dict, seed, observation_key)

        self._real_size = 0
        self._mb_size = 0

        self._real_capacity = capacity
        self._mb_capacity = capacity

        self._real_insert_index = 0
        self._mb_insert_index = 0

    def __len__(self):
        return self._real_size + self._mb_size

    def insert_real(self, data_dict: DatasetDict) -> None:
        _insert_recursively(
            self._dataset_dict["real"], data_dict, self._real_insert_index
        )

        self._real_insert_index = (self._real_insert_index + 1) % self._real_capacity
        self._real_size = min(self._real_size + 1, self._real_capacity)

    def insert_mb(self, data_dict: DatasetDict) -> None:
        _insert_recursively(self._dataset_dict["mb"], data_dict, self._mb_insert_index)

        self._mb_insert_index = (self._mb_insert_index + 1) % self._mb_capacity
        self._mb_size = min(self._mb_size + 1, self._mb_capacity)

    def insert_trajectory(self, data_dict: DatasetDict, real: bool = True) -> None:
        traj_len = data_dict["observations"].shape[0]

        for i in range(traj_len):
            item_dict = jax.tree_util.tree_map(lambda x: x[i], data_dict)
            if real:
                self.insert_real(item_dict)
            else:
                self.insert_mb(item_dict)

    def insert_batch_of_trajectories(
        self, data_dict: DatasetDict, real: bool = True
    ) -> None:
        num_trajs = data_dict["observations"].shape[0]

        for i in range(num_trajs):
            traj_dict = jax.tree_util.tree_map(lambda x: x[i], data_dict)
            self.insert_trajectory(traj_dict, real=real)

    def sample(
        self,
        batch_size: int,
        from_real: bool = False,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> Batch:
        """Samples from the real buffer."""

        buffer_size = self._real_size if from_real else self._mb_size

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(buffer_size, size=batch_size)
            else:
                indx = self.np_random.randint(buffer_size, size=batch_size)
        else:
            assert len(indx) == batch_size, "not the same batch size."

        key = "real" if from_real else "mb"
        dataset_dict = self._dataset_dict[key]

        if keys is None:
            keys = dataset_dict.keys()

        batch_dict = dict()
        for k in keys:
            if isinstance(dataset_dict[k], Mapping):
                batch_dict[k] = _sample(dataset_dict[k], indx)
            else:
                batch_dict[k] = dataset_dict[k][indx]

        batch = _dict_to_batch(batch_dict, self._observation_key)
        del batch_dict

        return batch

    def save(self, path: str) -> None:
        """Saves the buffer's dataset dict to the requested path."""

        with open(path, "wb") as f:
            payload = {
                "dataset_dict": self._dataset_dict,
                "real_size": self._real_size,
                "mb_size": self._mb_size,
            }
            dill.dump(payload, f, protocol=2)

    def load(self, path: str) -> None:
        """Loads the dataset dict from the requested path."""

        with open(path, "rb") as f:
            payload = dill.load(f)

            self._dataset_dict = payload["dataset_dict"]
            self._real_size = payload["real_size"]
            self._mb_size = payload["mb_size"]
