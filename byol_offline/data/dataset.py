import chex
import jax
import jax.numpy as jnp
import gym
from gym.utils import seeding
from typing import (
    NamedTuple,
    Optional,
    List,
    Tuple,
    Union,
    Mapping,
    Generator,
    Iterable
)
import random
import collections
import numpy as np

from byol_offline.types import DatasetDict

"""Basic dataset utilities."""


class Batch(NamedTuple):
    """Base batch."""
    
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    

class SequenceBatch(NamedTuple):
    """Sequence batch, where everything has an additional time dimension prepended."""
    
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    
    
def _preprocess(batch: Union[Batch, SequenceBatch]) -> Union[Batch, SequenceBatch]:
    """Preprocesses the data batch."""
    
    def _process_image(x: chex.Array) -> chex.Array:
        return x.astype(jnp.float32) / 255.0 - 0.5
    
    return jax.tree_util.tree_map(
        lambda x: _process_image(x) if x.dtype == jnp.uint8 else x,
        batch
    )
    

def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    """Checks all dataset attributes have the same number of items. Returns common length if true."""
    
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    
    return dataset_len


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()
    
    
def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


def _stack_dicts(dataset_dicts: List[DatasetDict], axis: int = 0) -> DatasetDict:
    """Stacks a bunch of dicts together into one large one."""

    stacked = dict()
    for k in dataset_dicts[0].keys():
        if isinstance(dataset_dicts[0][k], Mapping):
            stacked[k] = _stack_dicts([dataset_dict[k] for dataset_dict in dataset_dicts], axis=axis)
        else:
            stacked[k] = np.stack([dataset_dict[k] for dataset_dict in dataset_dicts], axis=axis)
            
    return stacked


def _dict_to_batch(
    batch_dict: DatasetDict, observation_key: Optional[str] = None, is_sequence: bool = False,
) -> Union[Batch, SequenceBatch]:
    """Takes a batch dictionary, and turns it into a batch."""
    
    if "observations" in batch_dict and batch_dict["observations"] is not None:
        if observation_key is not None:
            assert isinstance(batch_dict["observations"], Mapping)
            assert observation_key in batch_dict["observations"].keys()
            
            if "next_observations" in batch_dict and len(batch_dict["next_observations"]) > 0:
                assert isinstance(batch_dict["next_observations"], Mapping)
                assert observation_key in batch_dict["next_observations"].keys()
                
                observations = batch_dict["observations"][observation_key]
                next_observations = batch_dict["next_observations"][observation_key]
            else:
                observations = batch_dict["observations"][observation_key][..., :-1]
                next_observations = batch_dict["observations"][observation_key][..., 1:]
                
            # flatten observations
            observations = np.reshape(observations, observations.shape[:-2] + (-1,))
            next_observations = np.reshape(next_observations, next_observations.shape[:-2] + (-1,))
        else:
            observations = batch_dict["observations"]
            next_observations = batch_dict["next_observations"]
    else:
        # unused, we need a placeholder
        observations = np.empty(1)
        next_observations = np.empty(1)
    
    cls = SequenceBatch if is_sequence else Batch
    return cls(
        observations=observations, actions=batch_dict["actions"],
        rewards=batch_dict["rewards"], next_observations=next_observations,
        dones=batch_dict["dones"], masks=batch_dict["masks"]
    )
    
    
def _batch_to_dict(batch: Batch, observation_key: Optional[str] = None) -> DatasetDict:
    """Reverse of `_dict_to_batch`. Assumes returned dict only has one key."""
    
    assert observation_key is not None
    batch_dict = dict()
    
    for k in ["actions", "rewards", "dones", "masks"]:
        batch_dict[k] = getattr(batch, k)
        
    batch_dict["observations"] = {observation_key: batch.observations}
    batch_dict["next_observations"] = {observation_key: batch.next_observations}
    
    return batch_dict


class Dataset(object):
    """Base dataset class."""
    
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None, observation_key: Optional[str] = None):
        
        if len(dataset_dict.keys()) > 0:
            for key in ["observations", "actions", "rewards", "next_observations", "dones", "masks"]:
                assert key in dataset_dict, "Not an appropriate dataset dict."
        
        self._dataset_dict = dataset_dict
        self._len = _check_lengths(dataset_dict)
        
        # do seeding similar to OpenAI Gym
        self._np_random = None
        self._seed = None
        
        if seed is None:
            self.seed(seed)
            
        self._observation_key = observation_key
        self._trajectory_info = None
        
    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        
        return self._np_random
    
    
    def seed(self, seed: Optional[int] = None) -> List:
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]
    
    
    def __len__(self):
        return self._len
    
    
    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> Batch:
        """Samples from the dataset."""
        
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)
        else:
            assert len(indx) == batch_size, "not the same batch size."
            
        if keys is None:
            keys = self._dataset_dict.keys()
        
        batch_dict = dict()
        for k in keys:
            if isinstance(self._dataset_dict[k], Mapping):
                batch_dict[k] = _sample(self._dataset_dict[k], indx)
            else:
                batch_dict[k] = self._dataset_dict[k][indx]
                
        # print(jax.tree_util.tree_map(lambda x: x.shape, batch_dict))
                
        # group everything that is not [observations, actions, rewards, next_observations, dones, masks] into `extras`
        batch = _dict_to_batch(batch_dict, self._observation_key)
        del batch_dict
        
        return batch
    
    
    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        keys: Optional[Iterable[str]] = None,
    ) -> SequenceBatch:
        """Samples a batch of sequences from the dataset."""
        
        if self._trajectory_info is None:
            self._trajectory_info = self._trajectory_boundaries_and_returns()
            
        episode_starts, episode_ends, _ = self._trajectory_info
        
        # now we have to find sequences which are long enough
        good_trajectory_indices = [
            (s, e) for s, e in zip(episode_starts, episode_ends)
            if e - s >= sequence_length
        ]
        
        # now we choose good starting indices for each datapoint
        starting_indices = []
        for _ in range(batch_size):
            s, e = random.choice(good_trajectory_indices) # e is exclusive, so max is e - sequence_length
            L = e - s
            offset = np.random.randint(0, L - sequence_length) if L > sequence_length else 0
            assert s + offset + sequence_length < e
            
            starting_indices.append(s + offset)
            
        if keys is None:
            keys = self._dataset_dict.keys()
            
        sequence_batch_dicts = []
        for starting_index in starting_indices:
            indx = np.arange(starting_index, starting_index + sequence_length)
            
            batch_dict = dict()
            for k in keys:
                if isinstance(self._dataset_dict[k], Mapping):
                    batch_dict[k] = _sample(self._dataset_dict[k], indx)
                else:
                    batch_dict[k] = self._dataset_dict[k][indx]
            
            sequence_batch_dicts.append(batch_dict)
            
        sequence_batch_dict = _stack_dicts(sequence_batch_dicts, axis=1)
        del sequence_batch_dicts
        
        return _dict_to_batch(sequence_batch_dict, self._observation_key, is_sequence=True)
    
    
    def _trajectory_boundaries_and_returns(self) -> Tuple[List, List, List]:
        """Gets trajectory boundaries and returns."""
        
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self._dataset_dict["rewards"][i]

            if self._dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0
        
        assert len(episode_starts) == len(episode_ends) == len(episode_returns)
        return episode_starts, episode_ends, episode_returns
    

class ReplayBuffer(Dataset):
    """Standard replay buffer."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        observation_key: Optional[str] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space
            
        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict, observation_key=observation_key)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        
    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict) -> None:
        _insert_recursively(self._dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: Mapping = {}) -> Generator[Batch, None, None]:
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n: int) -> None:
            for _ in range(n):
                if "sequence_length" in sample_args:
                    data = self.sample_sequences(**sample_args)
                else:
                    data = self.sample(**sample_args)
                
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)