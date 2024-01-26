import abc
import chex
import jax
import dill
from typing import Any, Tuple

from byol_offline.types import MetricsDict

"""Base classes."""

class Learner(abc.ABC):
    """Base learner class."""
    
    def _update(self, *args, step: int, **kwargs) -> Tuple[Any, MetricsDict]:
        """Updates the learner."""
        
        raise NotImplementedError("Update function not implemented.")
    
    def save(self, checkpoint_path: str) -> None:
        """Saves the model to the requested path."""
        
        if jax.process_index() == 0:
            with open(checkpoint_path, "wb") as f:
                dill.dump(self._state, f, protocol=2)
            
    def load(self, checkpoint_path: str):
        """Loads the model from the requested path."""
        
        if jax.process_index() == 0:
            with open(checkpoint_path, "rb") as f:
                self._state = dill.load(f)


class ReinforcementLearner(Learner):
    """Base reinforcement learner class."""
    
    def _act(self, *args, eval: bool, **kwargs) -> Tuple[Any, chex.Array]:
        """Action selection function."""
        
        raise NotImplementedError("Action selection function not implemented.")