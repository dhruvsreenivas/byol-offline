import abc
import jax.numpy as jnp

from memory.replay_buffer import Transition

class Agent(abc.ABC):
    """Agent base class."""
    def __init__(self, byol=None, rnd=None):
        pass
    
    def _act(self, obs: jnp.ndarray, step: int, eval_mode: bool):
        """
        Returns the action that the agent will take in the environment.
        
        obs [jnp.ndarray]: observation
        step [int]: global step
        eval_mode [bool]: whether to act in eval mode or train mode
        """
    
    def _update(self, train_state, transitions: Transition, step: int):
        """
        Updates the agent with a batch of transitions.
        
        train_state [TrainState]: agent train state
        transitions [Transition]: batch of transitions
        step [int]: global step
        """