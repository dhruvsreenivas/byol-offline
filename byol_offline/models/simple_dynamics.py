import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import gym
from typing import NamedTuple, Sequence
from ml_collections import ConfigDict

from byol_offline.base_learner import Learner
from byol_offline.data import Batch
from byol_offline.types import LossFnOutput, MetricsDict

"""Simple dynamics models."""


class MLPDynamicsModel(hk.Module):
    """Simple MLP dynamics for state-based envs."""
    
    def __init__(self, observation_dim: int, hidden_dims: Sequence[int], act: str = "relu"):
        super().__init__(name="simple_mlp_dynamics")
        
        self._observation_dim = observation_dim
        self._hidden_dims = hidden_dims
        self._activation = getattr(jax.nn, act) if hasattr(jax.nn, act) else getattr(jnp, act)
    
    def __call__(self, obs: chex.Array, action: chex.Array) -> chex.Array:
        obs_action = jnp.concatenate([obs, action], -1)
        x = hk.nets.MLP(
            [*self._hidden_dims, self._observation_dim],
            activation=self._activation
        )(obs_action)
        
        return x
    

class SimpleDynamicsState(NamedTuple):
    """Training state for simple dynamics model."""
    
    params: hk.Params
    opt_state: optax.OptState

    
class SimpleDynamicsLearner(Learner):
    """Simple dynamics trainer."""
    
    def __init__(
        self,
        config: ConfigDict,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        
        assert config.model_type == "mlp_dynamics", "Haven't implemented more complicated models yet. Starting simple :)"
        
        model_fn = lambda s, a: MLPDynamicsModel(config.observation_dim, config.hidden_dim, act="relu")(s, a)
        model = hk.without_apply_rng(hk.transform(model_fn))
        
        # initialization
        key = jax.random.PRNGKey(seed)
        observations = observation_space.sample()
        actions = action_space.sample()
        
        params = model.init(key, observations, actions)
        
        optimizer = optax.sgd(config.learning_rate, momentum=0.9, nesterov=True) if config.optimizer_class == "sgd" else optax.adam(config.learning_rate)
        opt_state = optimizer.init(params)
        
        # initialize training state
        self._state = SimpleDynamicsState(
            params=params, opt_state=opt_state
        )
        
        # ----- define loss and update functions -----

        def loss_fn(params: hk.Params, batch: Batch) -> LossFnOutput:
            """Loss function."""
            
            outputs = model.apply(params, batch.observations, batch.actions)
            
            targets = jnp.where(
                config.train_for_diff,
                batch.next_observations - batch.observations,
                batch.next_observations
            )
            
            loss = ((outputs - targets) ** 2).mean()
            return loss, {"loss": loss}
        
        def update(
            train_state: SimpleDynamicsState, batch: Batch, step: int
        ) -> MetricsDict:
            """Update step."""
            
            del step
            
            loss_grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, metrics = loss_grad_fn(train_state.params, batch)
            
            update, new_opt_state = optimizer.update(grads, train_state.opt_state)
            new_params = optax.apply_updates(train_state.params, update)
            
            new_train_state = SimpleDynamicsState(
                params=new_params,
                opt_state=new_opt_state
            )
            return new_train_state, metrics

        self._update = jax.jit(update)