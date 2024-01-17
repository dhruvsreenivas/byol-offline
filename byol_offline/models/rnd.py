import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import gym
import numpy as np
from typing import NamedTuple, Tuple, Mapping
from ml_collections import ConfigDict

from byol_offline.data import Batch
from byol_offline.base_learner import Learner
from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.predictors import RNDPredictor
from byol_offline.types import LossFnOutput, MetricsDict

from utils import is_pixel_based

"""RND definition + trainer."""


class RNDState(NamedTuple):
    """Training state for RND."""
    
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState


class ConvRNDModel(hk.Module):
    """RND module for image observations."""
    
    def __init__(self, config: ConfigDict):
        super().__init__()
    
        # nets
        if config.dreamer:
            self._encoder = DreamerEncoder(config.depth)
        else:
            self._encoder = DrQv2Encoder()
            
        self._predictor = RNDPredictor(config.hidden_dim, config.repr_dim)
        
    def __call__(self, observations: chex.Array) -> chex.Array:
        # Observations are expected to be of size [B, H, W, C]
        
        reprs = self._encoder(observations)
        return self._predictor(reprs)

    
class ConvRNDModelWithActions(ConvRNDModel):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        
    def __call__(
        self, observations: chex.Array, actions: chex.Array
    ) -> chex.Array:
        
        reprs = self._encoder(observations)
        reprs = jnp.concatenate([reprs, actions], axis=-1)
        return self._predictor(reprs)


class MLPRNDModel(hk.Module):
    """State-based RND module."""
    
    def __init__(self, config: ConfigDict):
        super().__init__()
        
        self._encoder = hk.nets.MLP(
            config.encoder_dims,
            activation=jax.nn.swish
        )
        self._predictor = RNDPredictor(config.hidden_dim, config.repr_dim)
    
    def __call__(self, observations: chex.Array) -> chex.Array:
        reprs = self._encoder(observations)
        return self._predictor(reprs)


class MLPRNDModelWithActions(MLPRNDModel):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        
    def __call__(self, observations: chex.Array, actions: chex.Array) -> chex.Array:
        oa = jnp.concatenate([observations, actions], axis=-1)
        return super().__call__(oa)

# ==============================================================================

def make_rnd_network(config: ConfigDict, observation_space: gym.Space) -> hk.Transformed:
    """Makes Haiku pure functions for RND trainer."""
    
    def rnd_fn(observations: chex.Array, actions: chex.Array) -> chex.Array:
        if config.cat_actions:
            if is_pixel_based(observation_space):
                module = ConvRNDModelWithActions(config.pixel)
            else:
                module = MLPRNDModelWithActions(config.state)
                
            return module(observations, actions)
        else:
            if is_pixel_based(observation_space):
                module = ConvRNDModel(config.pixel)
            else:
                module = MLPRNDModel(config.state)
            
            return module(observations)
    
    rnd = hk.without_apply_rng(hk.transform(rnd_fn))
    return rnd


class RNDLearner(Learner):
    """RND trainer."""
    
    def __init__(
        self,
        config: ConfigDict,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        
        # initialize
        rnd = make_rnd_network(config, observation_space)
        
        # optimizer
        optimizer = getattr(optax, config.optimizer_class)(config.learning_rate)
        
        base_key = jax.random.PRNGKey(seed)
        
        observations = observation_space.sample()
        if isinstance(observations, Mapping):
            observations = observations["pixels"]
            
            H, W, C, S = observations.shape
            observations = np.reshape(observations, (H, W, C * S))
            observations = np.expand_dims(observations, axis=0)
        
        actions = action_space.sample()
        
        # state creation
        def make_initial_state(key: chex.PRNGKey) -> RNDState:
            online_key, target_key = jax.random.split(key)
            
            params = rnd.init(online_key, observations, actions)
            target_params = rnd.init(target_key, observations, actions)
            
            opt_state = optimizer.init(params)
            
            state = RNDState(
                params=params,
                target_params=target_params,
                opt_state=opt_state
            )
            return state
        
        # if parallelizing, pmap the initial function to make the state to parallelize it
        if config.pmap:
            init_fn = jax.pmap(make_initial_state, axis_name="devices")
        else:
            init_fn = make_initial_state
        
        self._state = init_fn(base_key)
    
        # ----- define loss + update functions -----
        
        def loss_fn(
            params: hk.Params, target_params: hk.Params, batch: Batch
        ) -> LossFnOutput:
            """RND loss function."""
            
            output = rnd.apply(params, batch.observations, batch.actions)
            target_output = rnd.apply(target_params, batch.observations, batch.actions)
            
            if config.l1:
                return jnp.mean(jnp.abs(target_output - output))
            
            # no need to do jax.lax.stop_gradient, as gradient is only taken w.r.t. first param
            loss = jnp.mean(jnp.square(target_output - output))
            return loss, {"loss": loss}

        
        def update(
            state: RNDState, batch: Batch, step: int
        ) -> Tuple[RNDState, MetricsDict]:
            """Update step."""
            
            del step
            
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, metrics = grad_fn(state.params, state.target_params, batch)
            
            # avg gradients across parallel devices
            if config.pmap:
                loss = jax.lax.pmean(loss, axis_name="devices")
                grads = jax.lax.pmean(grads, axis_name="devices")
            
            update, new_opt_state = optimizer.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, update)
            
            new_train_state = RNDState(
                params=new_params,
                target_params=state.target_params,
                opt_state=new_opt_state
            )
            return new_train_state, metrics
    

        def compute_uncertainty(
            state: RNDState, observations: chex.Array, actions: chex.Array, step: int
        ) -> Tuple[chex.Array, RNDState]:
            """Computes RND uncertainty bonus."""
            
            del step
            
            online_output = rnd.apply(state.params, observations, actions)
            target_output = rnd.apply(state.target_params, observations, actions)
            
            if config.l1:
                diff = jnp.abs(target_output - online_output).sum(-1)
            else:
                diff = jnp.square(target_output - online_output).sum(-1)
            
            return jax.lax.stop_gradient(diff)
        
        # define update + uncertainty computation
        # auto jits so don't need to do jax.jit before pmap
        if config.pmap:
            self._update = jax.pmap(update, axis_name="devices")
        else:
            self._update = jax.jit(update)
        
        self._compute_uncertainty = jax.jit(compute_uncertainty)