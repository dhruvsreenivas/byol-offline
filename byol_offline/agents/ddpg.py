import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import distrax
import gym
import numpy as np
from typing import NamedTuple, Tuple, Mapping
from ml_collections import ConfigDict

from byol_offline.base_learner import ReinforcementLearner
from byol_offline.data import Batch
from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.actor_critic import DDPGActor, DDPGCritic
from byol_offline.types import NetworkFns, DoubleQOutputs, LossFnOutput, MetricsDict

from utils import is_pixel_based

"""Deep deterministic policy gradient trainer implementation."""


class DDPGState(NamedTuple):
    """Training state for DDPG."""
    
    encoder_params: hk.Params
    actor_params: hk.Params
    critic_params: hk.Params
    target_critic_params: hk.Params
    
    encoder_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    
    rng_key: chex.PRNGKey
    
    
def make_ddpg_networks(
    config: ConfigDict,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> NetworkFns:
    """Makes all functions needed for DDPG."""
    
    encoder_config = config.encoder
    
    def encoder_fn(observation: chex.Array) -> chex.Array:
        if is_pixel_based(observation_space):
            pixel_encoder_config = encoder_config.pixel
            if pixel_encoder_config.type == "drqv2":
                return DrQv2Encoder()(observation)
            else:
                return DreamerEncoder(pixel_encoder_config.depth)(observation)
        else:
            state_encoder_config = encoder_config.state
            encoder = hk.nets.MLP(
                state_encoder_config.hidden_dims, activation=jax.nn.swish
            )
            return encoder(observation)
        
    def actor_fn(observation: chex.Array, std: chex.Numeric) -> distrax.Distribution:
        action_dim = action_space.shape[0]
        actor = DDPGActor(action_dim, config.feature_dim, config.ac_hidden_dim)
        return actor(observation, std)
    
    def critic_fn(observation: chex.Array, action: chex.Array) -> DoubleQOutputs:
        critic = DDPGCritic(config.feature_dim, config.ac_hidden_dim)
        return critic(observation, action)
    
    
    encoder = hk.without_apply_rng(hk.transform(encoder_fn))
    actor = hk.without_apply_rng(hk.transform(actor_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))
    
    return encoder, actor, critic


class DDPGLearner(ReinforcementLearner):
    """DDPG trainer."""
    
    def __init__(
        self,
        config: ConfigDict,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        
        # encoder (if we use BYOL-Explore reward, we can use Dreamer encoder for consistency)
        encoder, actor, critic = make_ddpg_networks(config, observation_space, action_space)
        
        # initialization
        key = jax.random.PRNGKey(seed)
        encoder_key, actor_key, critic_key, state_key = jax.random.split(key, 4)
        representation_dim = (
            config.encoder.state.hidden_dims[-1] if not is_pixel_based(observation_space)
            else 20000 if config.encoder.pixel.type == "drqv2"
            else 4096
        )
        
        observations = observation_space.sample()
        if isinstance(observations, Mapping):
            observations = observations["pixels"]
            H, W, C, S = observations.shape
            observations = np.reshape(observations, (H, W, C * S))
        actions = action_space.sample()
        representation_zeros = jnp.zeros((1, representation_dim))
        
        # params
        encoder_params = encoder.init(encoder_key, observations)
        actor_params = actor.init(actor_key, representation_zeros, jnp.zeros(1))
        critic_params = target_critic_params = critic.init(
            critic_key, representation_zeros, actions
        )
        
        # optimizers
        encoder_optimizer = optax.adam(config.encoder_lr)
        encoder_opt_state = encoder_optimizer.init(encoder_params)
        
        actor_optimizer = optax.adam(config.actor_lr)
        actor_opt_state = actor_optimizer.init(actor_params)
        
        critic_optimizer = optax.adam(config.critic_lr)
        critic_opt_state = critic_optimizer.init(critic_params)
        
        # training state
        self._state = DDPGState(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            
            rng_key=state_key
        )
        
        # ---- hyperparameters for training -----
        
        discount = config.discount
        ema = config.ema
        init_std = config.init_std
        final_std = config.final_std
        std_duration = config.std_duration
        std_clip_val = config.std_clip_val
        update_every_steps = config.update_every_steps
        
        # =================== START OF ALL FNS ===================
        
        def stddev_schedule(step: int) -> float:
            """Linear standard deviation schedule."""
            
            mix = jnp.clip(step / std_duration, 0.0, 1.0)
            return (1.0 - mix) * init_std + mix * final_std
        
        def act(
            state: DDPGState, obs: jnp.ndarray, step: int, eval: bool
        ) -> Tuple[DDPGState, chex.Array]:
            """Action selection function."""
            
            sample_key, key = jax.random.split(state.rng_key)
            
            features = encoder.apply(state.encoder_params, obs) # don't need batch dim here
            std = stddev_schedule(step)
            distribution = actor.apply(state.actor_params, features, std)
            
            mean = distribution.mode()
            sample = distribution.sample(seed=sample_key, clip=std_clip_val)
            action = jnp.where(eval, mean, sample)
            
            new_state = self._state._replace(rng_key=key)
            return new_state, action
    
        # =================== WARMSTARTING ===================
        
        def bc_loss(
            encoder_params: hk.Params, actor_params: hk.Params,
            batch: Batch, key: chex.PRNGKey, step: int
        ) -> LossFnOutput:
            """BC warmstarting loss function."""
            
            std = stddev_schedule(step)
            features = encoder.apply(encoder_params, batch.observations) # no need to expand batch dim
            distribution = actor.apply(actor_params, features, std)
            
            sampled_actions = distribution.sample(seed=key, clip=std_clip_val)
            loss = jnp.mean(jnp.square(sampled_actions - batch.actions)) # mse loss, exactly like DrQ + BC
            return loss, {"bc_loss": loss}
        
        
        def bc_update(state: DDPGState, batch: Batch, step: int) -> Tuple[DDPGState, MetricsDict]:
            update_key, state_key = jax.random.split(state.rng_key)
            
            loss_grad_fn = jax.grad(bc_loss, argnums=(0, 1), has_aux=True)
            (encoder_grads, actor_grads), metrics = loss_grad_fn(
                state.encoder_params, state.actor_params,
                batch.observations, batch.actions, update_key, step
            )
            
            # encoder update
            encoder_update, new_encoder_opt_state = encoder_optimizer.update(encoder_grads, state.encoder_opt_state)
            new_encoder_params = optax.apply_updates(state.encoder_params, encoder_update)
            
            # actor update
            actor_update, new_actor_opt_state = actor_optimizer.update(actor_grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(state.actor_params, actor_update)
            
            new_state = state._replace(
                encoder_params=new_encoder_params,
                actor_params=new_actor_params,
                
                encoder_opt_state=new_encoder_opt_state,
                actor_opt_state=new_actor_opt_state,
                
                rng_key=state_key
            )
            return new_state, metrics
        
        # ----- define RL loss/update functions -----
        
        def critic_loss(
            encoder_params: hk.Params, critic_params: hk.Params,
            target_critic_params: hk.Params, actor_params: hk.Params,
            batch: Batch, key: chex.PRNGKey, step: int
        ) -> LossFnOutput:
            """DDPG critic loss."""
            
            # encode observations
            features = encoder.apply(encoder_params, batch.observations)
            next_features = encoder.apply(encoder_params, batch.next_observations)
            
            # get the targets
            std = stddev_schedule(step)
            distribution = actor.apply(actor_params, next_features, std)
            next_actions = distribution.sample(seed=key, clip=std_clip_val)
            
            # get target Q values
            nq1, nq2 = critic.apply(target_critic_params, next_features, next_actions)
            nv = jnp.minimum(nq1, nq2)
            target_q = jax.lax.stop_gradient(batch.rewards + discount * batch.masks * nv)

            # get the current Q values
            q1, q2 = critic.apply(critic_params, features, batch.actions)
            
            # compute loss, and return
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss, {"critic_loss": q_loss}
        
        
        def actor_loss(
            actor_params: hk.Params, encoder_params: hk.Params,
            critic_params: hk.Params, batch: Batch,
            key: chex.PRNGKey, step: int
        ) -> LossFnOutput:
            """Actor loss function."""

            # encode observations
            features = encoder.apply(encoder_params, batch.observations)
            features = jax.lax.stop_gradient(features) # just so we don't have to deal with gradients passing through no matter what

            std = stddev_schedule(step)
            distribution = actor.apply(actor_params, features, std)
            actions = distribution.sample(seed=key, clip=std_clip_val)
            
            q1, q2 = critic.apply(critic_params, features, actions)
            q = jnp.minimum(q1, q2)
            actor_loss = -jnp.mean(q)
            
            return actor_loss, {"actor_loss": actor_loss, "q1": jnp.mean(q1)}
        
        
        def update_critic(
            state: DDPGState, batch: Batch,
            key: chex.PRNGKey, step: int
        ) -> Tuple[DDPGState, MetricsDict]:
            """Update the critic."""
            
            # assume that the transitions is a sequence of consecutive (s, a, r, s', d) tuples
            grad_fn = jax.grad(critic_loss, argnums=(0, 1), has_aux=True)
            
            (encoder_grads, critic_grads), metrics = grad_fn(
                state.encoder_params,
                state.critic_params,
                state.target_critic_params,
                state.actor_params,
                batch, key, step
            )
            
            # update both encoder and critic
            enc_update, new_encoder_opt_state = encoder_optimizer.update(encoder_grads, state.encoder_opt_state)
            new_encoder_params = optax.apply_updates(state.encoder_params, enc_update)
            
            critic_update, new_critic_opt_state = critic_optimizer.update(critic_grads, state.critic_opt_state)
            new_critic_params = optax.apply_updates(state.critic_params, critic_update)
            
            new_state = state._replace(
                encoder_params=new_encoder_params,
                critic_params=new_critic_params,
                
                encoder_opt_state=new_encoder_opt_state,
                critic_opt_state=new_critic_opt_state,
            )
            return new_state, metrics
        
        
        def update_actor(
            state: DDPGState, batch: Batch,
            key: jax.random.PRNGKey, step: int
        ) -> Tuple[DDPGState, MetricsDict]:
            """Update the actor."""
                
            grad_fn = jax.grad(actor_loss, has_aux=True)
            grads, metrics = grad_fn(
                state.actor_params,
                state.encoder_params,
                state.critic_params,
                batch, key, step
            )
            
            # update actor only
            actor_update, new_actor_opt_state = actor_optimizer.update(grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(actor_update, state.actor_params)
            
            new_state = state._replace(
                actor_parms=new_actor_params,
                actor_opt_state=new_actor_opt_state,
            )
            return new_state, metrics
        
        
        def update(state: DDPGState, batch: Batch, step: int) -> Tuple[DDPGState, MetricsDict]:
            """Full update."""
            
            if step % update_every_steps != 0:
                return state, {}
            
            critic_key, actor_key, state_key = jax.random.split(state.rng_key, 3)
            
            # update critic and actor
            state, critic_metrics = update_critic(state, batch, critic_key, step)
            state, actor_metrics = update_actor(state, batch, actor_key, step)
            
            # update target parameters
            new_target_critic_params = optax.incremental_update(
                state.critic_params, state.target_critic_params, ema
            )
            state = state._replace(
                target_critic_params=new_target_critic_params,
                rng_key=state_key
            )
            
            return state, {
                **critic_metrics, **actor_metrics
            }
        
        self._act = act
        self._bc_update = jax.jit(bc_update)
        self._update = jax.jit(update)