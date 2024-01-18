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
from byol_offline.networks.actor_critic import SACActor, SACCritic
from byol_offline.types import NetworkFns, DoubleQOutputs, LossFnOutput, MetricsDict

from utils import is_pixel_based

"""Soft actor-critic trainer implementation."""


class SACState(NamedTuple):
    """Training state for SAC."""
    
    encoder_params: hk.Params
    actor_params: hk.Params
    critic_params: hk.Params
    target_critic_params: hk.Params
    log_temperature: chex.Array
    
    encoder_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    temperature_opt_state: optax.OptState
    
    rng_key: chex.PRNGKey
    

def make_sac_networks(
    config: ConfigDict,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> NetworkFns:
    """Make networks for SAC."""
    
    encoder_config = config.encoder
    
    def encoder_fn(observation: chex.Array) -> chex.Array:
        if is_pixel_based(observation_space):
            pixel_encoder_config = encoder_config.pixel
            if pixel_encoder_config.dreamer:
                return DreamerEncoder(pixel_encoder_config.depth)(observation)
            else:
                return DrQv2Encoder()(observation)
        else:
            state_encoder_config = encoder_config.state
            encoder = hk.nets.MLP(
                state_encoder_config.hidden_dims, activation=jax.nn.swish
            )
            return encoder(observation)
        
    def actor_fn(observation: chex.Array) -> distrax.Distribution:
        action_dim = action_space.shape[0]
        actor = SACActor(action_dim, config.ac_hidden_dim)
        return actor(observation)
    
    def critic_fn(observation: chex.Array, action: chex.Array) -> DoubleQOutputs:
        critic = SACCritic(config.ac_hidden_dim)
        return critic(observation, action)
    
    encoder = hk.without_apply_rng(hk.transform(encoder_fn))
    actor = hk.without_apply_rng(hk.transform(actor_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))
    
    return encoder, actor, critic
    
    
class SACLearner(ReinforcementLearner):
    """Soft actor-critic."""
    
    def __init__(
        self,
        config: ConfigDict,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        
        # set up networks
        encoder, actor, critic = make_sac_networks(config, observation_space, action_space)
        
        # initialization
        rng = jax.random.PRNGKey(seed)
        encoder_key, actor_key, critic_key, state_key = jax.random.split(rng, 4)
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
        
        encoder_params = encoder.init(encoder_key, observations)
        actor_params = actor.init(actor_key, representation_zeros, jnp.zeros(1))
        critic_params = target_critic_params = critic.init(
            critic_key, representation_zeros, actions
        )
        
        log_temperature = jnp.asarray(0., dtype=jnp.float32)
        
        # optimizers
        encoder_optimizer = optax.adam(config.encoder_lr)
        encoder_opt_state = encoder_optimizer.init(encoder_params)
        
        actor_optimizer = optax.adam(config.actor_lr)
        actor_opt_state = actor_optimizer.init(actor_params)
        
        critic_optimizer = optax.adam(config.critic_lr)
        critic_opt_state = critic_optimizer.init(critic_params)
        
        temperature_optimizer = optax.adam(config.alpha_lr)
        temperature_opt_state = temperature_optimizer.init(log_temperature)
        
        # train state
        self._state = SACState(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            log_temperature=log_temperature,
            
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            temperature_opt_state=temperature_opt_state,
            
            rng_key=state_key
        )
        
        # ----- hyperparameters -----
        discount = config.discount
        ema = config.ema
        target_update_frequency = config.target_update_frequency
        target_entropy = -action_space.shape[0] / 2
        
        # =================== Action function ===================
        
        def act(
            state: SACState, observation: chex.Array,
            step: int, eval: bool = False
        ) -> Tuple[SACState, chex.Array]:
            del step
            
            sample_key, key = jax.random.split(state.rng_key)
            
            features = encoder.apply(state.encoder_params, observation)
            distribution = actor.apply(state.actor_params, features)
            
            mean = distribution.mode()
            sample = distribution.sample(seed=sample_key)
            action = jnp.where(eval, mean, sample)
            
            new_state = state._replace(rng_key=key)
            return new_state, action
        
        # =================== WARMSTARTING ===================
        
        def bc_loss(
            encoder_params: hk.Params,
            actor_params: hk.Params,
            batch: Batch,
            key: chex.PRNGKey,
            step: int
        ) -> LossFnOutput:
            """BC warmstart loss."""
            
            del step

            features = encoder.apply(encoder_params, batch.observations) # no need to expand batch dim
            distribution = actor.apply(actor_params, features)
            
            sampled_actions = distribution.sample(seed=key)
            loss = jnp.mean(jnp.square(sampled_actions - batch.actions)) # mse loss, exactly like DrQ + BC
            return loss, {"bc_loss": loss}
        
        
        def bc_update(state: SACState, batch: Batch, step: int) -> Tuple[SACState, MetricsDict]:
            """BC update step."""
            
            update_key, key = jax.random.split(state.rng_key)
            
            grad_fn = jax.grad(bc_loss, argnums=(0, 1), has_aux=True)
            (encoder_grads, actor_grads), metrics = grad_fn(state.encoder_params, state.actor_params, batch, update_key, step)
            
            # encoder update
            enc_update, new_enc_opt_state = encoder_optimizer.update(encoder_grads, state.encoder_opt_state)
            new_enc_params = optax.apply_updates(state.encoder_params, enc_update)
            
            # actor update
            act_update, new_act_opt_state = actor_optimizer.update(actor_grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(state.actor_params, act_update)
            
            # get new state
            new_train_state = state._replace(
                encoder_params=new_enc_params,
                actor_params=new_actor_params,
                
                encoder_opt_state=new_enc_opt_state,
                actor_opt_state=new_act_opt_state,
                
                rng_key=key
            )
            return new_train_state, metrics
        
        # =================== AGENT LOSS/UPDATE FUNCTIONS ===================
        
        def critic_loss(
            encoder_params: hk.Params,
            critic_params: hk.Params,
            target_critic_params: hk.Params,
            actor_params: hk.Params,
            temperature: chex.Array,
            batch: Batch,
            key: chex.PRNGKey,
        ) -> LossFnOutput:
            """Critic loss function."""
            
            # encode observations
            features = encoder.apply(encoder_params, batch.observations)
            next_features = encoder.apply(encoder_params, batch.next_observations)
            
            # get the targets
            distribution = actor.apply(actor_params, next_features)
            next_actions, next_log_probs = distribution.sample_and_log_prob(seed=key)
            
            nq1, nq2 = critic.apply(target_critic_params, next_features, next_actions)
            nv = jnp.squeeze(jnp.minimum(nq1, nq2) - temperature * next_log_probs)
            target_q = jax.lax.stop_gradient(batch.rewards + discount * batch.masks * nv)

            # get the actual q values
            q1, q2 = critic.apply(critic_params, features, batch.actions)
            
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss, {"critic_loss": q_loss}
        

        def actor_loss(
            actor_params: hk.Params,
            encoder_params: hk.Params,
            critic_params: hk.Params,
            temperature: chex.Array,
            batch: Batch,
            key: chex.PRNGKey,
        ) -> LossFnOutput:
            """Actor loss function."""
            
            # encode observations
            features = encoder.apply(encoder_params, batch.observations)
            features = jax.lax.stop_gradient(features) # no gradient flow through encoders

            distribution = actor.apply(actor_params, features)
            actions, log_probs = distribution.sample_and_log_prob(seed=key)
            
            q1, q2 = critic.apply(critic_params, features, actions)
            min_q = jnp.minimum(q1, q2)
            
            actor_loss = (temperature * log_probs - min_q).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}
        
        
        def temperature_loss(
            log_temperature: chex.Array,
            entropy: chex.Array,
        ) -> LossFnOutput:
            """Temperature loss function."""
            
            alpha_loss = jnp.exp(log_temperature) * (entropy - target_entropy).mean()
            return alpha_loss, {"alpha_loss": alpha_loss, "temperature": jnp.exp(log_temperature)}
        
        
        def update_critic(
            state: SACState,
            batch: Batch,
            key: chex.PRNGKey,
        ) -> Tuple[SACState, MetricsDict]:
            """Update the critic."""
            
            grad_fn = jax.grad(critic_loss, argnums=(0, 1), has_aux=True)
            
            (encoder_grads, critic_grads), metrics = grad_fn(
                state.encoder_params,
                state.critic_params,
                state.target_critic_params,
                state.actor_params,
                jnp.exp(state.log_temperature),
                batch, key
            )
            
            # update both encoder and critic
            encoder_update, new_encoder_opt_state = encoder_optimizer.update(encoder_grads, state.encoder_opt_state)
            new_encoder_params = optax.apply_updates(state.encoder_params, encoder_update)
            
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
            state: SACState,
            batch: Batch,
            key: chex.PRNGKey,
        ) -> Tuple[SACState, MetricsDict]:
            """Update the actor."""
            
            grad_fn = jax.grad(actor_loss, has_aux=True)
            
            grads, metrics = grad_fn(
                state.actor_params,
                state.encoder_params,
                state.critic_params,
                jnp.exp(state.log_temperature), 
                batch, key
            )
            
            # update actor params
            actor_update, new_actor_opt_state = actor_optimizer.update(grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(state.actor_params, actor_update)
            
            new_state = state._replace(
                actor_params=new_actor_params,
                actor_opt_state=new_actor_opt_state,
            )
            return new_state, metrics
        
        
        def update_temperature(
            state: SACState,
            entropy: chex.Array,
        ) -> Tuple[SACState, MetricsDict]:
            """Temperature update step."""
            
            grad_fn = jax.grad(temperature_loss, has_aux=True)
            grads, metrics = grad_fn(state.log_temperature, entropy)
            
            # update alpha
            temperature_update, new_temperature_opt_state = temperature_optimizer.update(grads, state.temperature_opt_state)
            new_log_temperature = optax.apply_updates(state.log_temperature, temperature_update)
            
            new_state = state._replace(
                log_temperature=new_log_temperature,
                temperature_opt_state=new_temperature_opt_state
            )
            return new_state, metrics
        
        def update(state: SACState, batch: Batch, step: int) -> Tuple[SACState, MetricsDict]:
            """Full SAC update."""
            
            critic_key, actor_key, state_key = jax.random.split(state.rng_key, 3)
            
            # critic update
            state, critic_metrics = update_critic(
                state, batch, critic_key
            )
            
            # actor update
            state, actor_metrics = update_actor(
                state, batch, actor_key
            )
            
            # temperature update
            state, temperature_metrics = update_temperature(
                state, actor_metrics["entropy"]
            )
            
            # update critic target
            new_target_critic_params = optax.incremental_update(
                state.critic_params, state.target_critic_params, ema
            )
            new_target_critic_params = jnp.where(
                step % target_update_frequency == 0,
                new_target_critic_params,
                state.target_critic_params
            )
            state = state._replace(
                target_critic_params=new_target_critic_params,
                rng_key=state_key
            )
            
            # logging
            metrics = {**critic_metrics, **actor_metrics, **temperature_metrics}
            return state, metrics

        self._act = jax.jit(act)
        self._bc_update = jax.jit(bc_update)
        self._update = jax.jit(update)