import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import distrax
import gym
from typing import NamedTuple, Tuple
from ml_collections import ConfigDict

from byol_offline.base_learner import Learner
from byol_offline.data import Batch
from byol_offline.types import LossFnOutput, MetricsDict

"""Simple one-step autoencoder modules."""


class AEState(NamedTuple):
    """Autoencoder training state."""
    
    ae_params: hk.Params
    ae_opt_state: optax.OptState
    rng_key: jax.random.PRNGKey
    

class DeterVAEOutput(NamedTuple):
    """VAE output."""
    
    latent_dist: distrax.Distribution
    rec_output: jnp.ndarray


class CondAE(hk.Module):
    """Conditional AE from https://github.com/shidilrzf/Anti-exploration-RL/blob/1013a85b4b84656a06f86abee01c55a5e08272ee/rlkit/torch/networks.py#L215."""
    
    def __init__(self, config: ConfigDict):
        super().__init__()
        
        assert len(config.observation_dim) == 1, "This CondAE only supports MuJoCo based envs."
        self._state_embed_dim = config.state_embed_dim
        self._action_embed_dim = config.action_embed_dim
        
        self._encoder = hk.nets.MLP(
            [*config.hidden_dims, config.feature_dim],
            activation=jax.nn.relu,
            activate_final=False
        )
        self._decoder = hk.nets.MLP(
            [*config.hidden_dims, config.action_dim],
            activation=jax.nn.relu,
            activate_final=False
        )
        
    def __call__(self, s: chex.Array, a: chex.Array) -> chex.Array:
        # encode
        zs = hk.Linear(self._state_embed_dim)(s)
        za = hk.Linear(self._action_embed_dim)(a)
        z = jnp.concatenate([zs, za], axis=-1)
        z = self._encoder(z)
        
        # decode
        zs = jnp.concatenate([z, s], axis=-1)
        ahat = self._decoder(zs)
        return ahat
    

class VAE(hk.Module):
    """VAE from the Offline RL as Anti-Exploration paper (https://arxiv.org/pdf/2106.06431.pdf)."""
    
    def __init__(self, config: ConfigDict):
        super().__init__()
        
        assert len(config.observation_dim) == 1, """This VAE only supports mujoco based envs."""
        self._feature_dim = config.feature_dim
        
        self._encoder = hk.nets.MLP(
            config.hidden_dims,
            activation=jax.nn.relu,
            activate_final=False
        )
        self._decoder = hk.nets.MLP(
            [*config.hidden_dims, config.action_dim],
            activation=jax.nn.relu,
            activate_final=False
        )
        self._activate_final = config.activate_final
        self._clip_log_std = config.clip_log_std
    
    
    def __call__(self, s: chex.Array, a: chex.Array) -> DeterVAEOutput:
        # encode (embed first apparently)
        sa = jnp.concatenate([s, a], axis=-1)
        sa_rep = self._encoder(sa)
        
        mean = hk.Linear(self._feature_dim)(sa_rep)
        logstd = hk.Linear(self._feature_dim)(sa_rep)
        if self._clip_log_std:
            logstd = jnp.clip(logstd, -4, 15)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(logstd))
        
        # decode
        z = dist.sample(seed=hk.next_rng_key()) # sampling already does reparameterization trick in distrax
        sz = jnp.concatenate([s, z], axis=-1) # CVAE so condition on s
        ahat = self._decoder(sz)
        if self._activate_final:
            ahat = jnp.tanh(ahat)
        
        return DeterVAEOutput(latent_dist=dist, rec_output=ahat)
    
    
def make_ae_network(config: ConfigDict) -> hk.Transformed:
    """Makes Haiku transformed functions for autoencoders."""
    
    def ae_fn(x: chex.Array, a: chex.Array) -> chex.Array:
        if config.ae_type == "vae":
            network_cls = VAE
        else:
            network_cls = CondAE
            
        network = network_cls(config)
        return network(x, a)
    
    ae = hk.transform(ae_fn)
    return ae


class AutoEncoderLearner(Learner):
    """Autoencoder trainer."""
    
    def __init__(
        self,
        config: ConfigDict,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        
        # initialize net function
        ae = make_ae_network(config)
        
        # initialize parameters
        key = jax.random.PRNGKey(seed)
        init_key, state_key = jax.random.split(key)
        observations = observation_space.sample()
        actions = action_space.sample()
        
        params = ae.init(init_key, observations, actions)
        
        opt = getattr(optax, config.optimizer_class)(config.learning_rate)
        opt_state = opt.init(params)
        
        self._state = AEState(
            ae_params=params,
            ae_opt_state=opt_state,
            rng_key=state_key
        )
        
        # hparams
        feature_dim = config.feature_dim
        beta = config.beta
        
        # ----- define loss + update functions and uncertainty bonus -----
        
        def loss_fn(
            params: hk.Params, key: chex.PRNGKey, batch: Batch,
        ) -> LossFnOutput:
            """Autoencoder loss function."""
            
            output = ae.apply(params, key, batch.observations, batch.actions)
            
            if config.ae_type == "cond_ae":
                loss = jnp.mean(jnp.square(output - batch.actions))
                metrics = {"loss": loss}
            else:
                post_dist = output.latent_dist
                rec_output = output.rec_output
                rec_loss = jnp.mean(jnp.square(rec_output - batch.actions))
                
                standard_gaussian = distrax.MultivariateNormalDiag(
                    jnp.zeros((feature_dim,)),
                    jnp.ones((feature_dim,))
                )
                kl = post_dist.kl_divergence(standard_gaussian)
                kl = jnp.mean(kl)
                loss = rec_loss + beta * kl # want to minimize KL and reconstruction loss
                metrics = {"loss": loss, "rec_loss": rec_loss, "kl": kl}
            
            return loss, metrics
        
        
        def update(
            state: AEState, batch: Batch, step: int
        ) -> Tuple[AEState, MetricsDict]:
            """Autoencoder update step."""
            
            del step
            update_key, state_key = jax.random.split(state.rng_key)
            
            loss_grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, metrics = loss_grad_fn(state.ae_params, update_key, batch)
            
            update, new_opt_state = opt.update(grads, state.ae_opt_state)
            new_params = optax.apply_updates(state.ae_params, update)
            
            new_state = AEState(
                ae_params=new_params,
                ae_opt_state=new_opt_state,
                rng_key=state_key
            )
            return new_state, metrics
        
        
        def compute_uncertainty(
            state: AEState, observation: chex.Array, actions: chex.Array
        ) -> Tuple[chex.Array, AEState]:
            """Computes the uncertainty. Additionally returns new training state if RNG key is changed."""
            
            if config.ae_type == "vae":
                sample_key, new_state_key = jax.random.split(state.rng_key)
                
                output = ae.apply(state.ae_params, sample_key, observation, actions)
                ahat = output.rec_output
                uncertainty = jnp.mean(jnp.square(ahat - actions), axis=-1)
            else:
                new_state_key = state.rng_key
                ahat = ae.apply(state.ae_params, state.rng_key, observation, actions)
                uncertainty = jnp.mean(jnp.square(ahat - actions), axis=-1)
                
            new_state = state._replace(rng_key=new_state_key)
            return uncertainty, new_state
        
        
        self._update = jax.jit(update)
        self._compute_uncertainty = jax.jit(compute_uncertainty)