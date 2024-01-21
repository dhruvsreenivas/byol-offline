import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import distrax
import gym
import numpy as np
from ml_collections import ConfigDict
from typing import Tuple, NamedTuple, Union, Mapping

from byol_offline.base_learner import Learner
from byol_offline.data import SequenceBatch, _preprocess
from byol_offline.distributions import OneHotDistribution
from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.decoder import DrQv2Decoder, DreamerDecoder
from byol_offline.networks.rnn import RSSM, RSSMState, PosteriorInput, RSSMFnOutput
from byol_offline.networks.predictors import BYOLPredictor
from byol_offline.models.byol_utils import *
from byol_offline.types import ImagineOutput, ObserveOutput, MetricsDict, LossFnOutput

from utils import seq_batched_like, is_pixel_based, broadcast_to_local_devices

"""BYOL-Explore + Dreamer world model definition + trainer."""


class BYOLState(NamedTuple):
    """Training state for BYOL-Explore world model."""
    
    params: hk.Params
    target_params: hk.Params
    
    opt_state: optax.OptState
    rng_key: chex.PRNGKey
    
    
class DreamerOutput(NamedTuple):
    """World model output."""
    
    post_observation_mean: chex.Array
    post_reward_mean: chex.Array
    posts: chex.Array
    
    prior_observation_mean: chex.Array
    prior_reward_mean: chex.Array
    priors: chex.Array
    
    
class BYOLOutput(NamedTuple):
    """Outputs for BYOL loss."""

    predicted_latents: chex.Array
    target_embeddings: chex.Array
    

class WorldModelOutput(NamedTuple):
    """All outputs for world model."""
    
    dreamer: DreamerOutput
    byol: BYOLOutput


class ConvWorldModel(hk.Module):
    """World model for DMC tasks. Primarily inspired by DreamerV2 and DrQv2 repositories."""
    
    def __init__(self, config: ConfigDict, observation_channels: int):
        super().__init__()
        
        initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        
        # nets
        if config.dreamer:
            self._encoder = DreamerEncoder(config.depth)
        else:
            self._encoder = DrQv2Encoder()
        
        # rssm
        self._rssm = RSSM(config.rssm) # uses both closed gru and open gru here, closed for dreamer loss, open for byol loss
        
        # prediction heads
        if config.dreamer:
            self._decoder = DreamerDecoder(observation_channels, config.depth)
            self._byol_predictor = BYOLPredictor(1024)
        else:
            self._decoder = DrQv2Decoder(observation_channels)
            self._byol_predictor = BYOLPredictor(20000)
        
        self._reward_predictor = hk.nets.MLP(
            [*config.reward_hidden_dims, 1],
            w_init=initializer
        )
    
    # ===== eval functions (imagining/observing one step, many steps, etc...) =====
    
    def _encode(self, observations: chex.Array) -> chex.Array:
        """Encodes the observations."""
        
        observations = observations.astype(jnp.float32)
        return self._encoder(observations)
    
    
    def _decode(self, features: chex.Array) -> chex.Array:
        """Decodes features. Assumes double batching (as the only time we do this is when computing uncertainty)."""
        
        return hk.BatchApply(self._decoder)(features)
        
    
    def _open_gru_rollout(self, actions: chex.Array, state: RSSMState) -> chex.Array:
        """
        Rolls out the open GRU (in this case, the RSSM recurrent unit) over all actions, starting at initial state `state`.
        
        Only gets the deterministic part of the output, similar to BYOL-Explore.
        """
        
        def _scan_fn(carry: RSSMState, action: chex.Array) -> Tuple[RSSMState, chex.Array]:
            state = carry
            new_state, _ = self._rssm._onestep_prior(action, state)
            return new_state, new_state[-2] # new state (deter, stoch), deter state
        
        _, deter_states = hk.scan(_scan_fn, state, actions)
        return deter_states
    
    
    def _onestep_imagine(self, action: chex.Array, state: RSSMState) -> Tuple[RSSMState, ImagineOutput]:
        """Does one step of imagination by rolling out from the prior.
        
        Returns:
            new_state [RSSMState]: Resulting RSSM state.
            new_feature [chex.Array]: Latent feature from RSSM state.
            reward_mean [chex.Array]: Predicted reward distribution mean.
        """
        
        new_state, _ = self._rssm._onestep_prior(action, state)
        new_feature = self._rssm._get_feature(new_state)
        
        # run feature through only reward decoder
        reward_mean = self._reward_predictor(new_feature)
        
        return new_state, (new_feature, reward_mean)
    
    
    def _onestep_observe(self, emb: chex.Array, action: chex.Array, state: RSSMState) -> Tuple[RSSMState, ObserveOutput]:
        """Does one step of observation by rolling out from the posterior. Assumes observation is a latent state.
        
        Returns:
            new_state [RSSMState]: Resulting RSSM state.
            new_feature [chex.Array]: Latent feature.
            reward_mean [chex.Array]: Predicted reward distribution mean.
        """
        
        new_state, _ = self._rssm._onestep_posterior(emb, action, state)
        new_feature = self._rssm._get_feature(new_state)
        
        # run feature through reward decoder
        reward_mean = self._reward_predictor(new_feature)
        
        return new_state, (new_feature, reward_mean)
    
    # ===== things used in training/to define loss function in trainer =====
    
    def _dreamer_forward(self, observations: chex.Array, actions: chex.Array) -> DreamerOutput:
        """Full forward pass to get Dreamer outputs."""
        
        # observations should be of shape [T, B, H, W, C], actions of shape [T, B, action_dim]
        observations = observations.astype(jnp.float32)
        
        # first get embeddings of observations
        embeds = hk.BatchApply(self._encoder)(observations)
        
        # roll out RSSM. Always start with initial state at zeros.
        rssm_output = self._rssm(embeds, actions, None)
        
        # === posts: [T, B, dist_dim], priors: [T, B, dist_dim], features: [T, B, feature_dim] ===
        
        # get posterior outputs
        post_img_mean = hk.BatchApply(self._decoder)(rssm_output.post_features) # [T, B, H, W, C]
        post_reward_mean = hk.BatchApply(self._reward_predictor)(rssm_output.post_features)
        post_reward_mean = jnp.squeeze(post_reward_mean)
        
        # get prior outputs
        prior_img_mean = hk.BatchApply(self._decoder)(rssm_output.prior_features)
        prior_reward_mean = hk.BatchApply(self._reward_predictor)(rssm_output.prior_features)
        prior_reward_mean = jnp.squeeze(prior_reward_mean)
        
        # return
        return DreamerOutput(
            post_observation_mean=post_img_mean, post_reward_mean=post_reward_mean, posts=rssm_output.posts,
            prior_observation_mean=prior_img_mean, prior_reward_mean=prior_reward_mean, priors=rssm_output.priors
        )

    
    def _byol_forward(self, observations: chex.Array, actions: chex.Array) -> BYOLOutput:
        """Full output for BYOL loss computation."""
        
        # as above, observations should be of shape [T, B, H, W, C], actions of shape [T, B, action_dim]
        observations = observations.astype(jnp.float32)
        
        # first get embeddings
        B = observations.shape[1]
        embeds = hk.BatchApply(self._encoder)(observations)
        init_state = self._rssm._init_state(B)
        
        # we first use the RSSM encoder to do one-step before doing open-loop rollouts
        embed, action = embeds[0], actions[0] # x_0, a_0
        first_state, _ = self._rssm._onestep_posterior(embed, action, init_state) # h_0
        
        first_deter = first_state[-2]
        latent = self._byol_predictor(first_deter)
        latent = jnp.expand_dims(latent, axis=0)
        
        # --- collect h_t, latents (BYOL stuff) through open loop rollout ---
        deter_states = self._open_gru_rollout(actions[1:], first_state)
        latents = self._byol_predictor(deter_states)
        pred_latents = jnp.concatenate([latent, latents], axis=0) # [T, B, embed_dim] for both pred_latents and embeds
        
        # return
        return BYOLOutput(
            predicted_latents=pred_latents, target_embeddings=embeds
        )
    
    
    def _process_to_latent(self, observations: chex.Array, actions: chex.Array) -> chex.Array:
        """Processes the observations and actions to given features."""
        
        observations = observations.astype(jnp.float32)
        B = observations.shape[1]
        
        # first get embeddings
        embeds = hk.BatchApply(self._encoder)(observations)
        
        # now we scan with posterior model across time axis
        # this is the same as the RSSM posterior feature scan function
        def _posterior_feature_scan_fn(carry: RSSMState, embed_action: PosteriorInput) -> RSSMFnOutput:
            state = carry
            embed, action = embed_action
            
            new_state, _ = self._rssm._onestep_posterior(embed, action, state)
            new_feature = self._rssm._get_feature(new_state)
            return new_state, new_feature
        
        init = self._rssm._init_state(B)
        _, features = hk.scan(_posterior_feature_scan_fn, init, (embeds, actions))
        
        return features
        
    
    def __call__(self, observations: chex.Array, actions: chex.Array) -> WorldModelOutput:
        dreamer_out = self._dreamer_forward(observations, actions)
        byol_out = self._byol_forward(observations, actions)
        
        return WorldModelOutput(dreamer=dreamer_out, byol=byol_out)


class MLPWorldModel(hk.Module):
    """World model for D4RL tasks. Primarily inspired by MOPO repository."""
    
    def __init__(self, config: ConfigDict, observation_dim: int):
        super().__init__()
        
        initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        
        # nets
        self._encoder = hk.nets.MLP(
            [*config.hidden_dims, config.repr_dim],
            activation=jax.nn.swish
        )
        
        # RSSM
        self._rssm = RSSM(config.rssm) # uses both closed gru and open gru here, closed for dreamer loss, open for byol loss
        
        # predictors
        self._decoder = hk.nets.MLP(
            [*config.hidden_dims, observation_dim],
            activation=jax.nn.swish
        )
        self._reward_predictor = hk.nets.MLP(
            [*config.reward_hidden_dims, 1],
            w_init=initializer
        )
        self._byol_predictor = BYOLPredictor(config.repr_dim)
        
    # ===== eval functions (imagining/observing one step, many steps, etc...) =====
    
    def _encode(self, observations: chex.Array) -> chex.Array:
        """Encodes the observations."""
        
        return self._encoder(observations)
    
    
    def _decode(self, features: chex.Array) -> chex.Array:
        """Decodes features. Assumes double batching (as the only time we do this is when computing uncertainty)."""
        
        return hk.BatchApply(self._decoder)(features)
        
    
    def _open_gru_rollout(self, actions: chex.Array, state: RSSMState) -> chex.Array:
        """
        Rolls out the open GRU (in this case, the RSSM recurrent unit) over all actions, starting at init state `state`.
        
        Only gets the deterministic part of the output, similar to BYOL-Explore.
        """
        
        def _scan_fn(carry: RSSMState, action: chex.Array) -> Tuple[RSSMState, chex.Array]:
            state = carry
            new_state, _ = self._rssm._onestep_prior(action, state)
            return new_state, new_state[-2]
        
        _, deter_states = hk.scan(_scan_fn, state, actions)
        return deter_states
    
    
    def _onestep_imagine(self, action: chex.Array, state: RSSMState) -> Tuple[RSSMState, ImagineOutput]:
        """Does one step of imagination by rolling out from the prior.
        
        Returns:
            new_state [RSSMState]: Resulting RSSM state.
            new_feature [chex.Array]: Latent feature.
            reward_mean [chex.Array]: Predicted reward distribution mean.
        """
        
        new_state, _ = self._rssm._onestep_prior(action, state)
        new_feature = self._rssm._get_feature(new_state)
        
        # pass through reward decoder only
        reward_mean = self._reward_predictor(new_feature)
        
        return new_state, (new_feature, reward_mean)
    
    
    def _onestep_observe(self, emb: chex.Array, action: chex.Array, state: RSSMState) -> Tuple[RSSMState, ObserveOutput]:
        """Does one step of observation by rolling out from the posterior. Assumes observation is a latent state.
        
        Returns:
            new_state [RSSMState]: Resulting RSSM state.
            new_feature [chex.Array]: Latent feature.
            reward_mean [chex.Array]: Predicted reward distribution mean.
        """
        
        new_state, _ = self._rssm._onestep_posterior(emb, action, state)
        new_feature = self._rssm._get_feature(new_state)
        
        # pass through reward decoder only
        reward_mean = self._reward_predictor(new_feature)
        
        return new_state, (new_feature, reward_mean)
    
    # ===== forward passes used in training/to define loss function in trainer =====
    
    def _dreamer_forward(self, observations: chex.Array, actions: chex.Array) -> DreamerOutput:
        """Forward pass to compute outputs for Dreamer loss."""
        
        # observations should be of shape [T, B, observation_dim], actions of shape [T, B, action_dim]
        
        embeds = hk.BatchApply(self._encoder)(observations)
        rssm_output = self._rssm(embeds, actions, None)
        
        # state + reward prediction
        post_state_mean = hk.BatchApply(self._decoder)(rssm_output.post_features) # [T, B, H, W, C]
        post_reward_mean = hk.BatchApply(self._reward_predictor)(rssm_output.post_features) # [T, B, 1]
        post_reward_mean = jnp.squeeze(post_reward_mean) # [T, B], so as to be fine with the rewards coming from dataset
        
        prior_state_mean = hk.BatchApply(self._decoder)(rssm_output.prior_features)
        prior_reward_mean = hk.BatchApply(self._reward_predictor)(rssm_output.prior_features)
        prior_reward_mean = jnp.squeeze(prior_reward_mean)
        
        return DreamerOutput(
            post_observation_mean=post_state_mean, post_reward_mean=post_reward_mean, posts=rssm_output.posts,
            prior_observation_mean=prior_state_mean, prior_reward_mean=prior_reward_mean, priors=rssm_output.priors
        )
    
    
    def _byol_forward(self, observations: chex.Array, actions: chex.Array) -> BYOLOutput:
        """Forward pass to compute outputs for BYOL loss."""
        
        # as before, observations should be of shape [T, B, H, W, C], actions of shape [T, B, action_dim]
        
        # first get embeddings
        B = observations.shape[1]
        embeds = hk.BatchApply(self._encoder)(observations)
        init_state = self._rssm._init_state(B)
        
        embed, action = embeds[0], actions[0] # x_0, a_0
        first_state, _ = self._rssm._onestep_posterior(embed, action, init_state) # h_0
        
        first_deter = first_state[-2]
        latent = self._byol_predictor(first_deter)
        latent = jnp.expand_dims(latent, 0)
        
        # === collect h_t, latents (BYOL stuff) ===
        deter_states = self._open_gru_rollout(actions[1:], first_state)
        latents = hk.BatchApply(self._byol_predictor)(deter_states)
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeds
        
        # return
        return BYOLOutput(predicted_latents=pred_latents, target_embeddings=embeds)
    
    
    def _process_to_latent(self, observations: chex.Array, actions: chex.Array) -> chex.Array:
        """Processes all (s, a) to latent features."""
        
        observations = observations.astype(jnp.float32)
        B = observations.shape[1]
        
        # first get embeddings
        embeds = hk.BatchApply(self._encoder)(observations)
        
        # now we scan with posterior model across time axis
        # this is the same as the RSSM posterior feature scan function
        def _posterior_feature_scan_fn(carry: RSSMState, embed_action: PosteriorInput) -> RSSMFnOutput:
            state = carry
            embed, action = embed_action
            
            new_state, _ = self._rssm._onestep_posterior(embed, action, state)
            new_feature = self._rssm._get_feature(new_state)
            return new_state, new_feature
        
        init = self._rssm._init_state(B)
        _, features = hk.scan(_posterior_feature_scan_fn, init, (embeds, actions))
        
        return features
    
    
    def __call__(self, observations: chex.Array, actions: chex.Array) -> WorldModelOutput:
        dreamer_out = self._dreamer_forward(observations, actions)
        byol_out = self._byol_forward(observations, actions)
        
        return WorldModelOutput(dreamer=dreamer_out, byol=byol_out)

# ========================================================================================================================

def make_byol_network(config: ConfigDict, observation_space: gym.Space) -> hk.MultiTransformed:
    """Creates Haiku pure functions for various world model-related operations."""
    
    def wm_fn():
        if is_pixel_based(observation_space):
            observation = observation_space.sample()["pixels"]
            _, _, C, S = observation.shape
            wm = ConvWorldModel(config.pixel, C * S)
        else:
            wm = MLPWorldModel(config.state, observation_space.shape[0])
            
        def init(x: chex.Array, a: chex.Array) -> WorldModelOutput:
            # same as standard forward pass
            return wm(x, a)
        
        def init_state_fn(B: int) -> RSSMState:
            return wm._rssm._init_state(B)
        
        def encode_fn(x: chex.Array) -> chex.Array:
            return wm._encode(x)
        
        def decode_fn(x: chex.Array) -> chex.Array:
            return wm._decode(x)
        
        def dreamer_forward(x: chex.Array, a: chex.Array) -> DreamerOutput:
            return wm._dreamer_forward(x, a)
        
        def byol_forward(x: chex.Array, a: chex.Array) -> BYOLOutput:
            return wm._byol_forward(x, a)
        
        def observe_fn(x: chex.Array, a: chex.Array, s: RSSMState) -> Tuple[RSSMState, ObserveOutput]:
            return wm._onestep_observe(x, a, s)
        
        def imagine_fn(a: chex.Array, s: RSSMState) -> Tuple[RSSMState, ImagineOutput]:
            return wm._onestep_imagine(a, s)
        
        def process_to_latent_fn(x: chex.Array, a: chex.Array) -> chex.Array:
            return wm._process_to_latent(x, a)
            
        return init, (
            dreamer_forward, byol_forward,
            encode_fn, decode_fn, observe_fn, imagine_fn,
            init_state_fn, process_to_latent_fn
        )
    
    wm = hk.multi_transform(wm_fn)
    return wm


class BYOLLearner(Learner):
    """BYOL-Explore world model learner."""
    
    def __init__(
        self,
        config: ConfigDict,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        
        # set up networks
        wm = make_byol_network(config, observation_space)
        
        # optimizer
        optimizer = getattr(optax, config.optimizer_class)(config.learning_rate)
        
        # params
        base_key = jax.random.PRNGKey(seed)
        
        observations = observation_space.sample()
        if isinstance(observations, Mapping):
            observations = observations["pixels"]
            H, W, C, S = observations.shape
            observations = np.reshape(observations, (H, W, C * S))
        actions = action_space.sample()
        
        def make_initial_state(key: chex.PRNGKey) -> BYOLState:
            init_key, state_key = jax.random.split(key)
            
            wm_params = wm.init(
                init_key, seq_batched_like(observations), seq_batched_like(actions)
            )
            if config.initialize_target_with_online_params:
                target_params = wm_params
            else:
                target_init_key, _ = jax.random.split(init_key)
                target_params = wm.init(
                    target_init_key, seq_batched_like(observations),
                    seq_batched_like(actions)
                )
            
            wm_opt_state = optimizer.init(wm_params)
            
            state = BYOLState(
                params=wm_params,
                target_params=target_params,
                opt_state=wm_opt_state,
                rng_key=state_key
            )
            return state

        # similar to make initial state for BYOL: https://github.com/deepmind/deepmind-research/blob/master/byol/byol_experiment.py#L424
        if config.pmap:
            init_wm_fn = jax.pmap(make_initial_state, axis_name="devices")
            base_key = broadcast_to_local_devices(base_key)
        else:
            init_wm_fn = make_initial_state
        
        self._state = init_wm_fn(base_key)
        
        # hparams and functions to note
        (
            dreamer_forward, byol_forward,
            encode, decode, observe, imagine,
            initialize_state, process_to_latent
        ) = wm.apply
        
        ema = config.ema
        vae_beta = config.vae_beta # trades off reconstruction and KL losses in Dreamer
        beta = config.beta # trades off Dreamer ELBO loss and BYOL loss
        
        rssm_config = getattr(config, "pixel" if observations.ndim > 1 else "state").rssm
        discrete = rssm_config.stoch_discrete_dim > 1
        
        # ----- get forward pass methods -----
        
        def _get_latent_dist(
            stats: chex.Array, stop_gradient: bool = False,
        ) -> Union[distrax.Distribution, distrax.DistributionLike]:
            """Gets the distribution associated with the latent statistics.
            
            Sequential information should be here (e.g. we have time dim added.)
            """
            
            # first flatten stats -- this is important!
            stats = jnp.reshape(stats, (-1,) + stats.shape[2:]) # [T * B, stoch_dim, stoch_discrete_dim]
            
            # detach if necessary
            stats = jnp.where(stop_gradient, jax.lax.stop_gradient(stats), stats)
            
            if discrete:
                distribution = OneHotDistribution(logits=stats)
                distribution = distrax.Independent(distribution, 1)
            else:
                mean, std = jnp.split(stats, 2, -1)
                std = jax.nn.softplus(std) + 0.1
                distribution = distrax.MultivariateNormalDiag(mean, std)
            
            return distribution
        
        
        def _get_img_dist(mean: chex.Array) -> distrax.Distribution:
            """Gets the Dreamer observation reconstruction distribution."""
            
            distribution = distrax.Normal(mean, 1.0)
            distribution = distrax.Independent(distribution, 3)
            return distribution
        
        
        def _get_reward_dist(mean: jnp.ndarray) -> distrax.Distribution:
            """Gets the Dreamer reward distribution."""
            
            dist = distrax.Normal(mean, 1.0)
            return dist
        
        # ----- define one-step encode/decode + observe + imagine functions -----
        
        def _encode(state: BYOLState, observations: chex.Array) -> Tuple[BYOLState, chex.Array]:
            """Encodes observations."""
            
            forward_key, new_state_key = jax.random.split(state.rng_key)
            embeds = encode(state.params, forward_key, observations)
            
            new_state = state._replace(rng_key=new_state_key)
            return new_state, embeds
        
        
        def _decode(state: BYOLState, features: chex.Array) -> Tuple[BYOLState, chex.Array]:
            """Decodes features."""
            
            forward_key, new_state_key = jax.random.split(state.rng_key)
            reconstructed_observations = decode(state.params, forward_key, features)
            
            new_state = state._replace(rng_key=new_state_key)
            return new_state, reconstructed_observations

        
        def _observe(
            state: BYOLState, embed: chex.Array, action: chex.Array,
            hidden_state: RSSMState, sample: bool = False,
        ) -> Tuple[BYOLState, Tuple[RSSMState, chex.Array, chex.Array]]:
            """Does one observation step. Returns latent features and rewards."""
            
            forward_key, reward_key, state_key = jax.random.split(state.rng_key, 3)
            new_hidden_state, (latent_feature, reward_mean) = observe(
                state.params, forward_key, embed, action, hidden_state
            )
            
            reward_distribution = _get_reward_dist(reward_mean)
            reward = jnp.where(
                sample,
                reward_distribution.sample(seed=reward_key),
                reward_distribution.mode()
            )
            
            new_state = state._replace(rng_key=state_key)
            return new_state, (new_hidden_state, latent_feature, reward)
        
        
        def _imagine(
            state: BYOLState, action: chex.Array, hidden_state: RSSMState, sample: bool = False,
        ) -> Tuple[BYOLState, Tuple[RSSMState, chex.Array, chex.Array]]:
            """Does one imagine step in the environment. Returns latent features and rewards."""
            
            forward_key, reward_key, state_key = jax.random.split(state.rng_key, 3)
            new_hidden_state, (latent_feature, reward_mean) = imagine(
                state.params, forward_key, action, hidden_state
            )
            
            reward_distribution = _get_reward_dist(reward_mean)
            reward = jnp.where(
                sample,
                reward_distribution.sample(seed=reward_key),
                reward_distribution.mode()
            )
            
            new_state = state._replace(rng_key=state_key)
            return new_state, (new_hidden_state, latent_feature, reward)
        
        # ----- define loss and update functions -----
    
        def byol_loss_fn_window_size(
            wm_params: hk.Params, target_params: hk.Params,
            batch: SequenceBatch, key: chex.PRNGKey, window_size: int
        ) -> Tuple[chex.Array, chex.Array]:
            """Returns the BYOL-Explore loss."""
            
            obs_seq, action_seq = batch.observations, batch.actions
            
            T, B = obs_seq.shape[:2]
            pred_key, target_key = jax.random.split(key)

            # don't need to do for all idxs that work, we just do it for index T - window_size so we don't multiple count the same losses
            starting_idx = T - window_size
            obs_window = sliding_window(obs_seq, starting_idx, window_size) # [T, B, *obs_dims], everything except [T - window_size:] 0s, rolled to front
            action_window = sliding_window(action_seq, starting_idx, window_size) # [T, B, action_dim], everything except [T - window_size:] 0s, rolled to front
            
            pred_latents, _ = byol_forward(wm_params, pred_key, obs_window, action_window)
            pred_latents = jnp.reshape(pred_latents, (-1,) + pred_latents.shape[2:]) # [T * B, embed_dim]

            _, target_latents = byol_forward(target_params, target_key, obs_window, action_window) # (T, B, embed_dim)
            target_latents = jnp.reshape(target_latents, (-1,) + target_latents.shape[2:]) # (T * B, embed_dim)

            # normalize latents
            pred_latents = l2_normalize(pred_latents, axis=-1)
            target_latents = l2_normalize(target_latents, axis=-1)

            # take L2 loss and reshape for correctness in exploration bonus (BYOL loss)
            mses = jnp.square(pred_latents - jax.lax.stop_gradient(target_latents)) # (T * B, embed_dim)
            mses = jnp.reshape(mses, (-1, B) + mses.shape[1:]) # (T, B, embed_dim)
            mses = sliding_window(mses, 0, window_size) # zeros out losses we don't care about (i.e. past window size)
            mses = jnp.roll(mses, shift=starting_idx, axis=0)
            byol_loss_vec = jnp.sum(mses, -1) # (T, B)
            
            byol_loss = jnp.mean(byol_loss_vec)
            return byol_loss, byol_loss_vec
        
        
        def byol_loss_fn(
            wm_params: hk.Params, target_params: hk.Params,
            batch: SequenceBatch, key: chex.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            """Full BYOL loss function over all window sizes."""
            
            T, B = batch.observations.shape[:2]

            def ws_body_fn(ws, curr_state):
                curr_loss, curr_loss_window, key = curr_state
                loss_key, moveon_key = jax.random.split(key)
                
                loss, loss_window = byol_loss_fn_window_size(wm_params, target_params, batch, loss_key, ws)
                return curr_loss + loss, curr_loss_window + loss_window, moveon_key
            
            init_state = (0.0, jnp.zeros((T, B)), key)
            total_loss, total_loss_window, _ = jax.lax.fori_loop(1, T + 1, ws_body_fn, init_state)
            return total_loss / T, total_loss_window / T # take avgs
        
        
        def dreamer_loss_fn(
            wm_params: hk.Params, batch: SequenceBatch, key: chex.PRNGKey
        ) -> LossFnOutput:
            """Only Dreamer loss.
            
            No need for masking -- we can model entire sequence without overlap.
            """
            
            obs_seq, action_seq, reward_seq = (
                batch.observations, batch.actions, batch.rewards
            )
            
            dreamer_out = dreamer_forward(wm_params, key, obs_seq, action_seq)
            post_observation_mean = dreamer_out.post_observation_mean
            post_reward_mean = dreamer_out.post_reward_mean
            post_stats, prior_stats = dreamer_out.posts, dreamer_out.priors
            
            # get reconstruction loss
            observation_distribution = _get_img_dist(post_observation_mean)
            reward_distribution = _get_reward_dist(post_reward_mean)
            
            observation_log_likelihood = observation_distribution.log_prob(obs_seq) * batch.masks
            reward_log_likelihood = reward_distribution.log_prob(reward_seq) * batch.masks
            
            reconstruction_loss = -observation_log_likelihood.mean() - reward_log_likelihood.mean()
            
            # get KL loss
            post_dist = _get_latent_dist(post_stats)
            prior_dist = _get_latent_dist(prior_stats)
            
            if config.kl_balance == 0.5:
                kl = post_dist.kl_divergence(prior_dist)
                kl_loss = jnp.maximum(kl, config.kl_free_value).mean()
                
                # make KL a scalar for logging
                kl = kl.mean()
            else:
                post_dist_sg = _get_latent_dist(post_stats, stop_gradient=True)
                prior_dist_sg = _get_latent_dist(prior_stats, stop_gradient=True)
                
                # these should have the same value, but different gradient propagation
                left_kl = post_dist.kl_divergence(prior_dist_sg) # [T * B]
                right_kl = post_dist_sg.kl_divergence(prior_dist) # [T * B]
                
                left_loss = jnp.maximum(left_kl, config.kl_free_value).mean()
                right_loss = jnp.maximum(right_kl, config.kl_free_value).mean()
                
                kl_loss = (1.0 - config.kl_balance) * left_loss + config.kl_balance * right_loss
                
                kl = ((left_kl + right_kl) / 2).mean()
            
            total_loss = reconstruction_loss + vae_beta * kl_loss
            metrics = {
                "total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "kl": kl
            }
            return total_loss, metrics
        
        
        def total_loss_fn(
            wm_params: hk.Params, target_params: hk.Params,
            batch: SequenceBatch, key: chex.PRNGKey
        ) -> LossFnOutput:
            """Combination of BYOL and Dreamer losses."""
            
            byol_key, dreamer_key = jax.random.split(key)
            byol_loss, _ = byol_loss_fn(wm_params, target_params, batch, byol_key)
            dreamer_loss, metrics = dreamer_loss_fn(wm_params, batch, dreamer_key)
            
            # add BYOL loss to the 
            metrics["byol_loss"] = byol_loss
            
            total_loss = dreamer_loss + beta * byol_loss
            metrics["total_loss"] = total_loss
            return total_loss, metrics
        
        
        def update(
            state: BYOLState, batch: SequenceBatch, step: int,
        ) -> Tuple[BYOLState, MetricsDict]:
            """Updates the model."""
            
            del step
            
            # first we preprocess the batch
            batch = _preprocess(batch)
            
            update_key, state_key = jax.random.split(state.rng_key)
            grad_fn = jax.grad(total_loss_fn, has_aux=True)
            grads, metrics = grad_fn(state.params, state.target_params, batch, update_key)
            
            if config.pmap:
                # all reduce to one device
                grads = jax.lax.pmean(grads, axis_name="devices")
                metrics = jax.lax.pmean(metrics, axis_name="devices")
            
            update, new_opt_state = optimizer.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, update)
            
            new_target_params = optax.incremental_update(new_params, state.target_params, ema)
            new_state = BYOLState(
                params=new_params,
                target_params=new_target_params,
                opt_state=new_opt_state,
                rng_key=state_key
            )
            
            return new_state, metrics
        
        
        def compute_uncertainty(
            state: BYOLState, batch: SequenceBatch,
        ) -> Tuple[BYOLState, chex.Array]:
            """
            Computes transition uncertainties according to part (iv) in BYOL-Explore paper.
            
            :param obs_seq: Sequence of observations, of shape [T, B, obs_dim]
            :param action_seq: Sequence of actions, of shape [T, B, action_dim]
            
            :return uncertainties: Model uncertainties, of shape [T, B].
            """
            
            # losses are of shape [T, B], result of only BYOL loss accumulation
            loss_key, new_state_key = jax.random.split(state.rng_key)
            _, losses = byol_loss_fn(state.params, state.target_params, batch, loss_key)
            
            new_state = state._replace(rng_key=new_state_key)
            return new_state, jax.lax.stop_gradient(losses)
        
        
        def _initialize_state(state: BYOLState, batch_size: int) -> Tuple[BYOLState, RSSMState]:
            """Initializes the model state."""
            
            forward_key, new_state_key = jax.random.split(state.rng_key)
            init_state = initialize_state(state.params, forward_key, batch_size)
            
            new_state = state._replace(rng_key=new_state_key)
            return new_state, init_state
        
        
        def _process_to_latent(
            state: BYOLState, obs_seq: chex.Array, action_seq: chex.Array
        ) -> Tuple[BYOLState, chex.Array]:
            """Processes the observations/actions to latent features."""

            forward_key, new_state_key = jax.random.split(state.rng_key)
            
            # forward pass
            features = process_to_latent(state.params, forward_key, obs_seq, action_seq)
            
            # change key
            new_state = state._replace(rng_key=new_state_key)
            
            return new_state, features
        
        # ====== eval methods ======
        
        def eval(state: BYOLState, batch: SequenceBatch) -> Tuple[BYOLState, chex.Array, chex.Array]:
            """Evaluate from prior and posterior on a batch in the offline dataset."""
            
            # first we preprocess the batch
            batch = _preprocess(batch)
            
            # only get the first batch, unflatten so we can actually pass it through
            obs_seq, action_seq = batch.observations[:, 0, ...], batch.actions[:, 0, ...]
            obs_seq, action_seq = jnp.expand_dims(obs_seq, 1), jnp.expand_dims(action_seq, 1)
            
            eval_key, new_state_key = jax.random.split(state.rng_key)
            dreamer_out = dreamer_forward(state.params, eval_key, obs_seq, action_seq)
            
            post_img_means = dreamer_out.post_observation_mean
            prior_img_means = dreamer_out.prior_observation_mean
            
            # now that we have these, we should squeeze them
            post_img_means = post_img_means.squeeze()
            prior_img_means = prior_img_means.squeeze()
            
            def get_image_to_log(img_means: chex.Array) -> chex.Array:
                img_means = (img_means + 0.5) * 255.0
                img_means = img_means.astype(jnp.uint8)
                img_means = img_means[:, :, :, :3] # just the first image
                return img_means
            
            post_img_means = get_image_to_log(post_img_means)
            prior_img_means = get_image_to_log(prior_img_means)
            
            new_state = state._replace(rng_key=new_state_key)
            return new_state, jax.lax.stop_gradient(post_img_means), jax.lax.stop_gradient(prior_img_means)
        
        
        # whether to parallelize across devices, make sure to have multiple devices here for this for better performance
        # only pmapping the update here because we don't need to do it for eval and uncertainty computation (as they are both evaluation protocols mainly)
        if config.pmap:
            self._update = jax.pmap(update, axis_name="devices") # this is the only function that has any parallel components to it
        else:
            self._update = jax.jit(update)
        
        
        self._compute_uncertainty = jax.jit(compute_uncertainty)
        
        self._encode = jax.jit(_encode)
        self._decode = jax.jit(_decode)
        self._process_to_latent = jax.jit(_process_to_latent)
        self._eval = jax.jit(eval)
        
        self._initialize_state = _initialize_state # no need to JIT this
        self._observe = jax.jit(_observe)
        self._imagine = jax.jit(_imagine)