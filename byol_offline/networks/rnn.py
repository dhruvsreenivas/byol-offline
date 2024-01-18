import chex
import haiku as hk
import jax
import jax.numpy as jnp
import distrax
from ml_collections import ConfigDict
from typing import NamedTuple, Optional, Tuple, Union

from byol_offline.types import RecurrentOutput

RSSMState = Union[
    Tuple[chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
]

RSSMStatistics = Union[chex.Array, Tuple[chex.Array, chex.Array]]
RSSMFnOutput = Tuple[RSSMState, chex.Array]
PosteriorInput = Tuple[chex.Array, chex.Array]


class RSSMOutput(NamedTuple):
    priors: chex.Array
    posts: chex.Array
    
    prior_features: chex.Array
    post_features: chex.Array


glorot_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
ortho_init = hk.initializers.Orthogonal()


class LayerNormGRU(hk.RNNCore):
    """GRU with additional layer norm, as in DreamerV2."""
    
    def __init__(self, hidden_size: int, norm: bool = False, act: str = "tanh"):
        super().__init__()
        self._hidden_size = hidden_size
        self._act = getattr(jax.nn, act) if hasattr(jax.nn, act) else getattr(jnp, act)
        
        self._layer = hk.Linear(3 * hidden_size, with_bias=norm is not None, w_init=glorot_w_init)
        self._use_norm = norm
        if norm:
            self._norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            
    def initial_state(self, batch_size: Optional[int]) -> chex.Array:
        state = jnp.zeros(self._hidden_size)
        if batch_size is not None:
            # add batch dim at position 0
            state = jnp.expand_dims(state, 0)
            state = jnp.tile(state, reps=(batch_size, 1))
        return state
    
    def __call__(self, x: chex.Array, state: chex.Array) -> RecurrentOutput:
        parts = self._layer(jnp.concatenate([x, state], axis=-1))
        if self._use_norm:
            parts = self._norm(parts)
        
        reset, cand, update = jnp.split(parts, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        out = update * cand + (1 - update) * state
        return out, out

    
class RSSM(hk.Module):
    """
    Recurrent state space model used in DreamerV1/V2 (discrete or continuous).
    Similar to BYOL-Explore latent world model.
    """
    
    
    def __init__(self, config: ConfigDict):
        super().__init__()
        
        self._embed_dim = 1024 if config.dreamer else 20000
        self._deter_dim = config.gru_hidden_size
        self._stoch_dim = config.stoch_dim
        
        assert config.stoch_discrete_dim >= 1, "stoch discrete dim nonpositive"
        self._stoch_discrete_dim = config.stoch_discrete_dim
        self._discrete = self._stoch_discrete_dim > 1
        
        self._pre_gru = hk.Sequential([
            hk.Linear(config.gru_hidden_size, w_init=glorot_w_init),
            jax.nn.elu
        ])
        
        if config.use_layer_norm:
            self._gru = LayerNormGRU(config.gru_hidden_size, norm=True)
        else:
            self._gru = hk.GRU(config.gru_hidden_size, w_i_init=glorot_w_init, w_h_init=ortho_init)
        
        # DreamerV2 online uses only 1 prior, and since we're not ensembling here we don't mind doing the same thing
        dist_dim = config.stoch_dim * config.stoch_discrete_dim if self._discrete else 2 * config.stoch_dim
        self._prior_mlp = hk.Sequential([
            hk.Linear(config.hidden_dim, w_init=glorot_w_init),
            jax.nn.elu,
            hk.Linear(dist_dim, w_init=glorot_w_init)
        ])
        self._post_mlp = hk.Sequential([
            hk.Linear(config.hidden_dim, w_init=glorot_w_init),
            jax.nn.elu,
            hk.Linear(dist_dim, w_init=glorot_w_init)
        ])
        
    
    def _init_state(self, batch_size: Optional[int]) -> RSSMState:
        """Returns the initial state of the RSSM as a tuple. Dicts aren't useful for this.
        
        In the caes of categorical distributions, we return [logits, deter, stoch].
        In the case of Gaussian distribution, we return [mean, std, deter, stoch].
        """
        
        bs = batch_size if batch_size is not None else 1
        
        if self._discrete:
            return (
                jnp.zeros((bs, self._stoch_dim, self._stoch_discrete_dim)),
                jnp.zeros((bs, self._deter_dim)),
                jnp.zeros((bs, self._stoch_dim, self._stoch_discrete_dim))
            )
        else:
            return (
                jnp.zeros((bs, self._stoch_dim)),
                jnp.ones((bs, self._stoch_dim)),
                jnp.zeros((bs, self._deter_dim)),
                jnp.zeros((bs, self._stoch_dim))
            )
    
    
    def _get_feature(self, state: RSSMState) -> chex.Array:
        """Gets the hidden latent state (combination of deterministic + stochastic parts)."""
        
        deter = state[-2]
        stoch = state[-1]
        
        if self._discrete:
            # need to flatten it in discrete case
            stoch = jnp.reshape(stoch, stoch.shape[:-2] + (-1,))
        
        return jnp.concatenate([deter, stoch], axis=-1)

    
    def _get_dist_and_stats(self, stats: chex.Array) -> Tuple[distrax.Distribution, RSSMStatistics]:
        """Gets the latent distribution and statistics (e.g. logits that define categorical, mean/std of Gaussian)."""
        
        if self._discrete:
            logits = jnp.reshape(stats, stats.shape[:-1] + (self._stoch_dim, self._stoch_discrete_dim))
            dist_class = distrax.straight_through_wrapper(distrax.OneHotCategorical)
            distribution = dist_class(logits=logits)
            distribution = distrax.Independent(distribution, 1)
            
            stats = (logits,)
        else:
            mean, std = jnp.split(stats, 2, -1)
            std = jax.nn.softplus(std) + 0.1
            distribution = distrax.MultivariateNormalDiag(mean, std)
            
            stats = (mean, std)
        
        return distribution, stats
    
    
    def _onestep_prior(self, action: chex.Array, state: RSSMState) -> RSSMFnOutput:
        """Does one step of the prior distribution."""
        
        deter, stoch = state[-2:]
        
        sta = jnp.concatenate([stoch, action], axis=-1)
        sta = self._pre_gru(sta)
        
        x, new_deter = self._gru(x, deter)
        prior_stats = self._prior_mlp(x)

        # now that we have the prior stats, we can grab the distribution and new state
        prior_distribution, prior_stats = self._get_dist_and_stats(prior_stats)
        new_stoch = prior_distribution.sample(seed=hk.next_rng_key())
        
        new_state = prior_stats + (new_deter, new_stoch)
        return new_state, prior_stats
    
    
    def _onestep_post(self, embed: chex.Array, action: chex.Array, state: RSSMState) -> RSSMFnOutput:
        """Gets the posterior outputs of the RSSM."""
        
        prior_state, _ = self._onestep_prior(action, state)
        
        deter = prior_state[-2]
        de = jnp.concatenate([deter, embed], axis=-1)
        posterior_stats = self._post_mlp(de)
        
        posterior_distribution, posterior_stats = self._get_dist_and_stats(posterior_stats)
        new_stoch = posterior_distribution.sample(seed=hk.next_rng_key())
        
        new_state = posterior_stats + (deter, new_stoch)
        return new_state, posterior_stats
    
    
    def __call__(
        self, embeds: chex.Array, actions: chex.Array, state: Optional[chex.Array] = None
    ) -> RSSMOutput:
        """Calls the RSSM."""
        
        B = embeds.shape[1]
        state = self._init_state(B) if state is None else state
        
        # --- scan fns ---
        
        # prior
        def _prior_scan_fn(carry: RSSMState, action: chex.Array) -> RSSMFnOutput:
            state = carry
            new_state, prior = self._onestep_prior(action, state)
            return new_state, prior
        
        
        def _prior_feature_scan_fn(carry: RSSMState, action: chex.Array) -> RSSMFnOutput:
            state = carry
            new_state, _ = self._onestep_prior(action, state)
            feature = self._get_feature(new_state)
            return new_state, feature
        
        
        # post
        def _post_scan_fn(carry: RSSMState, embed_action: PosteriorInput) -> RSSMFnOutput:
            state = carry
            embed, action = embed_action
            
            new_state, post = self._onestep_post(embed, action, state)
            return new_state, post
        
        
        def _post_feature_scan_fn(carry: RSSMState, embed_action: PosteriorInput) -> RSSMFnOutput:
            state = carry
            embed, action = embed_action
            
            new_state, _ = self._onestep_post(embed, action, state)
            feature = self._get_feature(new_state)
            return new_state, feature
        
        # --- run the scan ---
        
        init = state
        _, priors = hk.scan(_prior_scan_fn, init, actions)
        _, posts = hk.scan(_post_scan_fn, init, (embeds, actions))
        
        _, prior_features = hk.scan(_prior_feature_scan_fn, init, actions)
        _, post_features = hk.scan(_post_feature_scan_fn, init, (embeds, actions))
        
        return RSSMOutput(
            priors=priors, posts=posts,
            prior_features=prior_features, post_features=post_features
        )