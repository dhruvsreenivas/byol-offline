import chex
import haiku as hk
import jax
import jax.numpy as jnp
import distrax
from typing import NamedTuple, Optional
from ml_collections import ConfigDict

from byol_offline.types import RecurrentOutput

"""Various recurrent modules."""

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
            hk.Linear(config.gru_hidden_size, w_init=glorot_w_init), # no need for layernorm in dreamer
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
        
    def _init_feature(self, batch_size: Optional[int]) -> chex.Array:
        """Sets up the initial state features for the RSSM."""
        
        stoch_shape = self._stoch_dim * self._stoch_discrete_dim # 1 if not discrete
        if batch_size is not None:
            return jnp.zeros((batch_size, stoch_shape + self._deter_dim))
        else:
            return jnp.zeros((stoch_shape + self._deter_dim))

    def _get_dist(self, stats: chex.Array) -> distrax.Distribution:
        """Gets the distribution associated with the statistics provided."""
        
        if self._discrete:
            logits = jnp.reshape(stats, stats.shape[:-1] + (self._stoch_dim, self._stoch_discrete_dim))
            dist_class = distrax.straight_through_wrapper(distrax.OneHotCategorical)
            dist = dist_class(logits=logits)
            dist = distrax.Independent(dist, 1)
        else:
            mean, std = jnp.split(stats, 2, -1)
            std = jax.nn.softplus(std) + 0.1
            dist = distrax.MultivariateNormalDiag(mean, std)
        
        return dist
        
    def _onestep_prior(self, action: chex.Array, state: chex.Array) -> RecurrentOutput:
        """Performs a sampling step from the prior distribution of the RSSM."""
        
        deter, stoch = jnp.split(state, [self._deter_dim], -1)

        sta = jnp.concatenate([stoch, action], -1)
        x = self._pre_gru(sta)

        new_deter, _ = self._gru(x, deter)
        prior_stats = self._prior_mlp(new_deter)
        dist = self._get_dist(prior_stats)
        new_stoch = dist.sample(seed=hk.next_rng_key()).reshape(action.shape[0], -1)
        
        new_state = jnp.concatenate([new_deter, new_stoch], -1)
        return new_state, prior_stats
    
    def _onestep_post(self, embed: chex.Array, action: chex.Array, state: chex.Array) -> RecurrentOutput:
        new_state, _ = self._onestep_prior(action, state)
        
        deter = new_state[..., :self._deter_dim]
        de = jnp.concatenate([deter, embed], -1)
        post_stats = self._post_mlp(de)
        post_dist = self._get_dist(post_stats)
        new_stoch = post_dist.sample(seed=hk.next_rng_key()).reshape(action.shape[0], -1)
        
        new_state = jnp.concatenate([deter, new_stoch], -1)
        return new_state, post_stats
    
    def __call__(
        self, embeds: chex.Array, actions: chex.Array, state: Optional[chex.Array] = None
    ) -> RSSMOutput:
        """Calls the RSSM."""
        
        B = embeds.shape[1]
        state = self._init_feature(B) if state is None else state
        
        # === scan fns ===
        
        # prior
        def _prior_scan_fn(carry: chex.Array, act: chex.Array) -> RecurrentOutput:
            state = carry
            new_state, prior = self._onestep_prior(act, state)
            return new_state, prior
        
        def _prior_feature_scan_fn(carry: chex.Array, act: chex.Array) -> RecurrentOutput:
            state = carry
            new_state, _ = self._onestep_prior(act, state)
            return new_state, new_state
        
        # post
        def _post_scan_fn(carry: chex.Array, inp: chex.Array) -> RecurrentOutput:
            state = carry
            emb, act = jnp.split(inp, [self._embed_dim], -1)
            new_state, post = self._onestep_post(emb, act, state)
            return new_state, post
        
        def _post_feature_scan_fn(carry: chex.Array, inp: chex.Array) -> RecurrentOutput:
            state = carry
            emb, act = jnp.split(inp, [self._embed_dim], -1)
            new_state, _ = self._onestep_post(emb, act, state)
            return new_state, new_state
        
        init = state
        ea = jnp.concatenate([embeds, actions], -1) # [T, B, embed_dim + action_dim]
        _, priors = hk.scan(_prior_scan_fn, init, actions)
        _, posts = hk.scan(_post_scan_fn, init, ea)
        
        _, prior_features = hk.scan(_prior_feature_scan_fn, init, actions)
        _, post_features = hk.scan(_post_feature_scan_fn, init, ea)
        
        return RSSMOutput(
            priors=priors, posts=posts,
            prior_features=prior_features, post_features=post_features
        )