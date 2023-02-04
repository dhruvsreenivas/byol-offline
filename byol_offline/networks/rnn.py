import jax
import jax.numpy as jnp
import haiku as hk
import distrax
from typing import Optional

glorot_w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
ortho_init = hk.initializers.Orthogonal()

class LayerNormGRU(hk.RNNCore):
    '''LayerNorm GRU used in DreamerV2.'''
    def __init__(self, hidden_size, norm=False, act='tanh'):
        super().__init__()
        self._hidden_size = hidden_size
        self._act = jax.lax.tanh if act == 'tanh' else jax.nn.elu
        
        self._layer = hk.Linear(3 * hidden_size, with_bias=norm is not None, w_init=glorot_w_init)
        self._use_norm = norm
        if norm:
            self._norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            
    def initial_state(self, batch_size: Optional[int]):
        state = jnp.zeros(self._hidden_size)
        if batch_size is not None:
            # add batch dim at position 0
            state = jnp.expand_dims(state, 0)
            state = jnp.tile(state, reps=(batch_size, 1))
        return state
    
    def __call__(self, x, state):
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
    '''
    Recurrent state space model used in DreamerV1/V2 (discrete or continuous). Similar to BYOL-Explore latent world model.
    '''
    def __init__(self, cfg):
        super().__init__()
        
        self._embed_dim = 1024 if cfg.dreamer else 20000
        self._deter_dim = cfg.gru_hidden_size
        self._stoch_dim = cfg.stoch_dim
        
        assert cfg.stoch_discrete_dim >= 1, "stoch discrete dim nonpositive"
        self._stoch_discrete_dim = cfg.stoch_discrete_dim
        self._discrete = self._stoch_discrete_dim > 1
        
        self._pre_gru = hk.Sequential([
            hk.Linear(cfg.gru_hidden_size, w_init=glorot_w_init), # no need for layernorm in dreamer
            jax.nn.elu
        ])
        
        if cfg.use_ln:
            self._gru = LayerNormGRU(cfg.gru_hidden_size, norm=True)
        else:
            self._gru = hk.GRU(cfg.gru_hidden_size, w_i_init=glorot_w_init, w_h_init=ortho_init)
        
        # DreamerV2 online uses only 1 prior, and since we're not ensembling here we don't mind doing the same thing
        dist_dim = cfg.stoch_dim * cfg.stoch_discrete_dim if self._discrete else 2 * cfg.stoch_dim
        self._prior_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=glorot_w_init),
            jax.nn.elu,
            hk.Linear(dist_dim, w_init=glorot_w_init)
        ])
        self._post_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=glorot_w_init),
            jax.nn.elu,
            hk.Linear(dist_dim, w_init=glorot_w_init)
        ])
        
    def _init_feature(self, batch_size: Optional[int]):
        stoch_shape = self._stoch_dim * self._stoch_discrete_dim # 1 if not discrete!
        if batch_size is not None:
            return jnp.zeros((batch_size, stoch_shape + self._deter_dim))
        else:
            return jnp.zeros((stoch_shape + self._deter_dim))

    def _get_dist(self, stats: jnp.ndarray):
        if self._discrete:
            logits = jnp.reshape(stats, stats.shape[:-1] + (self._stoch_dim, self._stoch_discrete_dim))
            dist_class = distrax.straight_through_wrapper(distrax.OneHotCategorical)
            dist = dist_class(logits=logits)
            dist = distrax.Independent(dist, 1)
        else:
            mean, std = jnp.split(stats, 2, -1)
            std = jax.nn.softplus(std) + 0.1 # TODO maybe not hardcode 0.1 or softplus activation
            dist = distrax.MultivariateNormalDiag(mean, std)
        
        return dist
        
    def _onestep_prior(self, action, state):
        deter, stoch = jnp.split(state, [self._deter_dim], -1)

        sta = jnp.concatenate([stoch, action], -1)
        x = self._pre_gru(sta)

        new_deter, _ = self._gru(x, deter)
        prior_stats = self._prior_mlp(new_deter)
        dist = self._get_dist(prior_stats)
        new_stoch = dist.sample(seed=hk.next_rng_key()).reshape(action.shape[0], -1)
        
        new_state = jnp.concatenate([new_deter, new_stoch], -1)
        return new_state, prior_stats
    
    def _onestep_post(self, embed, action, state):
        new_state, _ = self._onestep_prior(action, state)
        
        deter = new_state[..., :self._deter_dim]
        de = jnp.concatenate([deter, embed], -1)
        post_stats = self._post_mlp(de)
        post_dist = self._get_dist(post_stats)
        new_stoch = post_dist.sample(seed=hk.next_rng_key()).reshape(action.shape[0], -1)
        
        new_state = jnp.concatenate([deter, new_stoch], -1)
        return new_state, post_stats
    
    def __call__(self, embeds, actions, state=None):
        B = embeds.shape[1]
        state = self._init_feature(B) if state is None else state
        
        # scan fns
        def _prior_scan_fn(carry, act):
            state = carry
            new_state, prior = self._onestep_prior(act, state)
            return new_state, prior
        
        def _post_scan_fn(carry, inp):
            state = carry
            emb, act = jnp.split(inp, [self._embed_dim], -1)
            new_state, post = self._onestep_post(emb, act, state)
            return new_state, post
        
        def _feature_scan_fn(carry, inp):
            state = carry
            emb, act = jnp.split(inp, [self._embed_dim], -1)
            new_state, _ = self._onestep_post(emb, act, state)
            return new_state, new_state
        
        init = state
        ea = jnp.concatenate([embeds, actions], -1) # (T, B, embed_dim + action_dim)
        _, priors = hk.scan(_prior_scan_fn, init, actions)
        _, posts = hk.scan(_post_scan_fn, init, ea)
        _, features = hk.scan(_feature_scan_fn, init, ea)
        
        return priors, posts, features