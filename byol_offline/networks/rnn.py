import jax
import jax.numpy as jnp
import haiku as hk
import distrax
from typing import Optional, Dict

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
        
    def _init_state(self, batch_size: Optional[int]) -> Dict[str, jnp.ndarray]:
        # going to be (dist_params (logit/mean, std), deter, stoch) -> dicts are not useful in jax when it comes to jax.lax.scan and indexing
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
                
    def _get_feat(self, state: Dict[str, jnp.ndarray]):
        deter = state[-2]
        stoch = state[-1]
        stoch = jnp.reshape(stoch, stoch.shape[:-2] + (-1,)) # (stoch_dim, stoch_discrete_dim) -> (stoch_dim * stoch_discrete_dim)
        return jnp.concatenate([deter, stoch], axis=-1)

    def _get_dist_and_stats_tup(self, stats: jnp.ndarray):
        if self._discrete:
            logits = jnp.reshape(stats, stats.shape[:-1] + (self._stoch_dim, self._stoch_discrete_dim))
            dist_class = distrax.straight_through_wrapper(distrax.OneHotCategorical)
            dist = dist_class(logits=logits)
            dist = distrax.Independent(dist, 1)
            
            stats_tup = (logits,)
        else:
            mean, std = jnp.split(stats, 2, -1)
            std = jax.nn.softplus(std) + 0.1 # TODO maybe not hardcode 0.1 or softplus activation
            dist = distrax.MultivariateNormalDiag(mean, std)
            
            stats_tup = (mean, std)
        
        return dist, stats_tup
        
    def _onestep_prior(self, action, state):
        deter, stoch = state[-2], state[-1]
        stoch = jnp.reshape(stoch, stoch.shape[:-2] + (-1,))
        
        sta = jnp.concatenate([stoch, action], -1)
        x = self._pre_gru(sta)

        new_deter, _ = self._gru(x, deter)
        prior_stats = self._prior_mlp(new_deter)
        dist, stats_tup = self._get_dist_and_stats_tup(prior_stats)
        new_stoch = dist.sample(seed=hk.next_rng_key())
        
        new_state = stats_tup + (new_deter, new_stoch)
        return new_state, prior_stats
    
    def _onestep_post(self, embed, action, state):
        new_state, _ = self._onestep_prior(action, state)
        
        deter = new_state[-2]
        de = jnp.concatenate([deter, embed], -1)
        post_stats = self._post_mlp(de)
        post_dist, stats_tup = self._get_dist_and_stats_tup(post_stats)
        new_stoch = post_dist.sample(seed=hk.next_rng_key())
        
        new_state = stats_tup + (deter, new_stoch)
        return new_state, post_stats
    
    def __call__(self, embeds, actions, state=None):
        B = embeds.shape[1]
        state = self._init_state(B) if state is None else state
        
        # scan fns
        def _prior_scan_fn(carry, act):
            state = carry
            new_state, prior = self._onestep_prior(act, state)
            return new_state, prior
        
        def _post_scan_fn(carry, inp):
            state = carry
            emb, act = inp
            new_state, post = self._onestep_post(emb, act, state)
            return new_state, post
        
        def _post_feature_scan_fn(carry, inp):
            state = carry
            emb, act = inp
            new_state, _ = self._onestep_post(emb, act, state)
            feat = self._get_feat(new_state)
            return new_state, feat
        
        def _prior_feature_scan_fn(carry, act):
            state = carry
            new_state, _ = self._onestep_prior(act, state)
            feat = self._get_feat(new_state)
            return new_state, feat
        
        init = state
        _, priors = hk.scan(_prior_scan_fn, init, actions)
        _, posts = hk.scan(_post_scan_fn, init, (embeds, actions))
        _, post_features = hk.scan(_post_feature_scan_fn, init, (embeds, actions))
        _, prior_features = hk.scan(_prior_feature_scan_fn, init, actions)
        
        return priors, posts, post_features, prior_features