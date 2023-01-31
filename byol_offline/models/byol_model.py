import jax
import jax.numpy as jnp
import haiku as hk
import optax
import distrax
import dill
from typing import Tuple, NamedTuple, Dict, Union

from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.decoder import DrQv2Decoder, DreamerDecoder
from byol_offline.networks.rnn import *
from byol_offline.networks.predictors import BYOLPredictor
from byol_offline.models.byol_utils import *
from utils import MUJOCO_ENVS, seq_batched_zeros_like

class BYOLTrainState(NamedTuple):
    wm_params: hk.Params
    target_params: hk.Params
    wm_opt_state: optax.OptState

class ConvLatentWorldModel(hk.Module):
    '''Latent world model for DMC tasks. Primarily inspired by DreamerV2 and DrQv2 repositories.'''
    def __init__(self, cfg):
        super().__init__()
        
        # nets
        if cfg.dreamer:
            self._encoder = DreamerEncoder(cfg.depth)
        else:
            self._encoder = DrQv2Encoder()
        
        # rssm
        self._rssm = RSSM(cfg) # uses both closed gru and open gru here, closed for dreamer loss, open for byol loss
        
        # prediction heads
        if cfg.dreamer:
            self._decoder = DreamerDecoder(cfg.obs_shape[-1], cfg.depth)
            self._byol_predictor = BYOLPredictor(1024)
        else:
            self._decoder = DrQv2Decoder(cfg.obs_shape[-1])
            self._byol_predictor = BYOLPredictor(20000)
        
        self._reward_predictor = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
            w_init=glorot_w_init
        )
        
    def _open_gru_rollout(self, actions, state):
        '''
        Rolls out the open GRU (in this case, the RSSM recurrent unit) over all actions, starting at init state 'state'.
        
        Only gets the deterministic part of the output, similar to BYOL-Explore.
        '''
        def _scan_fn(carry, act):
            state = carry
            new_state, _ = self._rssm._onestep_prior(act, state)
            return new_state, new_state[..., :self._rssm._deter_dim] # new feature, deter state
        
        _, deter_states = hk.scan(_scan_fn, state, actions)
        return deter_states
        
    def _dreamer_extras(self, embeds, actions):
        # always start with init state at zeros
        posts, priors, features = self._rssm(embeds, actions, None)
        # posts: (T, B, dist_dim), priors: (T, B, dist_dim), features: (T, B, feature_dim)
        img_mean = hk.BatchApply(self._decoder)(features) # (T, B, H, W, C)
        reward_mean = hk.BatchApply(self._reward_predictor)(features)
        reward_mean = jnp.squeeze(reward_mean)
        
        return img_mean, reward_mean, posts, priors
        
    def __call__(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        
        # first get embeddings
        B = obs.shape[1]
        embeds = hk.BatchApply(self._encoder)(obs)
        init_feature = self._rssm._init_feature(B)
        
        embed, action = embeds[0], actions[0] # x_0, a_0
        first_feature, _ = self._rssm._onestep_post(embed, action, init_feature) # h_0
        latent = self._byol_predictor(first_feature[..., :self._rssm._deter_dim])
        latent = jnp.expand_dims(latent, 0)
        
        # === collect h_t, latents (byol stuff) ===
        deter_states = self._open_gru_rollout(actions[1:], first_feature)
        latents = self._byol_predictor(deter_states)
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeds
        
        # === dreamer stuff ===
        dreamer_extras = self._dreamer_extras(embeds, actions)
        
        # return
        return pred_latents, embeds, dreamer_extras

class MLPLatentWorldModel(hk.Module):
    '''Latent world model for D4RL tasks. Primarily inspired by MOPO repository.'''
    def __init__(self, cfg):
        super().__init__()
        
        # nets
        self._encoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.repr_dim],
            activation=jax.nn.swish
        )
        
        # rssm
        self._rssm = RSSM(cfg) # uses both closed gru and open gru here, closed for dreamer loss, open for byol loss
        
        # predictors
        self._decoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.obs_shape[0]],
            activation=jax.nn.swish
        )
        self._reward_predictor = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
            w_init=glorot_w_init
        )
        self._byol_predictor = BYOLPredictor(cfg.repr_dim)
        
    def _dreamer_extras(self, embeds, actions):
        posts, priors, features = self._rssm(embeds, actions, None)
        
        # state + reward prediction
        state_mean = hk.BatchApply(self._decoder)(features) # (T, B, H, W, C)
        reward_mean = hk.BatchApply(self._reward_predictor)(features)
        reward_mean = jnp.squeeze(reward_mean)
        
        return state_mean, reward_mean, posts, priors
    
    def _open_gru_rollout(self, actions, state):
        '''
        Rolls out the open GRU (in this case, the RSSM recurrent unit) over all actions, starting at init state 'state'.
        
        Only gets the deterministic part of the output, similar to BYOL-Explore.
        '''
        def _scan_fn(carry, act):
            state = carry
            new_state, _ = self._rssm._onestep_prior(act, state)
            return new_state, new_state[..., :self._rssm._deter_dim]
        
        _, deter_states = hk.scan(_scan_fn, state, actions)
        return deter_states
    
    def __call__(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        
        # first get embeddings
        B = obs.shape[1]
        embeds = hk.BatchApply(self._encoder)(obs)
        init_feature = self._rssm._init_feature(B)
        
        embed, action = embeds[0], actions[0] # x_0, a_0
        first_feature, _ = self._rssm._onestep_post(embed, action, init_feature) # h_0
        latent = self._byol_predictor(first_feature[..., :self._rssm._deter_dim])
        latent = jnp.expand_dims(latent, 0)
        
        # === collect h_t, latents (byol stuff) ===
        deter_states = self._open_gru_rollout(actions[1:], first_feature)
        latents = hk.BatchApply(self._byol_predictor)(deter_states)
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeds
        
        # === dreamer stuff ===
        dreamer_extras = self._dreamer_extras(embeds, actions)
        
        # return
        return pred_latents, embeds, dreamer_extras
    
class WorldModelTrainer:
    '''World model trainer.'''
    def __init__(self, cfg):
        # set up
        if cfg.task in MUJOCO_ENVS:
            wm_fn = lambda o, a: MLPLatentWorldModel(cfg.d4rl)(o, a)
        else:
            wm_fn = lambda o, a: ConvLatentWorldModel(cfg.vd4rl)(o, a)
        
        # should be ok with byol loss here, as byol loss is regardless deterministic and only depends on deter states
        wm = hk.transform(wm_fn) # rngs are used in the dreamer section of the model for generation, so cannot use without_apply_rng
        
        # params
        key = jax.random.PRNGKey(cfg.seed)
        k1, k2 = jax.random.split(key)
        
        wm_params = wm.init(k1, seq_batched_zeros_like(cfg.obs_shape), seq_batched_zeros_like(cfg.action_shape))
        target_params = wm.init(k2, seq_batched_zeros_like(cfg.obs_shape), seq_batched_zeros_like(cfg.action_shape))
        
        # copy params across devices if required
        if cfg.pmap:
            device_lst = jax.devices()
            wm_params = jax.device_put_replicated(wm_params, device_lst)
            target_params = jax.device_put_replicated(target_params, device_lst)
        
        # optimizer
        if cfg.optim == 'adam':
            wm_opt = optax.adam(cfg.lr)
        elif cfg.optim == 'adamw':
            wm_opt = optax.adamw(cfg.lr)
        else:
            wm_opt = optax.sgd(cfg.lr, momentum=0.9)
        
        wm_opt_init_fn = wm_opt.init
        if cfg.pmap:
            wm_opt_state = jax.pmap(wm_opt_init_fn)(wm_params)
        else:
            wm_opt_state = wm_opt_init_fn(wm_params)
        
        self.train_state = BYOLTrainState(
            wm_params=wm_params,
            target_params=target_params,
            wm_opt_state=wm_opt_state
        )
        
        ema = cfg.ema
        discrete = cfg.stoch_discrete_dim > 1
        beta = cfg.beta # trades off Dreamer ELBO loss and BYOL loss
        
        def _get_latent_dist(stats: jnp.ndarray) -> Union[distrax.Distribution, distrax.DistributionLike]:
            if discrete:
                stats = stats.reshape(stats.shape[:-1] + (cfg.stoch_dim, cfg.stoch_discrete_dim))
                dist = distrax.straight_through_wrapper(distrax.OneHotCategorical)(logits=stats)
            else:
                mean, std = jnp.split(stats, 2, -1)
                std = jax.nn.softplus(std) + 0.1
                dist = distrax.Normal(mean, std)
            
            return distrax.Independent(dist, 1)
        
        def _get_img_or_reward_dist(mean: jnp.ndarray, img=False) -> distrax.Distribution:
            mean = jnp.reshape(mean, (-1,) + mean.shape[2:]) # (T, B, ...) -> (T * B, ...)
            dist = distrax.Normal(mean, 1.0)
            dist = distrax.Independent(dist, 3 if img else 1)
            return dist
    
        # define loss functions + update functions
        @jax.jit
        def wm_loss_fn_window_size(wm_params: hk.Params,
                                   target_params: hk.Params,
                                   obs_seq: jnp.ndarray,
                                   action_seq: jnp.ndarray,
                                   reward_seq: jnp.ndarray,
                                   key: jax.random.PRNGKey,
                                   window_size: int) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
            
            T, B = obs_seq.shape[:2]
            pred_key, target_key = jax.random.split(key)

            # don't need to do for all idxs that work, we just do it for index T - window_size so we don't multiple count the same losses
            starting_idx = T - window_size
            obs_window = sliding_window(obs_seq, starting_idx, window_size) # (T, B, *obs_dims), everything except [T - window_size:] 0s, rolled to front
            action_window = sliding_window(action_seq, starting_idx, window_size) # (T, B, action_dim), everything except [T - window_size:] 0s, rolled to front
            reward_window = sliding_window(reward_seq, starting_idx, window_size) # (T, B)
            
            pred_latents, _, dreamer_extras = wm.apply(wm_params, pred_key, obs_window, action_window)
            pred_latents = jnp.reshape(pred_latents, (-1,) + pred_latents.shape[2:]) # (T * B, embed_dim)

            _, target_latents, _ = wm.apply(target_params, target_key, obs_window, action_window) # (T, B, embed_dim)
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
            
            # get dreamer losses and ONLY add to loss vec, not second term (i.e. leave exploration term alone as second term)
            img_mean, reward_mean, post_stats, prior_stats = dreamer_extras
            img_dist = _get_img_or_reward_dist(img_mean, img=True)
            reward_dist = _get_img_or_reward_dist(reward_mean, img=False)
            rec_loss = -img_dist.log_prob(obs_window).mean() - reward_dist.log_prob(reward_window).mean()
            
            post_dist = _get_latent_dist(post_stats)
            prior_dist = _get_latent_dist(prior_stats)
            kl = post_dist.kl_divergence(prior_dist).mean()
            
            metrics = {
                'rec': rec_loss,
                'kl': kl,
                'byol': jnp.mean(byol_loss_vec)
            }
            
            total_loss = jnp.mean(byol_loss_vec) + beta * (kl + rec_loss)
            return total_loss, (byol_loss_vec, metrics)
        
        @jax.jit
        def wm_loss_fn(wm_params: hk.Params,
                       target_params: hk.Params,
                       obs_seq: jnp.ndarray,
                       action_seq: jnp.ndarray,
                       reward_seq: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
            
            T, B = obs_seq.shape[:2]

            def ws_body_fn(ws, curr_state):
                curr_loss, (curr_loss_window, curr_metrics) = curr_state
                loss, (loss_window, metrics) = wm_loss_fn_window_size(wm_params, target_params, obs_seq, action_seq, reward_seq, ws)
                curr_metrics.update({
                    'rec': curr_metrics['rec'] + metrics['rec'],
                    'kl': curr_metrics['kl'] + metrics['kl'],
                    'byol': curr_metrics['byol'] + metrics['byol']
                })
                return curr_loss + loss, (curr_loss_window + loss_window, metrics)
            
            init_metrics = dict(rec=0.0, kl=0.0, byol=0.0)
            init_state = (0.0, (jnp.zeros((T, B)), init_metrics))
            total_loss, (total_loss_window, metrics) = jax.lax.fori_loop(1, T + 1, ws_body_fn, init_state)
            return total_loss / T, (total_loss_window / T, metrics) # take avgs
        
        def update(train_state: BYOLTrainState,
                   obs: jnp.ndarray,
                   actions: jnp.ndarray,
                   rewards: jnp.ndarray,
                   step: int):
            del step
            
            loss_grad_fn = jax.value_and_grad(wm_loss_fn, has_aux=True)
            (loss, (_, metrics)), grads = loss_grad_fn(train_state.wm_params, train_state.target_params, obs, actions, rewards)
            
            if cfg.pmap:
                loss = jax.lax.pmean(loss, axis_name='device') # maybe use jax.tree_util.tree_map later if this doesn't work and is actually needed
                grads = jax.lax.pmean(grads, axis_name='device')
            
            update, new_opt_state = wm_opt.update(grads, train_state.wm_opt_state)
            new_params = optax.apply_updates(train_state.wm_params, update)
            
            new_target_params = target_update_fn(new_params, train_state.target_params, ema)
            new_train_state = BYOLTrainState(wm_params=new_params, target_params=new_target_params, wm_opt_state=new_opt_state)
            
            metrics.update({'wm_loss': loss})
            return new_train_state, metrics
        
        def compute_uncertainty(obs_seq: jnp.ndarray,
                                action_seq: jnp.ndarray,
                                reward_seq: jnp.ndarray,
                                step: int):
            '''Computes transition uncertainties according to part (iv) in BYOL-Explore paper.
            
            :param obs_seq: Sequence of observations, of shape (seq_len, B, obs_dim)
            :param action_seq: Sequence of actions, of shape (seq_len, B, action_dim)
            :param reward_seq: Sequence of rewards, of shape (seq_len, B, 1)
            
            :return uncertainties: Model uncertainties, of shape (seq_len, B).
            '''
            del step
            _, (losses, _) = wm_loss_fn(self.train_state.wm_params, self.train_state.target_params, obs_seq, action_seq, reward_seq)
            # losses are of shape (T, B), result of only BYOL loss accumulation
            return jax.lax.stop_gradient(losses)
        
        self._update = jax.jit(update)
        self._compute_uncertainty = jax.jit(compute_uncertainty)
        
        # whether to parallelize across devices, make sure to have multiple devices here for this for better performance
        if cfg.pmap:
            self._update = jax.pmap(self._update)
            self._compute_uncertainty = jax.pmap(self._compute_uncertainty)
    
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            dill.dump(self.train_state, f, protocol=2)
            
    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                train_state = dill.load(f)
                self.train_state = train_state
        except FileNotFoundError:
            print('cannot load BYOL-Offline model')
            exit()