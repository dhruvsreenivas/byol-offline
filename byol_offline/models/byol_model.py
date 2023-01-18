import jax
import jax.numpy as jnp
import haiku as hk
import optax
import distrax
import dill
from typing import Tuple, NamedTuple

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
            self.encoder = DreamerEncoder(cfg.depth)
        else:
            self.encoder = DrQv2Encoder()
        
        self.closed_gru = ClosedLoopGRU(cfg.gru_hidden_size)
        self.open_gru = hk.GRU(cfg.gru_hidden_size)
        
        # prior + post mlps
        self.discrete = cfg.stoch_discrete_dim > 1
        dist_dim = cfg.stoch_dim * cfg.stoch_discrete_dim if self.discrete else 2 * cfg.stoch_dim
        self.prior_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim),
            jax.nn.elu,
            hk.Linear(dist_dim)
        ])
        self.post_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim),
            jax.nn.elu,
            hk.Linear(dist_dim)
        ])
        self.reward_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim),
            jax.nn.elu,
            hk.Linear(1)
        ])
        
        if cfg.dreamer:
            self.decoder = DreamerDecoder(cfg.frame_stack * 3, cfg.depth)
            self.predictor = BYOLPredictor(4096)
        else:
            self.decoder = DrQv2Decoder(cfg.frame_stack * 3)
            self.predictor = BYOLPredictor(20000)
        
    def __call__(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        # first get embeddings
        B = obs.shape[1]
        embeddings = hk.BatchApply(self.encoder)(obs)
        state = self.closed_gru.initial_state(B)
        
        embedding, action = embeddings[0], actions[0] # x_0, a_0
        state, _ = self.closed_gru(embedding, action, state) # h_0
        latent = self.predictor(state)
        latent = jnp.expand_dims(latent, 0)
        
        # collect h_t, latents
        states, _ = hk.dynamic_unroll(self.open_gru, actions[1:], initial_state=state) # [h_t]
        latents = hk.BatchApply(self.predictor)(states)
        
        # === dreamer stuff ===
        # img prediction
        states = jnp.concatenate([state, states]) # all deter states, shape (T, B, deter_dim)
        embeddings = jnp.concatenate([embedding, embeddings]) # (T, B, embed_dim)
        hz = jnp.concatenate([states, embeddings], axis=-1) # (T, B, deter_dim + embed_dim)
        img_mean = hk.BatchApply(self.decoder)(hz) # (T, B, H, W, C)
        
        # post + prior & reward
        post_stats = hk.BatchApply(self.post_mlp)(hz) # (T, B, dist_dim)
        prior_stats = hk.BatchApply(self.prior_mlp)(states) # (T, B, dist_dim)
        reward_mean = hk.BatchApply(self.reward_mlp)(hz) # (T, B, 1)
        
        # === byol stuff ===
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeddings
        
        # return
        dreamer_extras = (img_mean, reward_mean, post_stats, prior_stats)
        return pred_latents, embeddings, dreamer_extras

class MLPLatentWorldModel(hk.Module):
    '''Latent world model for D4RL tasks. Primarily inspired by MOPO repository.'''
    def __init__(self, cfg):
        super().__init__()
        
        # nets
        self.encoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.repr_dim],
            activation=jax.nn.swish
        )
        
        self.closed_gru = ClosedLoopGRU(cfg.gru_hidden_size)
        self.open_gru = hk.GRU(cfg.gru_hidden_size)
        
        # prior + post mlps
        self.discrete = cfg.stoch_discrete_dim > 1
        dist_dim = cfg.stoch_dim * cfg.stoch_discrete_dim if self.discrete else 2 * cfg.stoch_dim
        self.prior_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim),
            jax.nn.elu,
            hk.Linear(dist_dim)
        ])
        self.post_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim),
            jax.nn.elu,
            hk.Linear(dist_dim)
        ])
        self.reward_mlp = hk.Sequential([
            hk.Linear(cfg.hidden_dim),
            jax.nn.elu,
            hk.Linear(1)
        ])
        
        self.decoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.obs_shape[0]],
            activation=jax.nn.swish
        )
        self.predictor = BYOLPredictor(cfg.repr_dim)
    
    def __call__(self, obs, actions):
        # obs should be of shape (T, B, obs_dim), actions of shape (T, B, action_dim)
        # first get embeddings
        B = obs.shape[1]
        embeddings = hk.BatchApply(self.encoder)(obs)
        state = self.closed_gru.initial_state(B)
        
        embedding, action = embeddings[0], actions[0]
        state, _ = self.closed_gru(embedding, action, state)
        latent = self.predictor(state)
        latent = jnp.expand_dims(latent, 0)
        
        states, _ = hk.dynamic_unroll(self.open_gru, actions[1:], initial_state=state)
        latents = hk.BatchApply(self.predictor)(states)
        
        # === dreamer stuff ===
        # img prediction
        states = jnp.concatenate([state, states]) # all deter states, shape (T, B, deter_dim)
        embeddings = jnp.concatenate([embedding, embeddings]) # (T, B, embed_dim)
        hz = jnp.concatenate([states, embeddings], axis=-1) # (T, B, deter_dim + embed_dim)
        state_mean = hk.BatchApply(self.decoder)(hz) # (T, B, H, W, C)
        
        # post + prior & rewards
        post_stats = hk.BatchApply(self.post_mlp)(hz) # (T, B, dist_dim)
        prior_stats = hk.BatchApply(self.prior_mlp)(states) # (T, B, dist_dim)
        reward_mean = hk.BatchApply(self.reward_mlp)(hz)
        
        # === byol stuff ===
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeddings
        
        # return
        dreamer_extras = (state_mean, reward_mean, post_stats, prior_stats)
        return pred_latents, embeddings, dreamer_extras
    
class WorldModelTrainer:
    '''World model trainer.'''
    def __init__(self, cfg):
        # set up
        if cfg.task in MUJOCO_ENVS:
            wm_fn = lambda o, a: MLPLatentWorldModel(cfg.d4rl)(o, a)
        else:
            wm_fn = lambda o, a: ConvLatentWorldModel(cfg.vd4rl)(o, a)
        
        wm = hk.without_apply_rng(hk.transform(wm_fn))
        
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
        beta = cfg.beta # trades off Dreamer loss and BYOL loss
        
        @jax.jit
        def get_latent_dist(stats: jnp.ndarray) -> distrax.Distribution:
            if discrete:
                stats = stats.reshape(stats.shape[:-1] + (cfg.stoch_dim, cfg.stoch_discrete_dim))
                dist = distrax.OneHotCategorical(logits=stats)
                dist = distrax.straight_through_wrapper(dist)
            else:
                mean, std = jnp.split(stats, 2, -1)
                std = jax.nn.softplus(std) + 0.1
                dist = distrax.Normal(mean, std)
            
            return distrax.Independent(dist, 1)
        
        @jax.jit
        def get_img_or_reward_dist(mean: jnp.ndarray, img=False) -> distrax.Distribution:
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
                                   window_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            T, B = obs_seq.shape[:2]

            # don't need to do for all idxs that work, we just do it for index T - window_size so we don't multiple count the same losses
            starting_idx = T - window_size
            obs_window = sliding_window(obs_seq, starting_idx, window_size) # (T, B, *obs_dims), everything except [T - window_size:] 0s, rolled to front
            action_window = sliding_window(action_seq, starting_idx, window_size) # (T, B, action_dim), everything except [T - window_size:] 0s, rolled to front
            reward_window = sliding_window(reward_seq, starting_idx, window_size) # (T, B, 1) I think
            
            pred_latents, _, dreamer_extras = wm.apply(wm_params, obs_window, action_window)
            pred_latents = jnp.reshape(pred_latents, (-1,) + pred_latents.shape[2:]) # (T * B, embed_dim)

            _, target_latents, _ = wm.apply(target_params, obs_window, action_window) # (T, B, embed_dim)
            target_latents = jnp.reshape(target_latents, (-1,) + target_latents.shape[2:]) # (T * B, embed_dim)

            # normalize latents
            pred_latents = l2_normalize(pred_latents, axis=-1)
            target_latents = l2_normalize(target_latents, axis=-1)

            # take L2 loss
            mses = jnp.square(pred_latents - jax.lax.stop_gradient(target_latents)) # (T * B, embed_dim)
            mses = jnp.reshape(mses, (-1, B) + mses.shape[1:]) # (T, B, embed_dim)
            mses = sliding_window(mses, 0, window_size) # zeros out losses we don't care about (i.e. past window size)
            mses = jnp.roll(mses, shift=starting_idx, axis=0)
            byol_loss_vec = jnp.sum(mses, -1) # (T, B)
            
            # get dreamer losses and ONLY add to loss vec, not second term (i.e. leave exploration term alone as second term)
            img_mean, reward_mean, post_stats, prior_stats = dreamer_extras
            img_dist = get_img_or_reward_dist(img_mean, img=True)
            reward_dist = get_img_or_reward_dist(reward_mean, img=False)
            rec_loss = -img_dist.log_prob(obs_window).mean() - reward_dist.log_prob(reward_window).mean()
            
            post_dist = get_latent_dist(post_stats)
            prior_dist = get_latent_dist(prior_stats)
            kl = post_dist.kl_divergence(prior_dist).mean()
            
            total_loss = jnp.mean(byol_loss_vec) + beta * (kl + rec_loss)
            return total_loss, byol_loss_vec
        
        @jax.jit
        def wm_loss_fn(wm_params: hk.Params,
                       target_params: hk.Params,
                       obs_seq: jnp.ndarray,
                       action_seq: jnp.ndarray,
                       reward_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            T, B = obs_seq.shape[:2]

            def ws_body_fn(ws, curr_state):
                curr_loss, curr_loss_window = curr_state
                loss, loss_window = wm_loss_fn_window_size(wm_params, target_params, obs_seq, action_seq, reward_seq, ws)
                return curr_loss + loss, curr_loss_window + loss_window
            
            init_state = (0.0, jnp.zeros((T, B)))
            total_loss, total_loss_window = jax.lax.fori_loop(1, T + 1, ws_body_fn, init_state)
            return total_loss / T, total_loss_window / T # take avgs
        
        def update(train_state: BYOLTrainState,
                   obs: jnp.ndarray,
                   actions: jnp.ndarray,
                   rewards: jnp.ndarray,
                   step: int):
            del step
            
            loss_grad_fn = jax.value_and_grad(wm_loss_fn, has_aux=True)
            (loss, _), grads = loss_grad_fn(train_state.wm_params, train_state.target_params, obs, actions, rewards)
            
            if cfg.pmap:
                loss = jax.lax.pmean(loss, axis_name='device') # maybe use jax.tree_util.tree_map later if this doesn't work and is actually needed
                grads = jax.lax.pmean(grads, axis_name='device')
            
            update, new_opt_state = wm_opt.update(grads, train_state.wm_opt_state)
            new_params = optax.apply_updates(train_state.wm_params, update)
            
            new_target_params = target_update_fn(new_params, train_state.target_params, ema)
            
            metrics = {
                'wm_loss': loss
            }
            
            new_train_state = BYOLTrainState(wm_params=new_params, target_params=new_target_params, wm_opt_state=new_opt_state)
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
            _, losses, _ = wm_loss_fn(self.train_state.wm_params, self.train_state.target_params, obs_seq, action_seq)
            # losses are of shape (T, B), result of loss accumulation
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