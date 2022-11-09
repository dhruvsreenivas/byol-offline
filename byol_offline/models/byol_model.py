import jax
import jax.numpy as jnp
import haiku as hk
import optax
import dill
from typing import Tuple, NamedTuple

from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
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
        
        if cfg.dreamer:
            self.predictor = BYOLPredictor(4096)
        else:
            self.predictor = BYOLPredictor(20000)
        
    def __call__(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
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
        
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeddings
        return pred_latents, embeddings

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
        
        pred_latents = jnp.concatenate([latent, latents])
        return pred_latents, embeddings
    
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
        
        self.ema = cfg.ema
    
        # define loss functions + update functions
        @jax.jit
        def wm_loss_fn_window_size(wm_params: hk.Params,
                                   target_params: hk.Params,
                                   obs_seq: jnp.ndarray,
                                   action_seq: jnp.ndarray,
                                   window_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            T, B = obs_seq.shape[:2]

            # don't need to do for all idxs that work, we just do it for index T - window_size so we don't multiple count the same losses
            starting_idx = T - window_size
            obs_window = sliding_window(obs_seq, starting_idx, window_size) # (T, B, *obs_dims), everything except [T - window_size:] 0s, rolled to front
            action_window = sliding_window(action_seq, starting_idx, window_size) # (T, B, action_dim), everything except [T - window_size:] 0s, rolled to front
            
            pred_latents, _ = wm.apply(wm_params, obs_window, action_window)
            pred_latents = jnp.reshape(pred_latents, (-1,) + pred_latents.shape[2:]) # (T * B, embed_dim)

            _, target_latents = wm.apply(target_params, obs_window, action_window) # (T, B, embed_dim)
            target_latents = jnp.reshape(target_latents, (-1,) + target_latents.shape[2:]) # (T * B, embed_dim)

            # normalize latents
            pred_latents = l2_normalize(pred_latents, axis=-1)
            target_latents = l2_normalize(target_latents, axis=-1)

            # take L2 loss
            mses = jnp.square(pred_latents - jax.lax.stop_gradient(target_latents)) # (T * B, embed_dim)
            mses = jnp.reshape(mses, (-1, B) + mses.shape[1:]) # (T, B, embed_dim)
            mses = sliding_window(mses, 0, window_size) # zeros out losses we don't care about (i.e. past window size)
            mses = jnp.roll(mses, shift=starting_idx, axis=0)
            loss_vec = jnp.sum(mses, -1) # (T, B)
            return jnp.mean(loss_vec), loss_vec
        
        @jax.jit
        def wm_loss_fn(wm_params: hk.Params,
                       target_params: hk.Params,
                       obs_seq: jnp.ndarray,
                       action_seq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            T, B = obs_seq.shape[:2]

            def ws_body_fn(ws, curr_state):
                curr_loss, curr_loss_window = curr_state
                loss, loss_window = wm_loss_fn_window_size(wm_params, target_params, obs_seq, action_seq, ws)
                return curr_loss + loss, curr_loss_window + loss_window
            
            init_state = (0.0, jnp.zeros((T, B)))
            total_loss, total_loss_window = jax.lax.fori_loop(1, T + 1, ws_body_fn, init_state)
            return total_loss / T, total_loss_window / T # take avgs
        
        def update(train_state: BYOLTrainState,
                   obs: jnp.ndarray,
                   actions: jnp.ndarray,
                   step: int):
            del step
            
            loss_grad_fn = jax.value_and_grad(wm_loss_fn, has_aux=True)
            (loss, _), grads = loss_grad_fn(train_state.wm_params, train_state.target_params, obs, actions)
            
            if cfg.pmap:
                loss = jax.lax.pmean(loss, axis_name='device') # maybe use jax.tree_util.tree_map later if this doesn't work and is actually needed
                grads = jax.lax.pmean(grads, axis_name='device')
            
            update, new_opt_state = wm_opt.update(grads, train_state.wm_opt_state)
            new_params = optax.apply_updates(train_state.wm_params, update)
            
            new_target_params = target_update_fn(new_params, train_state.target_params, self.ema)
            
            metrics = {
                'wm_loss': loss
            }
            
            new_train_state = BYOLTrainState(wm_params=new_params, target_params=new_target_params, wm_opt_state=new_opt_state)
            return new_train_state, metrics
        
        def compute_uncertainty(obs_seq: jnp.ndarray,
                                action_seq: jnp.ndarray,
                                step: int):
            '''Computes transition uncertainties according to part (iv) in BYOL-Explore paper.
            
            :param obs_seq: Sequence of observations, of shape (seq_len, B, obs_dim)
            :param action_seq: Sequence of actions, of shape (seq_len, B, action_dim)
            
            :return uncertainties: Model uncertainties, of shape (seq_len, B).
            '''
            del step
            _, losses = wm_loss_fn(self.train_state.wm_params, self.train_state.target_params, obs_seq, action_seq)
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