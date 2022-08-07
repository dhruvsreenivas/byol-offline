from collections import namedtuple
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import dill
from typing import NamedTuple

from byol_offline.networks.encoder import DreamerEncoder
from byol_offline.networks.rnn import *
from byol_offline.reward_augs.byol import BYOLPredictor

class WorldModelTrainState(NamedTuple):
    wm_params: hk.Params
    target_params: hk.Params
    wm_opt_state: optax.OptState

class LatentWorldModel(hk.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # nets
        self.encoder = DreamerEncoder(cfg.obs_shape, cfg.depth)
        self.closed_gru = ClosedLoopGRU(cfg.gru_hidden_size)
        self.open_gru = hk.GRU(cfg.gru_hidden_size)
        
        self.predictor = BYOLPredictor(4096)
        
    def __call__(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        # first get embeddings
        T, B = obs.shape[0], obs.shape[1]
        obs = jnp.reshape(obs, (-1,) + obs.shape[2:])
        embeddings = self.encoder(obs)
        embeddings = jnp.reshape(embeddings, (T, -1) + embeddings.shape[1:]) # (T, B, embed_dim)
        state = self.closed_gru.initial_state(B)
        
        embedding, action = embeddings[0], actions[0]
        state, _ = self.closed_gru(embedding, action, state)
        latent = self.predictor(state)
        latent = jnp.expand_dims(latent, 0)
        
        states, _ = hk.dynamic_unroll(self.open_gru, actions[1:], initial_state=state)
        states = jnp.reshape(states, (-1,) + states.shape[2:])
        latents = self.predictor(states)
        latents = jnp.reshape(latents, (-1, B) + latents.shape[1:])
        
        pred_latents = jnp.concatenate([latent, latents])
        return pred_latents, embeddings
    
class WorldModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        wm_fn = lambda o, a: LatentWorldModel(cfg)(o, a)
        self.wm = hk.without_apply_rng(hk.transform(wm_fn))
        
        # params (make sure to initialize phi correctly)
        key = jax.random.PRNGKey(cfg.seed)
        k1, k2 = jax.random.split(key)
        
        # really dumb that I have to make an entire new world model just to get the targets to update properly
        wm_params = self.wm.init(k1, jnp.zeros((2, 1) + tuple(cfg.obs_shape)), jnp.zeros((2, 1) + tuple(cfg.action_shape)))
        target_params = self.wm.init(k2, jnp.zeros((2, 1) + tuple(cfg.obs_shape)), jnp.zeros((2, 1) + tuple(cfg.action_shape)))
        
        # optimizer
        self.wm_opt = optax.adam(cfg.lr)
        wm_opt_state = self.wm_opt.init(wm_params)
        
        self.train_state = WorldModelTrainState(wm_params=wm_params, target_params=target_params, wm_opt_state=wm_opt_state)
        
        self.ema = cfg.ema
    
    def update(self, obs, actions, step):
        del step
        
        @jax.jit
        def wm_loss_fn(wm_params, target_params, obs_seq, action_seq):
            pred_latents, _ = self.wm.apply(wm_params, obs_seq, action_seq)
            pred_latents = jnp.reshape(pred_latents, (-1,) + pred_latents.shape[2:]) # (T * B, embed_dim)
            
            _, target_latents = self.wm.apply(target_params, obs_seq, action_seq)
            target_latents = jnp.reshape(target_latents, (-1,) + target_latents.shape[2:]) # (T * B, embed_dim)
            target_latents = jax.lax.stop_gradient(target_latents)
            
            # normalize latents
            pred_latent_norms = jnp.linalg.norm(pred_latents, ord=2, axis=-1, keepdims=True)
            target_latent_norms = jnp.linalg.norm(target_latents, ord=2, axis=-1, keepdims=True)
            pred_latents = pred_latents / pred_latent_norms
            target_latents = target_latents / target_latent_norms
            
            loss = jnp.mean(jnp.square(pred_latents - target_latents))
            return loss
        
        @jax.jit
        def target_update_fn(params, target_params):
            new_target_params = jax.tree_util.tree_map(lambda x, y: self.ema * x + (1.0 - self.ema) * y, target_params, params)
            return new_target_params
        
        loss_grad_fn = jax.value_and_grad(wm_loss_fn)
        loss, grads = loss_grad_fn(self.train_state.wm_params, self.train_state.target_params, obs, actions)
        
        update, new_opt_state = self.wm_opt.update(grads, self.train_state.wm_opt_state)
        new_params = optax.apply_updates(self.train_state.wm_params, update)
        
        new_target_params = target_update_fn(new_params, self.train_state.target_params)
        
        metrics = {
            'wm_loss': loss.item()
        }
        
        self.train_state = WorldModelTrainState(wm_params=new_params, target_params=new_target_params, wm_opt_state=new_opt_state)
        return metrics
    
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            dill.dump(self.train_state, f, protocol=2)
            
    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                train_state = dill.load(f)
            self.train_state = train_state
        except FileNotFoundError:
            print('cannot load model')
            return None