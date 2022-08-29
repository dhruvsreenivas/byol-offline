import jax
import jax.numpy as jnp
import haiku as hk
import optax
import dill
import functools
from typing import NamedTuple

from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.predictors import RNDPredictor

class RNDTrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    
class RNDModel(hk.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
        # nets
        if cfg.dreamer:
            self.encoder = DreamerEncoder(cfg.obs_shape, cfg.depth)
        else:
            self.encoder = DrQv2Encoder(cfg.obs_shape)
            
        self.predictor = RNDPredictor(cfg)
        
    def __call__(self, obs):
        # Observations are expected to be of size (B, H, W, C)
        reprs = self.encoder(obs)
        return self.predictor(reprs)
    
class RNDModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        rnd_fn = lambda obs: RNDModel(cfg)(obs)
        self.rnd = hk.without_apply_rng(hk.transform(rnd_fn))
        
        # params
        key = jax.random.PRNGKey(cfg.seed)
        k1, k2 = jax.random.split(key)
        
        rnd_params = self.rnd.init(k1, jnp.zeros((1,) + tuple(cfg.obs_shape)))
        target_params = self.rnd.init(k2, jnp.zeros((1,) + tuple(cfg.obs_shape)))
        
        # optimizer
        self.rnd_opt = optax.adam(cfg.lr)
        rnd_opt_state = self.rnd_opt.init(rnd_params)
        
        self.train_state = RNDTrainState(
            params=rnd_params,
            target_params=target_params,
            opt_state=rnd_opt_state
        )
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def rnd_loss_fn(self, params, target_params, obs):
        output = self.rnd.apply(params, obs)
        target_output = self.rnd.apply(target_params, obs)
        
        # no need to do jax.lax.stop_gradient, as gradient is only taken w.r.t. first param
        return jnp.mean(jnp.square(target_output - output))
    
    def update(self, obs, step):
        del step
        
        grad_fn = jax.value_and_grad(self.rnd_loss_fn)
        loss, grads = grad_fn(self.train_state.params, self.train_state.target_params, obs)
        
        update, new_opt_state = self.rnd_opt.update(grads, self.train_state.opt_state)
        new_params = optax.apply_updates(self.train_state.params, update)
        
        metrics = {
            'rnd_loss': loss.item()
        }
        
        self.train_state = RNDTrainState(
            params=new_params,
            target_params=self.train_state.target_params,
            opt_state=new_opt_state
        )
        
        return metrics
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def compute_uncertainty(self, obs, actions, step):
        del step
        del actions
        
        online_output = self.rnd.apply(self.train_state.params, obs)
        target_output = self.rnd.apply(self.train_state.target_params, obs)
        
        squared_diff = jnp.square(target_output - online_output).sum(-1)
        return jax.lax.stop_gradient(squared_diff)
    
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
        