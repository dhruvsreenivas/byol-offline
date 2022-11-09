import jax
import jax.numpy as jnp
import haiku as hk
import optax
import dill
from typing import NamedTuple

from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.predictors import RNDPredictor
from utils import MUJOCO_ENVS, batched_zeros_like

class RNDTrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    
class ConvRNDModel(hk.Module):
    def __init__(self, cfg):
        super().__init__()
    
        # nets
        if cfg.dreamer:
            self.encoder = DreamerEncoder(cfg.depth)
        else:
            self.encoder = DrQv2Encoder()
            
        self.predictor = RNDPredictor(cfg)
        
    def __call__(self, obs):
        # Observations are expected to be of size (B, H, W, C)
        reprs = self.encoder(obs)
        return self.predictor(reprs)
    
class ConvRNDModelWithActions(ConvRNDModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def __call__(self, obs, actions):
        oa = jnp.concatenate([obs, actions], axis=-1)
        return super().__call__(oa)

class MLPRNDModel(hk.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim],
            activation=jax.nn.swish
        )
        self.predictor = RNDPredictor(cfg)
    
    def __call__(self, obs):
        reprs = self.encoder(obs)
        return self.predictor(reprs)

class MLPRNDModelWithActions(MLPRNDModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def __call__(self, obs, actions):
        oa = jnp.concatenate([obs, actions], axis=-1)
        return super().__call__(oa)
    
class RNDModelTrainer:
    '''RND model trainer.'''
    def __init__(self, cfg):
        # initialize here before defining all loss/update fns
        if cfg.task in MUJOCO_ENVS:
            if cfg.cat_actions:
                rnd_fn = lambda o, a: MLPRNDModelWithActions(cfg.d4rl)(o, a)
            else:
                rnd_fn = lambda o: MLPRNDModel(cfg.d4rl)(o)
        else:
            if cfg.cat_actions:
                rnd_fn = lambda o, a: ConvRNDModelWithActions(cfg.vd4rl)(o, a)
            else:
                rnd_fn = lambda o: ConvRNDModel(cfg.vd4rl)(o)
        
        rnd = hk.without_apply_rng(hk.transform(rnd_fn))
        
        # params
        key = jax.random.PRNGKey(cfg.seed)
        k1, k2 = jax.random.split(key)
        
        if cfg.cat_actions:
            rnd_params = rnd.init(k1, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
            target_params = rnd.init(k2, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
        else:
            rnd_params = rnd.init(k1, batched_zeros_like(cfg.obs_shape))
            target_params = rnd.init(k2, batched_zeros_like(cfg.obs_shape))
            
        # if parallelizing, put copy of params on all devices
        if cfg.pmap:
            device_lst = jax.devices()
            rnd_params = jax.device_put_replicated(rnd_params, device_lst)
            target_params = jax.device_put_replicated(target_params, device_lst)
        
        # optimizer
        if cfg.optim == 'adam':
            rnd_opt = optax.adam(cfg.lr)
        elif cfg.optim == 'adamw':
            rnd_opt = optax.adamw(cfg.lr)
        else:
            rnd_opt = optax.sgd(cfg.lr, momentum=0.9)
        
        opt_init_fn = rnd_opt.init
        if cfg.pmap:
            rnd_opt_state = jax.pmap(opt_init_fn)(rnd_params)
        else:
            rnd_opt_state = opt_init_fn(rnd_params)
        
        self.train_state = RNDTrainState(
            params=rnd_params,
            target_params=target_params,
            opt_state=rnd_opt_state
        )
    
        @jax.jit
        def rnd_loss_fn(params, target_params, obs):
            output = rnd.apply(params, obs)
            target_output = rnd.apply(target_params, obs)
            
            # no need to do jax.lax.stop_gradient, as gradient is only taken w.r.t. first param
            return jnp.mean(jnp.square(target_output - output))
        
        @jax.jit
        def rnd_loss_fn_actions(params, target_params, obs, actions):
            output = rnd.apply(params, obs, actions)
            target_output = rnd.apply(target_params, obs, actions)
            
            return jnp.mean(jnp.square(target_output - output))
    
        def update(train_state: RNDTrainState, obs: jnp.ndarray, actions: jnp.ndarray, step: int):
            del step
            
            if cfg.cat_actions:
                loss_grad_fn = jax.value_and_grad(rnd_loss_fn_actions)
                loss, grads = loss_grad_fn(train_state.params, train_state.target_params, obs, actions)
            else:
                loss_grad_fn = jax.value_and_grad(rnd_loss_fn)
                loss, grads = loss_grad_fn(train_state.params, train_state.target_params, obs)
                
            # avg gradients across parallel devices
            if cfg.pmap:
                loss = jax.lax.pmean(loss, axis_name='device')
                grads = jax.lax.pmean(grads, axis_name='device')
            
            update, new_opt_state = rnd_opt.update(grads, train_state.opt_state)
            new_params = optax.apply_updates(train_state.params, update)
            
            metrics = {
                'rnd_loss': loss
            }
            
            new_train_state = RNDTrainState(
                params=new_params,
                target_params=train_state.target_params,
                opt_state=new_opt_state
            )
        
            return new_train_state, metrics
    
        def compute_uncertainty(obs, actions, step):
            '''Computes RND uncertainty bonus.'''
            del step
            
            if cfg.cat_actions:
                online_output = rnd.apply(self.train_state.params, obs, actions)
                target_output = rnd.apply(self.train_state.target_params, obs, actions)
            else:
                online_output = rnd.apply(self.train_state.params, obs)
                target_output = rnd.apply(self.train_state.target_params, obs)
            
            squared_diff = jnp.square(target_output - online_output).sum(-1)
            return jax.lax.stop_gradient(squared_diff)
        
        # define update + uncertainty computation
        self._update = jax.jit(update)
        self._compute_uncertainty = jax.jit(compute_uncertainty)
        
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
            print('cannot load RND model')
            exit()
        
        