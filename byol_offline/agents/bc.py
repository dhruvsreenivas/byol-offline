import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple, Optional
import dill

from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.actor_critic import BCActor
from utils import MUJOCO_ENVS, flatten_data, batched_zeros_like
from memory.replay_buffer import Transition

class BCTrainState:
    params: hk.Params
    opt_state: optax.OptState
    encoder_params: Optional[hk.Params] = None
    encoder_opt_state: Optional[optax.OptState] = None
    rng_key: jax.random.PRNGKey
    
class BC:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # encoder
        if cfg.task not in MUJOCO_ENVS:
            encoder_fn = lambda obs: DrQv2Encoder()(obs)
            self.encoder = hk.without_apply_rng(hk.transform(encoder_fn))
        
        # policy itself
        actor_fn = lambda obs: BCActor(cfg.hidden_dim, cfg.action_shape)(obs)
        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        # initialization
        rng = jax.random.PRNGKey(cfg.seed)
        k1, k2, k3 = jax.random.split(rng, 3)
        if cfg.task not in MUJOCO_ENVS:
            encoder_params = self.encoder.init(k1, batched_zeros_like(cfg.obs_shape))
            actor_params = self.actor.init(k2, batched_zeros_like(20000))
        else:
            encoder_params = None
            actor_params = self.actor.init(k2, batched_zeros_like(cfg.obs_shape))
            
        # optimizer
        if cfg.task not in MUJOCO_ENVS:
            self.encoder_opt = optax.adam(cfg.encoder_lr)
            encoder_opt_state = self.encoder_opt.init(encoder_params)
        else:
            encoder_opt_state = None
        
        self.actor_opt = optax.adam(cfg.actor_lr)
        actor_opt_state = self.actor_opt.init(actor_params)
        
        self.train_state = BCTrainState(
            params=actor_params,
            opt_state=actor_opt_state,
            encoder_params=encoder_params,
            encoder_opt_state=encoder_opt_state,
            rng_key=k3
        )
        
        # =================== START OF ALL FNS ===================
        
        def act(obs: jnp.ndarray, eval_mode: bool):
            '''Choose an action to execute in env.'''
            rng, key = jax.random.split(self.train_state.rng_key)
            
            if cfg.task not in MUJOCO_ENVS:
                encoder_params = self.train_state.encoder_params
                features = self.encoder.apply(encoder_params, obs) # don't need batch dim here
            else:
                features = obs
            
            actor_params = self.train_state.actor_params
            dist = self.actor.apply(actor_params, features)
            
            mean = dist.mean()
            sample = dist.sample(seed=key)
            action = jnp.where(eval_mode, mean, sample)
            
            self.train_state = self.train_state._replace(
                rng_key=rng
            ) # no need to return, as this is not jitted (although probably possible to make it jitted here)
            
            return action
        
        @jax.jit
        def loss_fn(encoder_params: hk.Params,
                    actor_params: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    key: jax.random.PRNGKey,
                    step: int):
            del step
            
            if cfg.task not in MUJOCO_ENVS: # should be jittable
                features = self.encoder.apply(encoder_params, obs) # no need to expand batch dim
            else:
                features = obs
            
            dist = self.actor.apply(actor_params, features)
            sampled_actions = dist.sample(seed=key)
            loss = jnp.mean(jnp.square(sampled_actions - actions)) # mse loss, exactly like DrQ + BC
            return loss
        
        def update(train_state: BCTrainState, transitions: Transition, step: int):
            rng, key = jax.random.split(train_state.rng_key)
            
            loss_grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
            loss, (encoder_grads, actor_grads) = loss_grad_fn(train_state.encoder_params, train_state.actor_params, transitions.obs, transitions.actions, key, step)
            
            # encoder update
            enc_update, new_enc_opt_state = self.encoder_opt.update(encoder_grads, train_state.encoder_opt_state)
            new_enc_params = optax.apply_updates(train_state.encoder_params, enc_update)
            
            # actor update
            act_update, new_act_opt_state = self.actor_opt.update(actor_grads, train_state.actor_opt_state)
            new_actor_params = optax.apply_updates(train_state.actor_params, act_update)
            
            new_train_state = train_state._replace(
                encoder_params=new_enc_params,
                actor_params=new_actor_params,
                encoder_opt_state=new_enc_opt_state,
                actor_opt_state=new_act_opt_state,
                rng_key=rng
            )
            
            return new_train_state, {'bc_loss': loss}
        
        self._act = act
        self._update = jax.jit(update)
        
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            dill.dump(self.train_state, f, protocol=2)
            
    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                train_state = dill.load(f)
            self.train_state = train_state
        except FileNotFoundError:
            print('cannot load BC')
            return None
        