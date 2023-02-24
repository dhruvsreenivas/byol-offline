import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple
import dill

from byol_offline.agents.agent import Agent
from byol_offline.models.byol_model import WorldModelTrainer
from byol_offline.models.rnd_model import RNDModelTrainer
from byol_offline.networks.actor_critic import DuelingDQN
from byol_offline.agents.agent_utils import *

from utils import ATARI_ENVS, batched_zeros_like
from memory.replay_buffer import Transition

class DQNTrainState(NamedTuple):
    online_params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    rng_key: jax.random.PRNGKey
    
class DQN(Agent):
    def __init__(self, cfg, byol=None, rnd=None):
        # encoder
        assert cfg.task in ATARI_ENVS, "DQN only works for discrete envs like Atari."
        
        dqn_fn = lambda o: DuelingDQN(cfg.action_shape)(o)
        dqn = hk.without_apply_rng(hk.transform(dqn_fn))
        
        # pessimism
        if cfg.aug == 'rnd':
            assert rnd is not None, "Can't use RND when model doesn't exist."
            assert isinstance(rnd, RNDModelTrainer), "Not an RND model trainer--BAD!"
            def aug_fn(obs, acts):
                # dummy step because we delete it anyway
                return rnd.compute_uncertainty(obs, acts, 0)
        elif cfg.aug == 'byol':
            assert byol is not None, "Can't use BYOL-Explore when model doesn't exist."
            assert isinstance(byol, WorldModelTrainer), "Not a BYOL-Explore model trainer--BAD!"
            def aug_fn(obs, acts):
                # dummy step again because we delete it
                return byol.compute_uncertainty(obs, acts, 0)
        else:
            # no pessimism
            def aug_fn(obs, acts):
                return 0.0
            
        # init
        rng = jax.random.PRNGKey(cfg.seed)
        init_key, state_key = jax.random.split(rng, 3)
        params = target_params = dqn.init(init_key, batched_zeros_like(cfg.obs_shape))
        
        opt = optax.adam(cfg.lr)
        opt_state = opt.init(params)
        
        self.train_state = DQNTrainState(
            online_params=params,
            target_params=target_params,
            opt_state=opt_state,
            rng_key=state_key
        )
        
        gamma = cfg.discount
        eps = cfg.eps
        target_update_freq = cfg.target_update_freq
        penalize = cfg.penalize
        penalize_q = cfg.penalize_q
        lam = cfg.lam
        
        def act(obs: jnp.ndarray, step: int, eval_mode: bool = False):
            del step
            qs = dqn.apply(self.train_state.online_params, obs)
            
            best_action = jnp.argmax(qs)
            eps_key, rand_key, state_key = jax.random.split(self.train_state.rng_key, 3)
            p = jax.random.uniform(eps_key)
            
            action = jax.random.choice(rand_key, cfg.action_shape)
            action = jnp.where(p < eps, action, best_action)
            action = jnp.where(eval_mode, best_action, action)
            
            self.train_state = self.train_state._replace(
                rng_key=state_key
            )
            
            return action
        
        def get_aug(observations: jnp.ndarray, actions: jnp.ndarray):
            return aug_fn(observations, actions)
        
        # ====== Losses ======
        @jax.jit
        def dqn_loss(params: hk.Params,
                     target_params: hk.Params,
                     transitions: Transition,
                     key: jax.random.PRNGKey,
                     step: int):
            del key
            del step
            
            # get penalty
            penalty = get_aug(transitions.obs, transitions.actions) # (B)
            
            # penalize rewards
            if penalize and not penalize_q:
                transitions = transitions._replace(rewards=transitions.rewards - lam * penalty)
            
            # get target q values
            nq = dqn.apply(target_params, transitions.next_obs)
            nq = jnp.max(nq, axis=-1)
            nv = transitions.rewards + gamma * (1.0 - transitions.dones) * nq
            
            if penalize and penalize_q:
                # subtract penalties from q values
                    nv = nv - lam * penalty
            
            nv = jax.lax.stop_gradient(nv)
            
            # get current q values
            q = dqn.apply(params, transitions.obs)
            actions = jnp.expand_dims(transitions.actions, -1)
            q = jnp.take_along_axis(q, actions, -1).squeeze()
            
            loss = jnp.mean(jnp.square(q - nv))
            return loss
        
        def update(train_state: DQNTrainState,
                   transitions: Transition,
                   step: int):
            
            update_key, state_key = jax.random.split(train_state.rng_key)
            loss_grad_fn = jax.value_and_grad(dqn_loss)
            loss, grads = loss_grad_fn(train_state.online_params,
                                       train_state.target_params,
                                       transitions,
                                       update_key,
                                       step)
            
            update, new_opt_state = opt.update(grads, train_state.opt_state)
            new_params = optax.apply_updates(train_state.online_params, update)
            
            new_target_params = optax.periodic_update(new_params,
                                                      train_state.target_params,
                                                      step,
                                                      target_update_freq)
            
            new_state = DQNTrainState(
                online_params=new_params,
                target_params=new_target_params,
                opt_state=new_opt_state,
                rng_key=state_key
            )
            metrics = {'dqn_loss': loss}
            
            return new_state, metrics

        self._act = jax.jit(act)
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
            print('cannot load DQN')
            return None