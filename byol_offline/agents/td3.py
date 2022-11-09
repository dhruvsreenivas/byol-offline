import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple
import dill

from byol_offline.models.byol_model import WorldModelTrainer
from byol_offline.models.rnd_model import RNDModelTrainer
from byol_offline.networks.actor_critic import TD3Actor, TD3Critic
from byol_offline.agents.agent_utils import *
from utils import batched_zeros_like
from memory.replay_buffer import Transition

class TD3TrainState(NamedTuple):
    actor_params: hk.Params
    target_actor_params: hk.Params
    critic_params: hk.Params
    target_critic_params: hk.Params
    
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey

class TD3:
    '''TD3 implementation for MuJoCo only.'''
    def __init__(self, cfg, byol=None, rnd=None):
        # SET UP
        
        actor_fn = lambda o: TD3Actor(cfg.hidden_dim, cfg.action_shape, cfg.max_action)(o)
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        critic_fn = lambda o, a: TD3Critic(cfg.hidden_dim)(o, a)
        critic = hk.without_apply_rng(hk.transform(critic_fn))
        
        # reward pessimism
        if cfg.reward_aug == 'rnd':
            assert rnd is not None, "Can't use RND when model doesn't exist."
            assert type(rnd) == RNDModelTrainer, "Not an RND model trainer--BAD!"
            def reward_aug_fn(obs, acts):
                # dummy step because we delete it anyway
                return rnd._compute_uncertainty(obs, acts, 0)
        elif cfg.reward_aug == 'byol':
            assert byol is not None, "Can't use BYOL-Explore when model doesn't exist."
            assert type(byol) == WorldModelTrainer, "Not a BYOL-Explore model trainer--BAD!"
            def reward_aug_fn(obs, acts):
                # dummy step again because we delete it
                return byol._compute_uncertainty(obs, acts, 0)
        else:
            # no reward augmentation
            def reward_aug_fn(obs, acts):
                return jnp.float32(0.0)
            
        # initialization
        rng = jax.random.PRNGKey(cfg.seed)
        key1, key2, key3 = jax.random.split(rng, 3)
        
        actor_params = target_actor_params = actor.init(key1, batched_zeros_like(cfg.obs_shape))
        critic_params = target_critic_params = critic.init(key2, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
        
        # optimizer init
        actor_opt = optax.adam(cfg.actor_lr)
        actor_opt_state = actor_opt.init(actor_params)
        
        critic_opt = optax.adam(cfg.critic_lr)
        critic_opt_state = critic_opt.init(critic_params)
        
        self.train_state = TD3TrainState(
            actor_params=actor_params,
            target_actor_params=target_actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            rng_key=key3
        )
        
        # other TD3 hparams
        penalize = cfg.penalize
        reward_min = cfg.reward_min
        reward_max = cfg.reward_max
        reward_lambda = cfg.reward_lambda
        max_action = cfg.max_action
        discount = cfg.discount
        ema = cfg.ema
        policy_noise = cfg.policy_noise * max_action
        noise_clip = cfg.noise_clip * max_action
        policy_update_freq = cfg.policy_update_freq
        
        # ================== START OF ALL FNS ==================
        
        def act(obs: jnp.ndarray, step: int, eval_mode: bool):
            del step
            del eval_mode
            
            actor_params = self.train_state.actor_params
            return actor.apply(actor_params, obs)

        def get_reward_aug(observations: jnp.ndarray, actions: jnp.ndarray):
            return reward_aug_fn(observations, actions)
        
        @jax.jit
        def critic_loss(critic_params: hk.Params,
                        target_critic_params: hk.Params,
                        target_actor_params: hk.Params,
                        transitions: Transition,
                        key: jax.random.PRNGKey,
                        step: int):
            del step
            
            if penalize:
                reward_pen = get_reward_aug(transitions.obs, transitions.actions)
                penalized_rewards = get_penalized_rewards(transitions.rewards, reward_pen, reward_lambda, reward_min, reward_max)
                transitions = transitions._replace(rewards=penalized_rewards) # make sure gradients don't go back through world model
            
            # targets
            actions = transitions.actions
            noise = jax.random.normal(key, shape=actions.shape) * policy_noise
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            
            next_actions = actor.apply(target_actor_params, transitions.next_obs)
            next_actions = jnp.clip(next_actions + noise, -max_action, max_action)
            
            target_q1, target_q2 = critic.apply(target_critic_params, transitions.next_obs, next_actions)
            target_q = jnp.minimum(target_q1, target_q2)
            target_v = jax.lax.stop_gradient(transitions.rewards + discount * (1.0 - transitions.dones) * target_q)
            
            q1, q2 = critic.apply(critic_params, transitions.obs, transitions.actions)
            q_loss = jnp.mean(jnp.square(q1 - target_v)) + jnp.mean(jnp.square(q2 - target_v))
            return q_loss
        
        @jax.jit
        def actor_loss(actor_params: hk.Params,
                       critic_params: hk.Params,
                       transitions: Transition,
                       step: int):
            del step
            
            actions = actor.apply(actor_params, transitions.obs)
            q1, _ = critic.apply(critic_params, transitions.obs, actions)
            actor_loss = -jnp.mean(q1)
            return actor_loss
        
        @jax.jit
        def update_critic(train_state: TD3TrainState,
                          transitions: Transition,
                          key: jax.random.PRNGKey,
                          step: int):
            
            loss_grad_fn = jax.value_and_grad(critic_loss)
            loss, grads = loss_grad_fn(train_state.critic_params,
                                       train_state.target_critic_params,
                                       train_state.target_actor_params,
                                       transitions,
                                       key,
                                       step)
            
            update, new_critic_opt_state = critic_opt.update(grads, train_state.critic_opt_state)
            new_critic_params = optax.apply_updates(train_state.critic_params, update)
            
            metrics = {
                'critic_loss': loss
            }
            
            new_vars = {
                'critic_params': new_critic_params,
                'critic_opt_state': new_critic_opt_state
            }
            return metrics, new_vars
        
        @jax.jit
        def update_actor(train_state: TD3TrainState,
                         transitions: Transition,
                         step: int):
            
            loss_grad_fn = jax.value_and_grad(actor_loss)
            loss, grads = loss_grad_fn(train_state.actor_params,
                                       train_state.critic_params,
                                       transitions,
                                       step)
            
            update, new_actor_opt_state = actor_opt.update(grads, train_state.actor_opt_state)
            new_actor_params = optax.apply_updates(train_state.actor_params, update)
            
            metrics = {
                'actor_loss': loss
            }
            
            new_vars = {
                'actor_params': new_actor_params,
                'actor_opt_state': new_actor_opt_state
            }
            
            return metrics, new_vars
        
        def update(train_state: TD3TrainState,
                   transitions: Transition,
                   step: int):
            
            key1, key2 = jax.random.split(train_state.rng_key)
            
            critic_metrics, critic_new_vars = update_critic(
                train_state,
                transitions,
                key1,
                step
            )
            
            upd_train_state = train_state._replace(
                critic_params=critic_new_vars['critic_params'],
                critic_opt_state=critic_new_vars['critic_opt_state']
            )
            
            def update_actor_and_target(train_state: TD3TrainState):
                actor_metrics, actor_new_vars = update_actor(
                    train_state,
                    transitions,
                    step
                )
                
                upd_train_state = train_state._replace(
                    actor_params=actor_new_vars['actor_params'],
                    actor_opt_state=actor_new_vars['actor_opt_state']
                )
                
                new_target_critic_params = update_target(
                    upd_train_state.critic_params,
                    upd_train_state.target_critic_params,
                    ema
                )
                
                new_target_actor_params = update_target(
                    upd_train_state.actor_params,
                    upd_train_state.target_actor_params,
                    ema
                )
                
                upd_train_state = upd_train_state._replace(
                    target_critic_params=new_target_critic_params,
                    target_actor_params=new_target_actor_params
                )
                
                return upd_train_state, actor_metrics
            
            def stay_the_same(train_state: TD3TrainState):
                empty_dict = {
                    'actor_loss': jnp.inf
                }
                return train_state, empty_dict

            upd_train_state, actor_metrics = jax.lax.cond(
                step % policy_update_freq == 0,
                update_actor_and_target,
                stay_the_same,
                operand=upd_train_state
            )
            
            # update RNG key for next time
            new_train_state = upd_train_state._replace(
                rng_key=key2
            )
            
            metrics = {**critic_metrics, **actor_metrics}
            return new_train_state, metrics
        
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
            print('cannot load TD3')
            return None