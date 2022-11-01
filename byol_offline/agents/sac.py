import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from typing import NamedTuple
import dill

from byol_offline.models.byol_model import WorldModelTrainer
from byol_offline.models.rnd_model import RNDModelTrainer
from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.actor_critic import *
from byol_offline.agents.agent_utils import *
from utils import MUJOCO_ENVS, flatten_data, batched_zeros_like
from memory.replay_buffer import Transition

class SACTrainState(NamedTuple):
    encoder_params: hk.Params
    actor_params: hk.Params
    critic_params: hk.Params
    critic_target_params: hk.Params
    log_alpha_params: jnp.ndarray
    
    encoder_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    log_alpha_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey
    
class SAC:
    '''SACv2 agent from Denis Yarats' PyTorch SAC repo.'''
    def __init__(self, cfg, byol=None, rnd=None):
        # SET UP
        
        # encoder (if we use BYOL-Explore reward, we can use Dreamer encoder for consistency)
        if cfg.task not in MUJOCO_ENVS:
            if byol is None or cfg.reward_aug == 'rnd':
                encoder_fn = lambda obs: DrQv2Encoder()(obs)
            else:
                encoder_fn = lambda obs: DreamerEncoder(cfg.depth)(obs)
        else:
            encoder_fn = lambda obs: hk.nets.MLP(
                [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim],
                activation=jax.nn.swish
            )(obs)
        encoder = hk.without_apply_rng(hk.transform(encoder_fn))
        
        # actor + critic
        actor_fn = lambda obs, std: SACActor(cfg.action_shape, cfg.hidden_dim)(obs, std)
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        critic_fn = lambda obs, action: SACCritic(cfg.hidden_dim)(obs, action)
        critic = hk.without_apply_rng(hk.transform(critic_fn))

        # reward pessimism (currently using different encoder than the DrQv2 stuff--maybe have to change)
        if cfg.reward_aug == 'rnd':
            assert rnd is not None, "Can't use RND when model doesn't exist."
            assert type(rnd) == RNDModelTrainer, "Not an RND model trainer--BAD!"
            def reward_aug_fn(obs, acts):
                # dummy step because we delete it anyway
                return rnd.compute_uncertainty(obs, acts, 0)
        else:
            assert byol is not None, "Can't use BYOL-Explore when model doesn't exist."
            assert type(byol) == WorldModelTrainer, "Not a BYOL-Explore model trainer--BAD!"
            def reward_aug_fn(obs, acts):
                # dummy step again because we delete it
                return byol.compute_uncertainty(obs, acts, 0)
        
        # initialization
        rng = jax.random.PRNGKey(cfg.seed)
        key1, key2, key3, key4 = jax.random.split(rng, 4)
        
        encoder_params = encoder.init(key1, batched_zeros_like(cfg.obs_shape))
        
        if cfg.task not in MUJOCO_ENVS:
            if byol is None or cfg.reward_aug == 'rnd':
                actor_params = actor.init(key2, batched_zeros_like(20000), jnp.zeros(1))
                critic_params = critic_target_params = critic.init(key3, batched_zeros_like(20000), batched_zeros_like(cfg.action_shape))
            else:
                actor_params = actor.init(key2, batched_zeros_like(4096), jnp.zeros(1))
                critic_params = critic_target_params = critic.init(key3, batched_zeros_like(4096), batched_zeros_like(cfg.action_shape))
        else:
            actor_params = actor.init(key2, batched_zeros_like(cfg.hidden_dim), jnp.zeros(1))
            critic_params = critic_target_params = critic.init(key3, batched_zeros_like(cfg.hidden_dim), batched_zeros_like(cfg.action_shape))
        
        log_alpha = jnp.asarray(0., dtype=jnp.float32)
        
        # optimizers
        encoder_opt = optax.adam(cfg.encoder_lr)
        encoder_opt_state = encoder_opt.init(encoder_params)
        
        actor_opt = optax.adam(cfg.actor_lr)
        actor_opt_state = actor_opt.init(actor_params)
        
        critic_opt = optax.adam(cfg.critic_lr)
        critic_opt_state = critic_opt.init(critic_params)
        
        log_alpha_opt = optax.adam(cfg.alpha_lr)
        log_alpha_opt_state = critic_opt.init(log_alpha)
        
        # train state
        self.train_state = SACTrainState(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            log_alpha_params=log_alpha,
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            log_alpha_opt_state=log_alpha_opt_state,
            rng_key=key4
        )
        
        reward_lambda = cfg.reward_lambda
        self.ema = cfg.ema
        target_update_frequency = cfg.target_update_frequency
        target_entropy = -np.prod(cfg.action_shape)
        
        # =================== START OF ALL FNS ===================
        
        def act(obs: jnp.ndarray, step: int, eval_mode: bool=False):
            del step
            
            rng, key = jax.random.split(self.train_state.rng_key)
            
            encoder_params = self.train_state.encoder_params
            features = encoder.apply(encoder_params, obs) # don't need batch dim here

            actor_params = self.train_state.actor_params
            dist = actor.apply(actor_params, features)
            
            mean = dist.mean()
            sample = dist.sample(seed=key)
            action = jnp.where(eval_mode, mean, sample)
            
            self.train_state = self.train_state._replace(
                rng_key=rng
            ) # no need to return, as this is not jitted
            
            return action

        def get_reward_aug(observations: jnp.ndarray, actions: jnp.ndarray):
            return reward_aug_fn(observations, actions)
        
        # =================== WARMSTARTING ===================
        
        @jax.jit
        def bc_loss(encoder_params: hk.Params,
                    actor_params: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    key: jax.random.PRNGKey,
                    step: int):
            del step

            features = encoder.apply(encoder_params, obs) # no need to expand batch dim
            dist = actor.apply(actor_params, features)
            
            sampled_actions = dist.sample(seed=key)
            loss = jnp.mean(jnp.square(sampled_actions - actions)) # mse loss, exactly like DrQ + BC
            return loss
        
        def bc_update(train_state: SACTrainState, transitions: Transition, step: int):
            rng, key = jax.random.split(train_state.rng_key)
            
            loss_grad_fn = jax.value_and_grad(bc_loss, argnums=(0, 1))
            loss, (encoder_grads, actor_grads) = loss_grad_fn(train_state.encoder_params, train_state.actor_params, transitions.obs, transitions.actions, key, step)
            
            # encoder update
            enc_update, new_enc_opt_state = encoder_opt.update(encoder_grads, train_state.encoder_opt_state)
            new_enc_params = optax.apply_updates(train_state.encoder_params, enc_update)
            
            # actor update
            act_update, new_act_opt_state = actor_opt.update(actor_grads, train_state.actor_opt_state)
            new_actor_params = optax.apply_updates(train_state.actor_params, act_update)
            
            new_train_state = train_state._replace(
                encoder_params=new_enc_params,
                actor_params=new_actor_params,
                encoder_opt_state=new_enc_opt_state,
                actor_opt_state=new_act_opt_state,
                rng_key=rng
            )
            
            return new_train_state, {'bc_loss': loss}
        
        # =================== AGENT LOSS/UPDATE FUNCTIONS ===================
        
        @jax.jit
        def critic_loss_byol(encoder_params: hk.Params,
                             critic_params: hk.Params,
                             critic_target_params: hk.Params,
                             actor_params: hk.Params,
                             log_alpha: jnp.ndarray,
                             transitions: Transition,
                             key: jax.random.PRNGKey,
                             step: int):
            del step
            
            # get reward aug
            reward_pen = get_reward_aug(transitions.obs, transitions.actions)
            transitions = transitions._replace(rewards=transitions.rewards - reward_lambda * jax.lax.stop_gradient(reward_pen)) # don't want extra gradients going back to encoder params

            # flatten transitions (BYOL loss is sequential)
            transitions = flatten_data(transitions)
            
            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            next_features = encoder.apply(encoder_params, transitions.next_obs)
            
            # get the targets
            dist = actor.apply(actor_params, next_features)
            next_actions = dist.sample(seed=key)
            log_probs = dist.log_prob(next_actions).sum(-1, keepdims=True)
            
            nq1, nq2 = critic.apply(critic_target_params, next_features, next_actions)
            v = jnp.squeeze(jnp.minimum(nq1, nq2) - jnp.exp(log_alpha) * log_probs)
            target_q = jax.lax.stop_gradient(transitions.rewards + cfg.discount * (1.0 - transitions.dones) * v)

            # get the actual q values
            q1, q2 = critic.apply(critic_params, features, transitions.actions)
            
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss

        @jax.jit
        def critic_loss_rnd(encoder_params: hk.Params,
                            critic_params: hk.Params,
                            critic_target_params: hk.Params,
                            actor_params: hk.Params,
                            log_alpha: jnp.ndarray,
                            transitions: Transition,
                            key: jax.random.PRNGKey,
                            step: int):
            del step
            
            # get reward aug
            reward_pen = get_reward_aug(transitions.obs, transitions.actions)
            transitions = transitions._replace(rewards=transitions.rewards - reward_lambda * jax.lax.stop_gradient(reward_pen)) # don't want extra gradients going back to encoder params
            
            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            next_features = encoder.apply(encoder_params, transitions.next_obs)
            
            # get the targets
            dist = actor.apply(actor_params, next_features)
            next_actions = dist.sample(seed=key)
            log_probs = dist.log_prob(next_actions).sum(-1, keepdims=True)
            
            nq1, nq2 = critic.apply(critic_target_params, features, next_actions)
            v = jnp.minimum(nq1, nq2) - jnp.exp(log_alpha) * log_probs
            target_q = jax.lax.stop_gradient(transitions.rewards + cfg.discount * (1.0 - transitions.dones) * v)

            # get the actual q values
            q1, q2 = critic.apply(critic_params, features, transitions.actions)
            
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss
        
        @jax.jit
        def actor_loss_byol(actor_params: hk.Params,
                            encoder_params: hk.Params,
                            critic_params: hk.Params,
                            log_alpha: jnp.ndarray,
                            transitions: Transition,
                            key: jax.random.PRNGKey,
                            step: int):
            del step

            # flatten all data
            transitions = flatten_data(transitions)
            
            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            features = jax.lax.stop_gradient(features) # just so we don't have to deal with gradients passing through to world model

            dist = actor.apply(actor_params, features)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions).sum(-1, keepdims=True)
            
            q1, q2 = critic.apply(critic_params, features, actions)
            min_q = jnp.minimum(q1, q2)
            actor_loss = jnp.exp(log_alpha) * log_probs - min_q
            return jnp.mean(actor_loss)
        
        @jax.jit
        def actor_loss_rnd(actor_params: hk.Params,
                           encoder_params: hk.Params,
                           critic_params: hk.Params,
                           log_alpha: jnp.ndarray,
                           transitions: Transition,
                           key: jax.random.PRNGKey,
                           step: int):
            del step
            
            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            features = jax.lax.stop_gradient(features) # just so we don't have to deal with gradients passing through to world model

            dist = actor.apply(actor_params, features)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions).sum(-1, keepdims=True)
            
            q1, q2 = critic.apply(critic_params, features, actions)
            min_q = jnp.minimum(q1, q2)
            actor_loss = jnp.exp(log_alpha) * log_probs - min_q
            return jnp.mean(actor_loss)
        
        @jax.jit
        def alpha_loss_byol(log_alpha: jnp.ndarray,
                            encoder_params: hk.Params,
                            actor_params: hk.Params,
                            transitions: Transition,
                            key: jax.random.PRNGKey,
                            step: int):
            del step

            # flatten data again
            transitions = flatten_data(transitions)
            
            features = encoder.apply(encoder_params, transitions.obs)
            features = jax.lax.stop_gradient(features)
            
            dist = actor.apply(actor_params, features)
            actions = dist.sample(seed=key)
            log_prob = dist.log_prob(actions).sum(-1, keepdims=True)
            
            alpha_loss = jnp.exp(log_alpha) - jax.lax.stop_gradient(-log_prob - target_entropy)
            return jnp.mean(alpha_loss)
        
        @jax.jit
        def alpha_loss_rnd(log_alpha: jnp.ndarray,
                           encoder_params: hk.Params,
                           actor_params: hk.Params,
                           transitions: Transition,
                           key: jax.random.PRNGKey,
                           step: int):
            del step
            
            features = encoder.apply(encoder_params, transitions.obs)
            features = jax.lax.stop_gradient(features)
            
            dist = actor.apply(actor_params, features)
            actions = dist.sample(seed=key)
            log_prob = dist.log_prob(actions).sum(-1, keepdims=True)
            
            alpha_loss = jnp.exp(log_alpha) - jax.lax.stop_gradient(-log_prob - target_entropy)
            return jnp.mean(alpha_loss)
        
        @jax.jit
        def update_critic(train_state: SACTrainState,
                          transitions: Transition,
                          key: jax.random.PRNGKey,
                          step: int):
            # assume that the transitions is a sequence of consecutive (s, a, r, s', d) tuples
            if cfg.reward_aug == 'byol':
                loss_grad_fn = jax.value_and_grad(critic_loss_byol, argnums=(0, 1))
            else:
                loss_grad_fn = jax.value_and_grad(critic_loss_rnd, argnums=(0, 1))
            
            loss, (encoder_grads, critic_grads) = loss_grad_fn(
                train_state.encoder_params,
                train_state.critic_params,
                train_state.critic_target_params,
                train_state.actor_params,
                train_state.log_alpha_params,
                transitions,
                key,
                step
            )
            
            # update both encoder and critic
            enc_update, enc_opt_state = encoder_opt.update(encoder_grads, train_state.encoder_opt_state)
            new_enc_params = optax.apply_updates(train_state.encoder_params, enc_update)
            
            critic_update, critic_opt_state = critic_opt.update(critic_grads, train_state.critic_opt_state)
            new_critic_params = optax.apply_updates(train_state.critic_params, critic_update)
            
            metrics = {
                'critic_loss': loss
            }
            
            new_vars = {
                'encoder_params': new_enc_params,
                'encoder_opt_state': enc_opt_state,
                'critic_params': new_critic_params,
                'critic_opt_state': critic_opt_state
            }
            return metrics, new_vars
        
        @jax.jit
        def update_actor(train_state: SACTrainState,
                         transitions: Transition,
                         key: jax.random.PRNGKey,
                         step: int):
            if cfg.reward_aug == 'byol':
                loss_grad_fn = jax.value_and_grad(actor_loss_byol)
            else:
                loss_grad_fn = jax.value_and_grad(actor_loss_rnd)
            
            a_loss, a_grads = loss_grad_fn(train_state.actor_params,
                                           train_state.encoder_params,
                                           train_state.critic_params,
                                           train_state.log_alpha_params, 
                                           transitions,
                                           key,
                                           step)
            
            # update actor params
            actor_update, actor_opt_state = actor_opt.update(a_grads, train_state.actor_opt_state)
            new_actor_params = optax.apply_updates(train_state.actor_params, actor_update)
            
            metrics = {
                'actor_loss': a_loss
            }
            
            new_vars = {
                'actor_params': new_actor_params,
                'actor_opt_state': actor_opt_state
            }
            return metrics, new_vars
        
        @jax.jit
        def update_log_alpha(train_state: SACTrainState,
                             transitions: Transition,
                             key: jax.random.PRNGKey,
                             step: int):
            if cfg.reward_aug == 'byol':
                loss_grad_fn = jax.value_and_grad(alpha_loss_byol)
            else:
                loss_grad_fn = jax.value_and_grad(alpha_loss_rnd)
            
            alpha_loss, alpha_grads = loss_grad_fn(log_alpha, encoder_params, actor_params, transitions, key, step)
            
            # update alpha
            alpha_update, new_log_alpha_opt_state = log_alpha_opt.update(alpha_grads, train_state.log_alpha_opt_state)
            new_log_alpha_params = optax.apply_updates(train_state.log_alpha_params, alpha_update)
            
            metrics = {
                'alpha_loss': alpha_loss
            }
            
            new_vars = {
                'log_alpha_params': new_log_alpha_params,
                'log_alpha_opt_state': new_log_alpha_opt_state
            }
            return metrics, new_vars
        
        def update(train_state: SACTrainState,
                   transitions: Transition,
                   step: int):
            key1, key2, key3, key4 = jax.random.split(train_state.rng_key, 3)
            
            # critic update
            critic_metrics, critic_new_vars = update_critic(
                train_state,
                transitions,
                key1,
                step
            )
            
            upd_train_state = train_state._replace(
                encoder_params=critic_new_vars['encoder_params'],
                critic_params=critic_new_vars['critic_params'],
                encoder_opt_state=critic_new_vars['encoder_opt_state'],
                critic_opt_state=critic_new_vars['critic_opt_state']
            )
            
            # actor update
            actor_metrics, actor_new_vars = update_actor(
                upd_train_state,
                transitions,
                key2,
                step
            )
            
            upd_train_state = upd_train_state._replace(
                actor_params=actor_new_vars['actor_params'],
                actor_opt_state=actor_new_vars['actor_opt_state']
            )
            
            # update log alpha
            log_alpha_metrics, log_alpha_new_vars = update_log_alpha(
                upd_train_state,
                transitions,
                key3,
                step
            )
            
            upd_train_state = upd_train_state._replace(
                log_alpha_params=log_alpha_new_vars['log_alpha_params'],
                log_alpha_opt_state=log_alpha_new_vars['log_alpha_opt_state']
            )
            
            # update critic target
            new_target_params = update_target(
                upd_train_state.critic_params,
                upd_train_state.critic_target_params,
                self.ema
            )
            
            new_target_params = jax.lax.cond(
                step % target_update_frequency == 0,
                lambda _: new_target_params,
                lambda _: upd_train_state.critic_target_params,
                operand=None
            )
            
            new_train_state = upd_train_state._replace(
                critic_target_params=new_target_params,
                rng_key=key4
            )
            
            # logging
            metrics = {**critic_metrics, **actor_metrics, **log_alpha_metrics}
            return new_train_state, metrics

        self._act = act
        self._bc_update = jax.jit(bc_update)
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
            print('cannot load SAC')
            return None