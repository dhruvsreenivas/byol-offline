import jax
import jax.numpy as jnp
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

class DDPGTrainState(NamedTuple):
    encoder_params: hk.Params
    actor_params: hk.Params
    critic_params: hk.Params
    critic_target_params: hk.Params
    
    encoder_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey # updated every time we act or learn

class DDPG:
    '''DDPG agent from DrQv2 (without the data augmentations).'''
    def __init__(self, cfg, byol=None, rnd=None):
        # SET UP
        
        # encoder (if we use BYOL-Explore reward, we can use Dreamer encoder for consistency)
        if cfg.task not in MUJOCO_ENVS:
            if byol is None or cfg.aug == 'rnd':
                encoder_fn = lambda obs: DrQv2Encoder()(obs)
            else:
                encoder_fn = lambda obs: DreamerEncoder(cfg.depth)(obs)
        else:
            encoder_fn = lambda obs: hk.nets.MLP(
                [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim],
                activation=jax.nn.swish
            )(obs)
        encoder = hk.without_apply_rng(hk.transform(encoder_fn))
        
        # actor + critic
        actor_fn = lambda obs, std: DDPGActor(cfg.action_shape, cfg.feature_dim, cfg.hidden_dim)(obs, std)
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        critic_fn = lambda obs, action: DDPGCritic(cfg.feature_dim, cfg.hidden_dim)(obs, action)
        critic = hk.without_apply_rng(hk.transform(critic_fn))
        
        # reward pessimism (currently using different encoder than the DrQv2 stuff--maybe have to change)
        if cfg.aug == 'rnd':
            assert rnd is not None, "Can't use RND when model doesn't exist."
            assert type(rnd) == RNDModelTrainer, "Not an RND model trainer--BAD!"
            def aug_fn(obs, acts):
                # dummy step because we delete it anyway
                return rnd._compute_uncertainty(obs, acts, 0)
        elif cfg.aug == 'byol':
            assert byol is not None, "Can't use BYOL-Explore when model doesn't exist."
            assert type(byol) == WorldModelTrainer, "Not a BYOL-Explore model trainer--BAD!"
            def aug_fn(obs, acts):
                # dummy step again because we delete it
                return byol._compute_uncertainty(obs, acts, 0)
        else:
            # no reward pessimism
            def aug_fn(obs, acts):
                return 0.0
        
        # initialization
        rng = jax.random.PRNGKey(cfg.seed)
        key1, key2, key3, key4 = jax.random.split(rng, 4)
        
        encoder_params = encoder.init(key1, batched_zeros_like(cfg.obs_shape))
        
        if cfg.task not in MUJOCO_ENVS:
            if byol is None or cfg.aug == 'rnd':
                actor_params = actor.init(key2, batched_zeros_like(20000), jnp.zeros(1))
                critic_params = critic_target_params = critic.init(key3, batched_zeros_like(20000), batched_zeros_like(cfg.action_shape))
            else:
                actor_params = actor.init(key2, batched_zeros_like(4096), jnp.zeros(1))
                critic_params = critic_target_params = critic.init(key3, batched_zeros_like(4096), batched_zeros_like(cfg.action_shape))
        else:
            actor_params = actor.init(key2, batched_zeros_like(cfg.hidden_dim), jnp.zeros(1))
            critic_params = critic_target_params = critic.init(key3, batched_zeros_like(cfg.hidden_dim), batched_zeros_like(cfg.action_shape))
        
        # optimizers
        encoder_opt = optax.adam(cfg.encoder_lr)
        encoder_opt_state = encoder_opt.init(encoder_params)
        
        actor_opt = optax.adam(cfg.actor_lr)
        actor_opt_state = actor_opt.init(actor_params)
        
        critic_opt = optax.adam(cfg.critic_lr)
        critic_opt_state = critic_opt.init(critic_params)
        
        # train state
        self.train_state = DDPGTrainState(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            rng_key=key4
        )
        
        # hyperparameters for training
        reward_min = cfg.reward_min
        reward_max = cfg.reward_max
        lam = cfg.lam
        ema = cfg.ema
        init_std = cfg.init_std
        final_std = cfg.final_std
        std_duration = cfg.std_duration
        std_clip_val = cfg.std_clip_val
        self.update_every_steps = cfg.update_every_steps
        
        # =================== START OF ALL FNS ===================
        
        def stddev_schedule(step: int):
            '''Linear standard deviation schedule.'''
            mix = jnp.clip(step / std_duration, 0.0, 1.0)
            return (1.0 - mix) * init_std + mix * final_std
        
        def act(obs: jnp.ndarray, step: int, eval_mode: bool):
            '''Choose an action to execute in env.'''
            rng, key = jax.random.split(self.train_state.rng_key)
            
            encoder_params = self.train_state.encoder_params
            features = encoder.apply(encoder_params, obs) # don't need batch dim here
            
            std = stddev_schedule(step)
            actor_params = self.train_state.actor_params
            dist = actor.apply(actor_params, features, std)
            
            mean = dist.mean()
            sample = dist.sample(seed=key, clip=std_clip_val)
            action = jnp.where(eval_mode, mean, sample)
            
            self.train_state = self.train_state._replace(
                rng_key=rng
            ) # no need to return, as this is not jitted
            
            return action
        
        def get_aug(observations: jnp.ndarray, actions: jnp.ndarray):
            return aug_fn(observations, actions)
    
        # =================== WARMSTARTING ===================
        
        @jax.jit
        def bc_loss(encoder_params: hk.Params,
                    actor_params: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    key: jax.random.PRNGKey,
                    step: int):
            std = stddev_schedule(step)
            features = encoder.apply(encoder_params, obs) # no need to expand batch dim
            dist = actor.apply(actor_params, features, std)
            
            sampled_actions = dist.sample(seed=key, clip=std_clip_val)
            loss = jnp.mean(jnp.square(sampled_actions - actions)) # mse loss, exactly like DrQ + BC
            return loss
        
        def bc_update(train_state: DDPGTrainState, transitions: Transition, step: int):
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
                             transitions: Transition,
                             key: jax.random.PRNGKey,
                             step: int):
            # get reward penalty
            reward_pen = get_aug(transitions.obs, transitions.actions)
            penalized_rewards = get_penalized_rewards(transitions.rewards, reward_pen, lam, reward_min, reward_max)
            transitions = transitions._replace(rewards=penalized_rewards) # don't want extra gradients going back to encoder params

            # flatten data, as this is BYOL critic loss (we've already added reward so no need to treat as sequence anymore)
            transitions = flatten_data(transitions)
            
            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            next_features = encoder.apply(encoder_params, transitions.next_obs)
            
            # get the targets
            std = stddev_schedule(step)
            dist = actor.apply(actor_params, next_features, std)
            next_actions = dist.sample(seed=key, clip=std_clip_val)
            
            nq1, nq2 = critic.apply(critic_target_params, next_features, next_actions)
            nv = jnp.squeeze(jnp.minimum(nq1, nq2))
            target_q = jax.lax.stop_gradient(transitions.rewards + cfg.discount * (1.0 - transitions.dones) * nv)

            # get the actual q values
            q1, q2 = critic.apply(critic_params, features, transitions.actions)
            
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss

        @jax.jit
        def critic_loss_rnd(encoder_params: hk.Params,
                            critic_params: hk.Params,
                            critic_target_params: hk.Params, 
                            actor_params: hk.Params,
                            transitions: Transition,
                            key, step):
            # get reward penalty
            reward_pen = get_aug(transitions.obs, transitions.actions)
            penalized_rewards = get_penalized_rewards(transitions.rewards, reward_pen, lam, reward_min, reward_max)
            transitions = transitions._replace(rewards=penalized_rewards) # don't want extra gradients going back to encoder params

            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            next_features = encoder.apply(encoder_params, transitions.next_obs)
            
            # get the targets
            std = stddev_schedule(step)
            dist = actor.apply(actor_params, next_features, std)
            next_actions = dist.sample(seed=key, clip=std_clip_val)
            
            nq1, nq2 = critic.apply(critic_target_params, next_features, next_actions)
            nv = jnp.squeeze(jnp.minimum(nq1, nq2))
            target_q = jax.lax.stop_gradient(transitions.rewards + cfg.discount * (1.0 - transitions.dones) * nv)

            # get the actual q values
            q1, q2 = critic.apply(critic_params, features, transitions.actions)
            
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss
        
        @jax.jit
        def actor_loss_byol(actor_params: hk.Params,
                            encoder_params: hk.Params,
                            critic_params: hk.Params,
                            transitions: Transition,
                            key: jax.random.PRNGKey,
                            step: int):
            # flatten data
            transitions = flatten_data(transitions)

            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            features = jax.lax.stop_gradient(features) # just so we don't have to deal with gradients passing through no matter what

            std = stddev_schedule(step)
            dist = actor.apply(actor_params, features, std)
            actions = dist.sample(seed=key, clip=std_clip_val)
            
            q1, q2 = critic.apply(critic_params, features, actions)
            actor_loss = -jnp.mean(jnp.minimum(q1, q2))
            return actor_loss
        
        @jax.jit
        def actor_loss_rnd(actor_params: hk.Params,
                           encoder_params: hk.Params,
                           critic_params: hk.Params,
                           transitions: Transition,
                           key: jax.random.PRNGKey,
                           step: int):
            # encode observations
            features = encoder.apply(encoder_params, transitions.obs)
            features = jax.lax.stop_gradient(features) # just so we don't have to deal with gradients passing through no matter what

            std = stddev_schedule(step)
            dist = actor.apply(actor_params, features, std)
            actions = dist.sample(seed=key, clip=std_clip_val)
            
            q1, q2 = critic.apply(critic_params, features, actions)
            actor_loss = -jnp.mean(jnp.minimum(q1, q2))
            return actor_loss
        
        @jax.jit
        def update_critic(train_state: DDPGTrainState,
                          transitions: Transition,
                          key: jax.random.PRNGKey,
                          step: int):
            # assume that the transitions is a sequence of consecutive (s, a, r, s', d) tuples
            if cfg.aug == 'byol':
                loss_grad_fn = jax.value_and_grad(critic_loss_byol, argnums=(0, 1))
            else:
                loss_grad_fn = jax.value_and_grad(critic_loss_rnd, argnums=(0, 1))
            
            loss, (encoder_grads, critic_grads) = loss_grad_fn(
                train_state.encoder_params,
                train_state.critic_params,
                train_state.critic_target_params,
                train_state.actor_params,
                transitions,
                key,
                step
            )
            
            # update both encoder and critic
            enc_update, new_enc_opt_state = encoder_opt.update(encoder_grads, train_state.encoder_opt_state)
            new_enc_params = optax.apply_updates(train_state.encoder_params, enc_update)
            
            critic_update, new_critic_opt_state = critic_opt.update(critic_grads, train_state.critic_opt_state)
            new_critic_params = optax.apply_updates(train_state.critic_params, critic_update)
            
            metrics = {
                'critic_loss': loss
            }
            
            new_vars = {
                'encoder_params': new_enc_params,
                'encoder_opt_state': new_enc_opt_state,
                'critic_params': new_critic_params,
                'critic_opt_state': new_critic_opt_state
            }
            return metrics, new_vars
        
        @jax.jit
        def update_actor(train_state: DDPGTrainState,
                         transitions: Transition,
                         key: jax.random.PRNGKey,
                         step: int):
            if cfg.aug == 'byol':
                loss_grad_fn = jax.value_and_grad(actor_loss_byol)
            else:
                loss_grad_fn = jax.value_and_grad(actor_loss_rnd)
            
            a_loss, a_grads = loss_grad_fn(train_state.actor_params,
                                           train_state.encoder_params,
                                           train_state.critic_params,
                                           transitions,
                                           key,
                                           step)
            
            # update actor only
            actor_update, new_actor_opt_state = actor_opt.update(a_grads, train_state.actor_opt_state)
            new_actor_params = optax.apply_updates(actor_update, train_state.actor_params)
            
            metrics = {
                'actor_loss': a_loss
            }
            
            new_vars = {
                'actor_params': new_actor_params,
                'actor_opt_state': new_actor_opt_state
            }
            return metrics, new_vars
        
        def update(train_state: DDPGTrainState,
                   transitions: Transition,
                   step: int):
            
            key1, key2, key3 = jax.random.split(train_state.rng_key, 3)
            
            def update_all():
                '''Actually update everything.'''
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
                
                # update actor
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
                
                # update critic target
                new_target_params = update_target(
                    upd_train_state.critic_params,
                    upd_train_state.critic_target_params,
                    ema
                )
                
                new_train_state = upd_train_state._replace(
                    critic_target_params=new_target_params,
                    rng_key=key3
                )
                
                # logging
                metrics = {**critic_metrics, **actor_metrics}
                return new_train_state, metrics
            
            empty_dict = {
                'critic_loss': jnp.inf,
                'actor_loss': jnp.inf
            } # just have to make it so we don't log these
            curr_state = (train_state, empty_dict)
            
            out = jax.lax.cond(step % self.update_every_steps == 0,
                               lambda _: update_all(),
                               lambda _: curr_state,
                               operand=None)
            
            return out
        
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
            print('cannot load DDPG')
            return None
        