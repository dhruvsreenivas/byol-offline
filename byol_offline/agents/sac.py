import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple
import functools
import dill

from byol_offline.models.byol_model import WorldModelTrainer
from byol_offline.models.rnd_model import RNDModelTrainer
from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.actor_critic import *
from byol_offline.agents.agent_utils import *
from utils import MUJOCO_ENVS

class SACTrainState(NamedTuple):
    encoder_params: hk.Params
    actor_params: hk.Params
    critic_params: hk.Params
    critic_target_params: hk.Params
    log_alpha_params: hk.Params
    
    encoder_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    log_alpha_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey
    
class SAC:
    '''SACv2 agent from Denis Yarats' PyTorch SAC repo.'''
    def __init__(self, cfg, byol=None, rnd=None):
        self.cfg = cfg
        
        # encoder (if we use BYOL-Explore reward, we can use Dreamer encoder for consistency)
        if cfg.task not in MUJOCO_ENVS:
            if byol is None or cfg.reward_aug == 'rnd':
                encoder_fn = lambda obs: DrQv2Encoder(cfg.obs_shape)(obs)
            else:
                encoder_fn = lambda obs: DreamerEncoder(cfg.obs_shape, cfg.depth)(obs)
        else:
            encoder_fn = lambda obs: hk.nets.MLP(
                [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim],
                activation=jax.nn.swish
            )(obs)
        self.encoder = hk.without_apply_rng(hk.transform(encoder_fn))
        
        # actor + critic
        actor_fn = lambda obs, std: SACActor(cfg.action_shape, cfg.hidden_dim)(obs, std)
        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        critic_fn = lambda obs, action: SACCritic(cfg.hidden_dim)(obs, action)
        self.critic = hk.without_apply_rng(hk.transform(critic_fn))

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
        self.reward_aug = reward_aug_fn
        
        # initialization
        rng = jax.random.PRNGKey(cfg.seed)
        key1, key2, key3, key4 = jax.random.split(rng, 4)
        
        encoder_params = self.encoder.init(key1, jnp.zeros((1,) + tuple(cfg.obs_shape)))
        
        if cfg.task not in MUJOCO_ENVS:
            if byol is None or cfg.reward_aug == 'rnd':
                actor_params = self.actor.init(key2, jnp.zeros((1, 20000)), jnp.zeros(1))
                critic_params = critic_target_params = self.critic.init(key3, jnp.zeros((1, 20000)), jnp.zeros((1,) + tuple(cfg.action_shape)))
            else:
                actor_params = self.actor.init(key2, jnp.zeros((1, 4096)), jnp.zeros(1))
                critic_params = critic_target_params = self.critic.init(key3, jnp.zeros((1, 4096)), jnp.zeros((1,) + tuple(cfg.action_shape)))
        else:
            actor_params = self.actor.init(key2, jnp.zeros((1, cfg.hidden_dim)), jnp.zeros(1))
            critic_params = critic_target_params = self.critic.init(key3, jnp.zeros((1, cfg.hidden_dim)), jnp.zeros((1,) + tuple(cfg.action_shape)))
        
        log_alpha = jnp.asarray(0., dtype=jnp.float32)
        
        # optimizers
        self.encoder_opt = optax.adam(cfg.encoder_lr)
        encoder_opt_state = self.encoder_opt.init(encoder_params)
        
        self.actor_opt = optax.adam(cfg.actor_lr)
        actor_opt_state = self.actor_opt.init(actor_params)
        
        self.critic_opt = optax.adam(cfg.critic_lr)
        critic_opt_state = self.critic_opt.init(critic_params)
        
        self.log_alpha_opt = optax.adam(cfg.alpha_lr)
        log_alpha_opt_state = self.critic_opt.init(log_alpha)
        
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
        
        self.reward_lambda = cfg.reward_lambda
        self.ema = cfg.ema
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_frequency = cfg.target_update_frequency
        self.target_entropy = cfg.target_entropy # TODO: define for SAC
        
    def act(self, obs, step, eval_mode=False):
        del step
        
        rng, key = jax.random.split(self.train_state.rng_key)
        
        encoder_params = self.train_state.encoder_params
        features = self.encoder.apply(encoder_params, jnp.expand_dims(obs, 0)).squeeze()

        actor_params = self.train_state.actor_params
        dist = self.actor.apply(actor_params, features)
        
        if eval_mode:
            action = dist.mean()
        else:
            action = dist.sample(seed=key)
        
        self.train_state = self.train_state._replace(
            rng_key=rng
        )
        
        return action
    
    def get_reward_aug(self, observations, actions):
        return self.reward_aug(observations, actions)
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def critic_loss(self, encoder_params, critic_params, critic_target_params, actor_params, log_alpha, transitions, key, step):
        del step
        
        # get reward aug
        reward_pen = self.get_reward_aug(transitions.obs, transitions.actions)
        transitions.rewards -= self.reward_lambda * jax.lax.stop_gradient(reward_pen) # don't want extra gradients going back to encoder params
        
        # encode observations
        features = self.encoder.apply(encoder_params, transitions.obs)
        next_features = self.encoder.apply(encoder_params, transitions.next_obs)
        
        # get the targets
        dist = self.actor.apply(actor_params, next_features)
        next_actions = dist.sample(seed=key)
        log_probs = dist.log_prob(next_actions).sum(-1, keepdims=True)
        
        nq1, nq2 = self.critic.apply(critic_target_params, features, next_actions)
        v = jnp.minimum(nq1, nq2) - jnp.exp(log_alpha) * log_probs
        target_q = jax.lax.stop_gradient(transitions.rewards + self.cfg.discount * (1.0 - transitions.dones) * v)

        # get the actual q values
        q1, q2 = self.critic.apply(critic_params, features, transitions.actions)
        
        q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
        return q_loss
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def actor_loss(self, actor_params, encoder_params, critic_params, log_alpha, transitions, key, step):
        del step
        
        # encode observations
        features = self.encoder.apply(encoder_params, transitions.obs)
        features = jax.lax.stop_gradient(features) # just so we don't have to deal with gradients passing through to encoder no matter what

        dist = self.actor.apply(actor_params, features)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions).sum(-1, keepdims=True)
        
        q1, q2 = self.critic.apply(critic_params, features, actions)
        min_q = jnp.minimum(q1, q2)
        actor_loss = jnp.exp(log_alpha) * log_probs - min_q
        return jnp.mean(actor_loss)
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def alpha_loss(self, log_alpha, encoder_params, actor_params, transitions, key, step):
        del step
        
        features = self.encoder.apply(encoder_params, transitions.obs)
        features = jax.lax.stop_gradient(features)
        
        dist = self.actor.apply(actor_params, features)
        actions = dist.sample(seed=key)
        log_prob = dist.log_prob(actions).sum(-1, keepdims=True)
        
        alpha_loss = jnp.exp(log_alpha) - jax.lax.stop_gradient(-log_prob - self.target_entropy)
        return jnp.mean(alpha_loss)
    
    def update_critic(self, encoder_params, critic_params, critic_target_params, actor_params, log_alpha, transitions, key, step):
        # assume that the transitions is a sequence of consecutive (s, a, r, s', d) tuples
        loss_grad_fn = jax.value_and_grad(self.critic_loss, argnums=(0, 1))
        (encoder_loss, c_loss), (encoder_grads, critic_grads) = loss_grad_fn(
            encoder_params,
            critic_params,
            critic_target_params,
            actor_params,
            log_alpha,
            transitions,
            key,
            step
        )
        
        # update both encoder and critic
        enc_update, enc_opt_state = self.encoder_opt.update(encoder_grads, self.train_state.encoder_opt_state)
        new_enc_params = optax.apply_updates(enc_update, self.train_state.encoder_params)
        
        critic_update, critic_opt_state = self.critic_opt.update(critic_grads, self.train_state.critic_opt_state)
        new_critic_params = optax.apply_updates(critic_update, self.train_state.critic_params)
        
        metrics = {
            'critic_loss': c_loss.item(),
            'encoder_loss': encoder_loss.item()
        }
        
        new_vars = {
            'encoder_params': new_enc_params,
            'encoder_opt_state': enc_opt_state,
            'critic_params': new_critic_params,
            'critic_opt_state': critic_opt_state
        }
        return metrics, new_vars
    
    def update_actor(self, actor_params, encoder_params, critic_params, log_alpha, transitions, key, step):
        loss_grad_fn = jax.value_and_grad(self.actor_loss)
        a_loss, a_grads = loss_grad_fn(actor_params, encoder_params, critic_params, log_alpha, transitions, key, step)
        
        # update actor params
        actor_update, actor_opt_state = self.actor_opt.update(a_grads, self.train_state.actor_opt_state)
        new_actor_params = optax.apply_updates(actor_update, self.train_state.actor_params)
        
        metrics = {
            'actor_loss': a_loss.item()
        }
        
        new_vars = {
            'actor_params': new_actor_params,
            'actor_opt_state': actor_opt_state
        }
        return metrics, new_vars
    
    def update_log_alpha(self, log_alpha, encoder_params, actor_params, transitions, key, step):
        loss_grad_fn = jax.value_and_grad(self.alpha_loss)
        alpha_loss, alpha_grads = loss_grad_fn(log_alpha, encoder_params, actor_params, transitions, key, step)
        
        # update alpha
        alpha_update, new_log_alpha_opt_state = self.log_alpha_opt.update(alpha_grads, self.train_state.log_alpha_opt_state)
        new_log_alpha_params = optax.apply_updates(alpha_update, self.train_state.log_alpha_params)
        
        metrics = {
            'alpha_loss': alpha_loss.item()
        }
        
        new_vars = {
            'log_alpha_params': new_log_alpha_params,
            'log_alpha_opt_state': new_log_alpha_opt_state
        }
        return metrics, new_vars
    
    def update(self, transitions, step):
        key1, key2, key3 = jax.random.split(self.train_state.rng_key, 3)
        
        # critic update
        critic_metrics, critic_new_vars = self.update_critic(
            self.train_state.encoder_params,
            self.train_state.critic_params,
            self.train_state.critic_target_params,
            self.train_state.actor_params,
            self.train_state.log_alpha_params,
            transitions,
            key1,
            step
        )
        
        upd_train_state = self.train_state._replace(
            encoder_params=critic_new_vars['encoder_params'],
            critic_params=critic_new_vars['critic_params'],
            encoder_opt_state=critic_new_vars['encoder_opt_state'],
            critic_opt_state=critic_new_vars['critic_opt_state']
        )
        
        # actor update
        actor_metrics, actor_new_vars = self.update_actor(
            upd_train_state.actor_params,
            upd_train_state.encoder_params,
            upd_train_state.critic_params,
            upd_train_state.log_alpha_params,
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
            self.ema
        )
        
        self.train_state = upd_train_state._replace(
            critic_target_params=new_target_params,
            rng_key=key3
        )
        
        # logging
        metrics = {**critic_metrics, **actor_metrics}
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
            print('cannot load SAC')
            return None