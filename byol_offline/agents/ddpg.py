import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple
import dill

from byol_offline.networks.encoder import DrQv2Encoder
from byol_offline.networks.actor_critic import *
from byol_offline.reward_augs.byol import *
from byol_offline.reward_augs.rnd import *
from byol_offline.agents.agent_utils import *

class DDPGTrainState(NamedTuple):
    encoder_params: hk.Params
    actor_params: hk.Params
    critic_params: hk.Params
    reward_aug_params: hk.Params
    critic_target_params: hk.Params
    
    encoder_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    reward_aug_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey # updated every time we act or learn

class DDPG:
    '''DDPG agent from DrQv2 (without the data augmentations).'''
    def __init__(self, cfg, wm=None):
        self.cfg = cfg
        # print(f'config stuff: {tuple(cfg.obs_shape), tuple(cfg.action_shape), cfg.feature_dim, cfg.hidden_dim, cfg.seed}')
        
        # encoder
        encoder_fn = lambda obs: DrQv2Encoder(cfg.obs_shape)(obs)
        self.encoder = hk.without_apply_rng(hk.transform(encoder_fn))
        
        # actor + critic
        actor_fn = lambda obs, std: Actor(cfg.action_shape, cfg.feature_dim, cfg.hidden_dim)(obs, std)
        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        critic_fn = lambda obs, action: Critic(cfg.feature_dim, cfg.hidden_dim)(obs, action)
        self.critic = hk.without_apply_rng(hk.transform(critic_fn))
        
        # reward pessimism
        if wm is None or cfg.reward_aug == 'rnd':
            reward_aug_fn = lambda obs: RNDPredictor(cfg.rnd)(obs)
        else:
            raise NotImplementedError('have not implemented BYOL-Explore reward penalty')
        
        self.reward_aug = hk.without_apply_rng(hk.transform(reward_aug_fn))
        self.target_reward_aug = hk.without_apply_rng(hk.transform(reward_aug_fn))
        self.reward_rms = RMS()
        
        # initialization
        rng = jax.random.PRNGKey(cfg.seed)
        key1, key2, key3, key4, key5, key6 = jax.random.split(rng, 6)
        
        encoder_params = self.encoder.init(key1, jnp.zeros((1,) + tuple(cfg.obs_shape)))
        actor_params = self.actor.init(key2, jnp.zeros((1, 20000)), jnp.zeros(1))
        critic_params = critic_target_params = self.critic.init(key3, jnp.zeros((1, 20000)), jnp.zeros((1,) + tuple(cfg.action_shape)))
        reward_aug_params = self.reward_aug.init(key4, jnp.zeros((1, 20000))) # 20000 is repr_dim of the DrQv2 encoder, but can't access it in hk.transform function call
        self.target_reward_aug_params = self.target_reward_aug.init(key5, jnp.zeros((1, 20000))) # stays fixed!
        
        # optimizers
        self.encoder_opt = optax.adam(cfg.encoder_lr)
        encoder_opt_state = self.encoder_opt.init(encoder_params)
        
        self.actor_opt = optax.adam(cfg.actor_lr)
        actor_opt_state = self.actor_opt.init(actor_params)
        
        self.critic_opt = optax.adam(cfg.critic_lr)
        critic_opt_state = self.critic_opt.init(critic_params)
        
        self.reward_aug_opt = optax.adam(cfg.reward_aug_lr)
        reward_aug_opt_state = self.reward_aug_opt.init(reward_aug_params)
        
        # train state
        self.train_state = DDPGTrainState(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            reward_aug_params=reward_aug_params,
            critic_target_params=critic_target_params,
            encoder_opt_state=encoder_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            reward_aug_opt_state=reward_aug_opt_state,
            rng_key=key6
        )
        
        # hyperparameters for training
        self.rnd_scale = cfg.rnd_scale
        self.reward_lambda = cfg.reward_lambda
        self.ema = cfg.ema
        self.init_std = cfg.init_std
        self.final_std = cfg.final_std
        self.std_duration = cfg.std_duration
        self.std_clip_val = cfg.std_clip_val
        self.update_every_steps = cfg.update_every_steps
        
    def stddev_schedule(self, step):
        mix = jnp.clip(step / self.std_duration, 0.0, 1.0)
        return (1.0 - mix) * self.init_std + mix * self.final_std
        
    def act(self, obs, step, eval_mode=False):
        rng, key = jax.random.split(self.train_state.rng_key)
        
        encoder_params = self.train_state.encoder_params
        features = self.encoder.apply(encoder_params, jnp.expand_dims(obs, 0)).squeeze()
        
        std = self.stddev_schedule(step)
        actor_params = self.train_state.actor_params
        dist = self.actor.apply(actor_params, features, std)
        
        if eval_mode:
            action = dist.mean()
        else:
            action = dist.sample(seed=key)
        
        # change RNG key while training
        self.train_state = DDPGTrainState(
            encoder_params=self.train_state.encoder_params,
            actor_params=self.train_state.actor_params,
            critic_params=self.train_state.critic_params,
            reward_aug_params=self.train_state.reward_aug_params,
            critic_target_params=self.train_state.critic_target_params,
            encoder_opt_state=self.train_state.encoder_opt_state,
            actor_opt_state=self.train_state.actor_opt_state,
            critic_opt_state=self.train_state.critic_opt_state,
            reward_aug_opt_state=self.train_state.reward_aug_opt_state,
            rng_key=rng
        )
        
        return action
    
    def get_reward_aug(self, reward_aug_params, encoder_params, obs):
        if self.cfg.reward_aug == 'rnd':
            reprs = self.encoder.apply(encoder_params, obs)
            prediction_error = self.reward_aug.apply(reward_aug_params, reprs)
            _, error_var = self.reward_rms(prediction_error)
            reward_pen = (prediction_error / (jnp.sqrt(error_var) + 1e-8)).squeeze()
        else:
            raise NotImplementedError('Need to implement WM to get BYOL prediction error.')
        
        return reward_pen
    
    def update_critic(self, encoder_params, critic_params, critic_target_params, actor_params, reward_aug_params, transitions, key, step):
        # define loss here bc jax dislikes the object oriented loss functions
        @jax.jit
        def critic_loss(encoder_params, critic_params, critic_target_params, actor_params, reward_aug_params, transitions, key, step):
            # get reward aug
            reward_pen = self.get_reward_aug(reward_aug_params, encoder_params, transitions.obs)
            transitions.rewards -= self.reward_lambda * jax.lax.stop_gradient(reward_pen) # don't want extra gradients going back to encoder params
            
            # encode observations
            features = self.encoder.apply(encoder_params, transitions.obs)
            next_features = self.encoder.apply(encoder_params, transitions.next_obs)
            
            # get the targets
            std = self.stddev_schedule(step)
            dist = self.actor.apply(actor_params, next_features, std)
            next_actions = dist.sample(seed=key)
            
            nq1, nq2 = self.critic.apply(critic_target_params, features, next_actions)
            v = jnp.minimum(nq1, nq2)
            target_q = jax.lax.stop_gradient(transitions.rewards + self.cfg.discount * (1.0 - transitions.dones) * v)

            # get the actual q values
            q1, q2 = self.critic.apply(critic_params, features, transitions.actions)
            
            q_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
            return q_loss
        
        loss_grad_fn = jax.value_and_grad(critic_loss, argnums=(0, 1))
        (encoder_loss, c_loss), (encoder_grads, critic_grads) = loss_grad_fn(
            encoder_params,
            critic_params,
            critic_target_params,
            actor_params,
            reward_aug_params,
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
            'encoder_critic_update_loss': encoder_loss.item()
        }
        
        new_vars = {
            'encoder_params': new_enc_params,
            'encoder_opt_state': enc_opt_state,
            'critic_params': new_critic_params,
            'critic_opt_state': critic_opt_state
        }
        
        return metrics, new_vars
    
    def update_actor(self, actor_params, encoder_params, critic_params, transitions, key, step):
        # define actor loss here
        @jax.jit
        def actor_loss(encoder_params, actor_params, critic_params, transitions, key, step):
            # encode observations
            features = self.encoder.apply(encoder_params, transitions.obs)

            std = self.stddev_schedule(step)
            dist = self.actor.apply(actor_params, features, std)
            actions = dist.sample(seed=key)
            
            q1, q2 = self.critic.apply(critic_params, features, actions)
            actor_loss = -jnp.mean(jnp.minimum(q1, q2))
            return actor_loss

        loss_grad_fn = jax.value_and_grad(actor_loss)
        a_loss, actor_grads = loss_grad_fn(encoder_params, actor_params, critic_params, transitions, key, step)
        
        # update both encoder and critic
        actor_update, actor_opt_state = self.critic_opt.update(actor_grads, self.train_state.actor_opt_state)
        new_actor_params = optax.apply_updates(actor_update, self.train_state.actor_params)
        
        metrics = {
            'actor_loss': a_loss.item()
        }
        
        new_vars = {
            'actor_params': new_actor_params,
            'actor_opt_state': actor_opt_state
        }
        return metrics, new_vars
    
    def update_reward_aug(self, reward_aug_params, encoder_params, transitions):
        # define reward loss fn here
        @jax.jit
        def reward_loss(reward_params, encoder_params, obs):
            reprs = self.encoder.apply(encoder_params, obs)
            online_preds = self.reward_aug.apply(reward_params, reprs)
            target_preds = self.target_reward_aug.apply(self.target_reward_aug_params, reprs)
            
            prediction_error = jnp.mean(jnp.square(online_preds - target_preds), axis=-1, keepdims=True) # (B, 1)
            return prediction_error
        
        reward_loss_grad_fn = jax.value_and_grad(reward_loss, argnums=(0, 1))
        (r_loss, encoder_loss), (reward_grads, encoder_grads) = reward_loss_grad_fn(reward_aug_params, encoder_params, transitions.obs)
        
        update, reward_aug_opt_state = self.reward_aug_opt.update(reward_grads, self.train_state.reward_aug_opt_state)
        new_reward_aug_params = optax.apply_updates(update, reward_aug_params)
        
        enc_update, new_enc_opt_state = self.encoder_opt.update(encoder_grads, self.train_state.encoder_opt_state)
        new_enc_params = optax.apply_updates(enc_update, encoder_params)
        
        metrics = {
            'reward_loss': r_loss.item(),
            'rnd_encoder_loss': encoder_loss.item()
        }
        
        new_vars = {
            'reward_aug_params': new_reward_aug_params,
            'reward_aug_opt_state': reward_aug_opt_state,
            'encoder_params': new_enc_params,
            'encoder_opt_state': new_enc_opt_state
        }
        
        return metrics, new_vars
    
    def update_target(self, params, target_params):
        target_params = jax.tree_util.tree_map(lambda x, y: self.ema * x + (1.0 - self.ema) * y, params, target_params)
        return target_params
    
    def update(self, transitions, step):
        # TODO keep in mind that the RND reward is using a different encoder--may have to switch that up
        key1, key2, key3 = jax.random.split(self.train_state.rng_key, 3)
        
        if step % self.update_every_steps == 0:
            return dict()
        
        # reward aug update
        reward_metrics, reward_new_vars = self.update_reward_aug(
            self.train_state.reward_aug_params,
            self.train_state.encoder_params,
            transitions.obs
        )
        
        # update to new intermediate train state with new params
        new_reward_aug_params = reward_new_vars['reward_aug_params']
        new_reward_aug_opt_state = reward_new_vars['reward_aug_opt_state']
        new_encoder_params = reward_new_vars['encoder_params']
        new_encoder_opt_state = reward_new_vars['encoder_opt_state']
        
        upd_train_state = DDPGTrainState(
            encoder_params=new_encoder_params,
            actor_params=self.train_state.actor_params,
            critic_params=self.train_state.critic_params,
            reward_aug_params=new_reward_aug_params,
            critic_target_params=self.train_state.critic_target_params,
            encoder_opt_state=new_encoder_opt_state,
            actor_opt_state=self.train_state.actor_opt_state,
            critic_opt_state=self.train_state.critic_opt_state,
            reward_aug_opt_state=new_reward_aug_opt_state,
            rng_key=self.train_state.rng_key # wasn't used in updating the RND network params
        )
        
        # critic update
        critic_metrics, critic_new_vars = self.update_critic(
            upd_train_state.encoder_params,
            upd_train_state.critic_params,
            upd_train_state.critic_target_params,
            upd_train_state.actor_params,
            transitions,
            key1,
            step
        )
        
        # update encoder + critic params & opt states
        upd_train_state = DDPGTrainState(
            encoder_params=critic_new_vars['encoder_params'],
            actor_params=upd_train_state.actor_params,
            critic_params=critic_new_vars['critic_params'],
            reward_aug_params=upd_train_state.reward_aug_params,
            critic_target_params=upd_train_state.critic_target_params,
            encoder_opt_state=critic_new_vars['encoder_opt_state'],
            actor_opt_state=upd_train_state.actor_opt_state,
            critic_opt_state=critic_new_vars['critic_opt_state'],
            reward_aug_opt_state=upd_train_state.reward_aug_opt_state,
            rng_key=upd_train_state.rng_key # only update RNG key at the end
        )
        
        # update actor
        actor_metrics, actor_new_vars = self.update_actor(
            upd_train_state.actor_params,
            upd_train_state.encoder_params,
            upd_train_state.critic_params,
            transitions,
            key2,
            step
        )
        
        upd_train_state = DDPGTrainState(
            encoder_params=upd_train_state.encoder_params,
            actor_params=actor_new_vars['actor_params'],
            critic_params=upd_train_state.critic_params,
            reward_aug_params=upd_train_state.reward_aug_params,
            critic_target_params=upd_train_state.critic_target_params,
            encoder_opt_state=upd_train_state.encoder_opt_state,
            actor_opt_state=actor_new_vars['actor_opt_state'],
            critic_opt_state=upd_train_state.critic_opt_state,
            reward_aug_opt_state=upd_train_state.reward_aug_opt_state,
            rng_key=upd_train_state.rng_key # only update RNG key at the end
        )
        
        # update critic target
        new_target_params = self.update_target(
            self.train_state.critic_params,
            self.train_state.critic_target_params
        )
        
        self.train_state = DDPGTrainState(
            encoder_params=upd_train_state.encoder_params,
            actor_params=upd_train_state.actor_params,
            critic_params=upd_train_state.critic_params,
            reward_aug_params=upd_train_state.reward_aug_params,
            critic_target_params=new_target_params,
            encoder_opt_state=upd_train_state.encoder_opt_state,
            actor_opt_state=upd_train_state.actor_opt_state,
            critic_opt_state=upd_train_state.critic_opt_state,
            reward_aug_opt_state=upd_train_state.reward_aug_opt_state,
            rng_key=key3
        )
        
        # logging
        metrics = {**reward_metrics, **critic_metrics, **actor_metrics}
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
            print('cannot load agent')
            return None
        