import jax
import jax.numpy as jnp
import haiku as hk
import optax
import distrax
from typing import NamedTuple
import dill

from utils import MUJOCO_ENVS, batched_zeros_like

class AETrainState(NamedTuple):
    ae_params: hk.Params
    ae_opt_state: optax.OptState
    rng_key: jax.random.PRNGKey
    
class DeterVAEOutput(NamedTuple):
    latent_dist: distrax.Distribution
    rec_output: jnp.ndarray
    
class CondAE(hk.Module):
    '''Conditional AE from https://github.com/shidilrzf/Anti-exploration-RL/blob/1013a85b4b84656a06f86abee01c55a5e08272ee/rlkit/torch/networks.py#L215'''
    def __init__(self, cfg):
        super().__init__()
        
        assert len(cfg.obs_shape) == 1, 'this CondAE only supports mujoco based envs'
        self._state_embed_dim = cfg.state_embed_dim
        self._action_embed_dim = cfg.action_embed_dim
        
        self._encoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.feature_dim],
            activation=jax.nn.relu,
            activate_final=False
        )
        self._decoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.action_shape[0]],
            activation=jax.nn.relu,
            activate_final=False
        )
        
    def __call__(self, s, a):
        # encode
        zs = hk.Linear(self._state_embed_dim)(s)
        za = hk.Linear(self._action_embed_dim)(a)
        z = jnp.concatenate([zs, za], axis=-1)
        z = self._encoder(z)
        
        # decode
        zs = jnp.concatenate([z, s], axis=-1)
        ahat = self._decoder(zs)
        return ahat
    
class VAE(hk.Module):
    '''VAE from the Offline RL as Anti-Exploration paper (https://arxiv.org/pdf/2106.06431.pdf)'''
    def __init__(self, cfg):
        super().__init__()
        assert len(cfg.obs_shape) == 1, 'this VAE only supports mujoco based envs'
        self._feature_dim = cfg.feature_dim
        
        self._encoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim],
            activation=jax.nn.relu,
            activate_final=False
        )
        self._decoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.action_shape[0]],
            activation=jax.nn.relu,
            activate_final=False
        )
        self._activate_final = cfg.activate_final
        self._clip_log_std = cfg.clip_log_std
        
    def __call__(self, s, a):
        # encode (embed first apparently)
        sa = jnp.concatenate([s, a], axis=-1)
        sa_rep = self._encoder(sa)
        mean = hk.Linear(self._feature_dim)(sa_rep)
        logstd = hk.Linear(self._feature_dim)(sa_rep)
        if self._clip_log_std:
            logstd = jnp.clip(logstd, -4, 15)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(logstd))
        
        # decode
        z = dist.sample(seed=hk.next_rng_key()) # sampling already does reparameterization trick in distrax
        sz = jnp.concatenate([s, z], axis=-1) # CVAE so condition on s
        ahat = self._decoder(sz)
        if self._activate_final:
            ahat = jnp.tanh(ahat)
        
        return DeterVAEOutput(latent_dist=dist, rec_output=ahat)
        
class AETrainer:
    def __init__(self, cfg):
        assert cfg.task in MUJOCO_ENVS, "VAE doesn't support DMC or Atari envs just yet -- prob best to go RND there."
        
        # set up net
        if cfg.type == 'vae':
            ae_fn = lambda s, a: VAE(cfg)(s, a)
        elif cfg.type == 'cond_ae':
            ae_fn = lambda s, a: CondAE(cfg)(s, a)
        else:
            raise NotImplementedError('No other AE methods implemented here.')
        ae = hk.transform(ae_fn, apply_rng=True) # need for sampling for VAE, not for AE
        
        key = jax.random.PRNGKey(cfg.seed)
        init_key, state_key = jax.random.split(key)
        params = ae.init(init_key, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
        
        opt = optax.adam(learning_rate=cfg.lr)
        opt_state = opt.init(params)
        
        self.train_state = AETrainState(
            ae_params=params,
            ae_opt_state=opt_state,
            rng_key=state_key
        )
        
        # hparams
        feature_dim = cfg.feature_dim
        beta = cfg.beta
        
        # define loss functions + uncertainty bonus
        @jax.jit
        def loss_fn(params, key, obs, actions):
            output = ae.apply(params, key, obs, actions)
            
            if cfg.type == 'cond_ae':
                loss = jnp.mean(jnp.square(output - actions))
                extras = None
            else:
                post_dist = output.latent_dist
                rec_output = output.rec_output
                rec_loss = jnp.mean(jnp.square(rec_output - actions))
                
                standard_gaussian = distrax.MultivariateNormalDiag(
                    jnp.zeros((feature_dim,)),
                    jnp.ones((feature_dim,))
                )
                kl = post_dist.kl_divergence(standard_gaussian)
                kl = jnp.mean(kl)
                loss = rec_loss + beta * kl # want to minimize KL and reconstruction loss
                extras = {'rec_loss': rec_loss, 'kl': kl}
            
            return loss, extras
        
        def update(train_state: AETrainState, obs: jnp.ndarray, actions: jnp.ndarray, step: int):
            del step
            update_key, state_key = jax.random.split(train_state.rng_key)
            
            loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, extras), grads = loss_grad_fn(train_state.ae_params, update_key, obs, actions)
            
            update, new_opt_state = opt.update(grads, train_state.ae_opt_state)
            new_params = optax.apply_updates(train_state.ae_params, update)
            
            new_state = AETrainState(
                ae_params=new_params,
                ae_opt_state=new_opt_state,
                rng_key=state_key
            )
            metrics = {'loss': loss}
            if cfg.type == 'vae':
                metrics.update(extras)
                
            return new_state, metrics
        
        def compute_uncertainty(obs, actions):
            if cfg.type == 'vae':
                output = ae.apply(self.train_state.ae_params, self.train_state.rng_key, obs, actions)
                ahat = output.rec_output
                uncertainty = jnp.mean(jnp.square(ahat - actions), axis=-1)
            else:
                ahat = ae.apply(self.train_state.ae_params, self.train_state.rng_key, obs, actions)
                uncertainty = jnp.mean(jnp.square(ahat - actions), axis=-1)
                
            return uncertainty
        
        self._update = jax.jit(update)
        self._compute_uncertainty = jax.jit(compute_uncertainty)
        
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            dill.dump(self.train_state, f, protocol=2)
        
    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                train_state = dill.load(f)
                self.train_state = train_state
        except FileNotFoundError:
            print('cannot load AE model')
            exit()