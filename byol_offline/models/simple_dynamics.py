import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple
import dill

from memory.replay_buffer import Transition

class MLPDynamicsModel(hk.Module):
    '''Simple MLP dynamics for state-based envs.'''
    def __init__(self, state_dim, hidden_size, act='relu'):
        super().__init__(name='simple_mlp_dynamics')
        self._state_dim = state_dim
        self._hidden_size = hidden_size
        self._activation = jax.nn.relu if act == 'relu' else jnp.tanh
    
    def __call__(self, obs, action):
        obs_action = jnp.concatenate([obs, action], -1)
        x = hk.nets.MLP(
            [self._hidden_size, self._hidden_size, self._state_dim],
            activation=self._activation
        )(obs_action)
        
        return x

class SimpleTrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    
class SimpleDynamicsTrainer:
    '''Dynamics trainer similar to MILO.'''
    def __init__(self, cfg):
        self.cfg = cfg
        
        if cfg.model_type == 'mlp_dynamics':
            model_fn = lambda s, a: MLPDynamicsModel(cfg.obs_shape[0], cfg.hidden_dim, act='relu')(s, a)
        else:
            raise NotImplementedError("haven't implemented more complicated models yet. Starting simple :)")

        model = hk.without_apply_rng(hk.transform(model_fn))
        
        # initialization
        key = jax.random.PRNGKey(cfg.seed)
        params = model.init(key, jnp.zeros((1,) + tuple(cfg.obs_shape)), jnp.zeros((1,) + tuple(cfg.action_shape)))
        
        opt = optax.sgd(cfg.lr, momentum=0.9, nesterov=True) if cfg.optim == 'sgd' else optax.adam(cfg.lr)
        opt_state = opt.init(params)
        
        self.train_state = SimpleTrainState(
            params=params,
            opt_state=opt_state
        )
        
        self.train_for_diff = cfg.train_for_diff
        
        @jax.jit
        def loss_fn(params: hk.Params,
                    transitions: Transition):
            outputs = model.apply(params, transitions.obs, transitions.actions)
            
            def state_diff():
                return transitions.next_obs - transitions.obs
            
            targets = jax.lax.cond(self.train_for_diff,
                                   lambda _: state_diff(),
                                   lambda _: transitions.next_obs,
                                   operand=None)
            
            loss = jnp.mean(jnp.square(outputs - targets))
            return loss
        
        def update(train_state: SimpleTrainState,
                   transitions: Transition,
                   step: int):
            del step
            
            loss_grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = loss_grad_fn(train_state.params, transitions)
            
            metrics = {
                'dynamics_loss': loss
            }
            
            update, new_opt_state = opt.update(grads, train_state.opt_state)
            new_params = optax.apply_updates(train_state.params, update)
            
            new_train_state = SimpleTrainState(
                params=new_params,
                opt_state=new_opt_state
            )
            
            return new_train_state, metrics

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
            print('cannot load simple dynamics model')
            exit()
        
        