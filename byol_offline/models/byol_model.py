import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import distrax
import dill
from typing import Tuple, NamedTuple, Dict, Union

from byol_offline.networks.encoder import DrQv2Encoder, DreamerEncoder
from byol_offline.networks.decoder import DrQv2Decoder, DreamerDecoder
from byol_offline.networks.rnn import *
from byol_offline.networks.predictors import BYOLPredictor
from byol_offline.models.byol_utils import *
from utils import MUJOCO_ENVS, seq_batched_zeros_like

class BYOLTrainState(NamedTuple):
    wm_params: hk.Params
    target_params: hk.Params
    wm_opt_state: optax.OptState
    rng_key: jax.random.PRNGKey

class ConvWorldModel(hk.Module):
    '''World model for DMC tasks. Primarily inspired by DreamerV2 and DrQv2 repositories.'''
    def __init__(self, cfg):
        super().__init__()
        
        # nets
        if cfg.dreamer:
            self._encoder = DreamerEncoder(cfg.depth)
        else:
            self._encoder = DrQv2Encoder()
        
        # rssm
        self._rssm = RSSM(cfg) # uses both closed gru and open gru here, closed for dreamer loss, open for byol loss
        
        # prediction heads
        if cfg.dreamer:
            self._decoder = DreamerDecoder(cfg.obs_shape[-1], cfg.depth)
            self._byol_predictor = BYOLPredictor(1024)
        else:
            self._decoder = DrQv2Decoder(cfg.obs_shape[-1])
            self._byol_predictor = BYOLPredictor(20000)
        
        self._reward_predictor = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
            w_init=glorot_w_init
        )
        
    def _open_gru_rollout(self, actions, state):
        '''
        Rolls out the open GRU (in this case, the RSSM recurrent unit) over all actions, starting at init state 'state'.
        
        Only gets the deterministic part of the output, similar to BYOL-Explore.
        '''
        def _scan_fn(carry, act):
            state = carry
            new_state, _ = self._rssm._onestep_prior(act, state)
            return new_state, new_state[-2] # new feature, deter state
        
        _, deter_states = hk.scan(_scan_fn, state, actions)
        return deter_states
    
    # ===== eval functions (imagining/observing one step, many steps, etc...) =====
    
    def _onestep_imagine(self, action, state):
        new_state, _ = self._rssm._onestep_prior(action, state)
        img_mean = self._decoder(new_state)
        reward_mean = self._reward_predictor(new_state)
        
        return img_mean, reward_mean
    
    def _onestep_observe(self, obs, action, state):
        emb = self._encoder(obs)
        new_state, _ = self._rssm._onestep_post(emb, action, state)
        img_mean = self._decoder(new_state)
        reward_mean = self._reward_predictor(new_state)
        
        return img_mean, reward_mean
    
    # ===== things used in training/to define loss function in trainer =====
    
    def _dreamer_forward(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        embeds = hk.BatchApply(self._encoder)(obs)
        
        # always start with init state at zeros
        posts, priors, post_features, prior_features = self._rssm(embeds, actions, None)
        # posts: (T, B, dist_dim), priors: (T, B, dist_dim), features: (T, B, feature_dim)
        post_img_mean = hk.BatchApply(self._decoder)(post_features) # (T, B, H, W, C)
        post_reward_mean = hk.BatchApply(self._reward_predictor)(post_features)
        post_reward_mean = jnp.squeeze(post_reward_mean)
        
        prior_img_mean = hk.BatchApply(self._decoder)(prior_features)
        prior_reward_mean = hk.BatchApply(self._reward_predictor)(prior_features)
        prior_reward_mean = jnp.squeeze(prior_reward_mean)
        
        return post_img_mean, post_reward_mean, prior_img_mean, prior_reward_mean, posts, priors
    
    def _byol_forward(self, obs, actions):
        # first get embeddings
        B = obs.shape[1]
        embeds = hk.BatchApply(self._encoder)(obs)
        init_state = self._rssm._init_state(B)
        
        embed, action = embeds[0], actions[0] # x_0, a_0
        first_state, _ = self._rssm._onestep_post(embed, action, init_state) # h_0
        latent = self._byol_predictor(first_state[-2])
        latent = jnp.expand_dims(latent, 0)
        
        # === collect h_t, latents (byol stuff) ===
        deter_states = self._open_gru_rollout(actions[1:], first_state)
        latents = self._byol_predictor(deter_states)
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeds
        
        # return
        return pred_latents, embeds
        
    def __call__(self, obs, actions):
        dreamer_out = self._dreamer_forward(obs, actions)
        byol_out = self._byol_forward(obs, actions)
        
        return dreamer_out, byol_out

class MLPWorldModel(hk.Module):
    '''World model for D4RL tasks. Primarily inspired by MOPO repository.'''
    def __init__(self, cfg):
        super().__init__()
        
        # nets
        self._encoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.repr_dim],
            activation=jax.nn.swish
        )
        
        # rssm
        self._rssm = RSSM(cfg) # uses both closed gru and open gru here, closed for dreamer loss, open for byol loss
        
        # predictors
        self._decoder = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.obs_shape[0]],
            activation=jax.nn.swish
        )
        self._reward_predictor = hk.nets.MLP(
            [cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
            w_init=glorot_w_init
        )
        self._byol_predictor = BYOLPredictor(cfg.repr_dim)
        
    # ===== eval functions (imagining/observing one step, many steps, etc...) =====
    
    def _onestep_imagine(self, action, state):
        new_state, _ = self._rssm._onestep_prior(action, state)
        state_mean = self._decoder(new_state)
        reward_mean = self._reward_predictor(new_state)
        
        return state_mean, reward_mean
    
    def _onestep_observe(self, obs, action, state):
        emb = self._encoder(obs)
        new_state, _ = self._rssm._onestep_post(emb, action, state)
        state_mean = self._decoder(new_state)
        reward_mean = self._reward_predictor(new_state)
        
        return state_mean, reward_mean
    
    # ===== things used in training/to define loss function in trainer =====
    
    def _dreamer_forward(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        
        embeds = hk.BatchApply(self._encoder)(obs)
        posts, priors, post_features, prior_features = self._rssm(embeds, actions, None)
        
        # state + reward prediction
        post_state_mean = hk.BatchApply(self._decoder)(post_features) # (T, B, H, W, C)
        post_reward_mean = hk.BatchApply(self._reward_predictor)(post_features) # (T, B, 1)
        post_reward_mean = jnp.squeeze(post_reward_mean) # (T, B), so as to be fine with the rewards coming from dataset
        
        prior_state_mean = hk.BatchApply(self._decoder)(prior_features)
        prior_reward_mean = hk.BatchApply(self._reward_predictor)(prior_features)
        prior_reward_mean = jnp.squeeze(prior_reward_mean)
        
        return post_state_mean, post_reward_mean, prior_state_mean, prior_reward_mean, posts, priors
    
    def _byol_forward(self, obs, actions):
        # obs should be of shape (T, B, H, W, C), actions of shape (T, B, action_dim)
        
        # first get embeddings
        B = obs.shape[1]
        embeds = hk.BatchApply(self._encoder)(obs)
        init_state = self._rssm._init_state(B)
        
        embed, action = embeds[0], actions[0] # x_0, a_0
        first_state, _ = self._rssm._onestep_post(embed, action, init_state) # h_0
        latent = self._byol_predictor(first_state[-2])
        latent = jnp.expand_dims(latent, 0)
        
        # === collect h_t, latents (byol stuff) ===
        deter_states = self._open_gru_rollout(actions[1:], first_state)
        latents = hk.BatchApply(self._byol_predictor)(deter_states)
        pred_latents = jnp.concatenate([latent, latents]) # (T, B, embed_dim) for both pred_latents and embeds
        
        # return
        return pred_latents, embeds
    
    def _open_gru_rollout(self, actions, state):
        '''
        Rolls out the open GRU (in this case, the RSSM recurrent unit) over all actions, starting at init state 'state'.
        
        Only gets the deterministic part of the output, similar to BYOL-Explore.
        '''
        def _scan_fn(carry, act):
            state = carry
            new_state, _ = self._rssm._onestep_prior(act, state)
            return new_state, new_state[-2]
        
        _, deter_states = hk.scan(_scan_fn, state, actions)
        return deter_states
    
    def __call__(self, obs, actions):
        dreamer_out = self._dreamer_forward(obs, actions)
        byol_out = self._byol_forward(obs, actions)
        
        return dreamer_out, byol_out
    
class WorldModelTrainer:
    '''World model trainer.'''
    def __init__(self, cfg):
        # set up
        if cfg.task in MUJOCO_ENVS:
            def wm_fn():
                wm = MLPWorldModel(cfg.d4rl)
                
                def init(o, a):
                    # same as standard forward pass
                    return wm(o, a)
                
                def dreamer_forward(o, a):
                    return wm._dreamer_forward(o, a)
                
                def byol_forward(o, a):
                    return wm._byol_forward(o, a)
                
                def imagine_fn(a, s):
                    return wm._onestep_imagine(a, s)
                
                return init, (dreamer_forward, byol_forward, imagine_fn)
        else:
            def wm_fn():
                wm = ConvWorldModel(cfg.vd4rl)
                
                def init(o, a):
                    # same as standard forward pass
                    return wm(o, a)
                
                def dreamer_forward(o, a):
                    return wm._dreamer_forward(o, a)
                
                def byol_forward(o, a):
                    return wm._byol_forward(o, a)
                
                def imagine_fn(a, s):
                    return wm._onestep_imagine(a, s)
                
                return init, (dreamer_forward, byol_forward, imagine_fn)
        
        # should be ok with byol loss here, as byol loss is regardless deterministic and only depends on deter states
        # rngs are used in the dreamer section of the model for generation, so cannot use without_apply_rng
        wm = hk.multi_transform(wm_fn)
        
        # optimizer
        if cfg.optim == 'adam':
            wm_opt = optax.adam(cfg.lr)
        elif cfg.optim == 'adamw':
            wm_opt = optax.adamw(cfg.lr)
        else:
            wm_opt = optax.sgd(cfg.lr, momentum=0.9)
        
        # params
        key = jax.random.PRNGKey(cfg.seed)
        
        def make_initial_state(k):
            init_key, target_key, state_key = jax.random.split(k, 3)
            wm_params = wm.init(init_key, seq_batched_zeros_like(cfg.obs_shape), seq_batched_zeros_like(cfg.action_shape))
            target_params = wm.init(target_key, seq_batched_zeros_like(cfg.obs_shape), seq_batched_zeros_like(cfg.action_shape))
            
            wm_opt_state = wm_opt.init(wm_params)
            
            train_state = BYOLTrainState(
                wm_params=wm_params,
                target_params=target_params,
                wm_opt_state=wm_opt_state,
                rng_key=state_key
            )
            return train_state

        # similar to make initial state for BYOL: https://github.com/deepmind/deepmind-research/blob/master/byol/byol_experiment.py#L424
        if cfg.pmap:
            init_wm = jax.pmap(make_initial_state, axis_name='i')
        else:
            init_wm = make_initial_state
        
        self.train_state = init_wm(key)
        
        # hparams and fns to note
        dreamer_forward, byol_forward, imagine_onestep = wm.apply
        
        ema = cfg.ema
        discrete = cfg.stoch_discrete_dim > 1
        beta = cfg.beta # trades off Dreamer ELBO loss and BYOL loss
        
        def _get_latent_dist(stats: jnp.ndarray, sg: bool = False) -> Union[distrax.Distribution, distrax.DistributionLike]:
            if sg:
                stats = jax.lax.stop_gradient(stats)
            
            if discrete:
                stats = stats.reshape(stats.shape[:-1] + (cfg.stoch_dim, cfg.stoch_discrete_dim))
                dist = distrax.straight_through_wrapper(distrax.OneHotCategorical)(logits=stats)
            else:
                mean, std = jnp.split(stats, 2, -1)
                std = jax.nn.softplus(std) + 0.1
                dist = distrax.Normal(mean, std)
            
            return distrax.Independent(dist, 1)
        
        def _get_img_dist(mean: jnp.ndarray) -> distrax.Distribution:
            dist = distrax.Normal(mean, 1.0)
            dist = distrax.Independent(dist, 3)
            return dist
        
        def _get_reward_dist(mean: jnp.ndarray) -> distrax.Distribution:
            dist = distrax.Normal(mean, 1.0)
            return dist
    
        # define loss functions + update functions
        def byol_loss_fn_window_size(wm_params: hk.Params,
                                     target_params: hk.Params,
                                     obs_seq: jnp.ndarray,
                                     action_seq: jnp.ndarray,
                                     key: jax.random.PRNGKey,
                                     window_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
            '''Only BYOL-Explore loss.'''
            T, B = obs_seq.shape[:2]
            pred_key, target_key = jax.random.split(key)

            # don't need to do for all idxs that work, we just do it for index T - window_size so we don't multiple count the same losses
            starting_idx = T - window_size
            obs_window = sliding_window(obs_seq, starting_idx, window_size) # (T, B, *obs_dims), everything except [T - window_size:] 0s, rolled to front
            action_window = sliding_window(action_seq, starting_idx, window_size) # (T, B, action_dim), everything except [T - window_size:] 0s, rolled to front
            
            pred_latents, _ = byol_forward(wm_params, pred_key, obs_window, action_window)
            pred_latents = jnp.reshape(pred_latents, (-1,) + pred_latents.shape[2:]) # (T * B, embed_dim)

            _, target_latents = byol_forward(target_params, target_key, obs_window, action_window) # (T, B, embed_dim)
            target_latents = jnp.reshape(target_latents, (-1,) + target_latents.shape[2:]) # (T * B, embed_dim)

            # normalize latents
            pred_latents = l2_normalize(pred_latents, axis=-1)
            target_latents = l2_normalize(target_latents, axis=-1)

            # take L2 loss and reshape for correctness in exploration bonus (BYOL loss)
            mses = jnp.square(pred_latents - jax.lax.stop_gradient(target_latents)) # (T * B, embed_dim)
            mses = jnp.reshape(mses, (-1, B) + mses.shape[1:]) # (T, B, embed_dim)
            mses = sliding_window(mses, 0, window_size) # zeros out losses we don't care about (i.e. past window size)
            mses = jnp.roll(mses, shift=starting_idx, axis=0)
            byol_loss_vec = jnp.sum(mses, -1) # (T, B)
            
            byol_loss = jnp.mean(byol_loss_vec)
            return byol_loss, byol_loss_vec
        
        def dreamer_loss_fn(wm_params: hk.Params,
                            obs_seq: jnp.ndarray,
                            action_seq: jnp.ndarray,
                            reward_seq: jnp.ndarray,
                            key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
            '''Only dreamer loss, and so no need for masking, and can model entire sequence without overlap.'''
            dreamer_out = dreamer_forward(wm_params, key, obs_seq, action_seq)
            post_img_mean, post_reward_mean, _, _, post_stats, prior_stats = dreamer_out
            
            img_dist = _get_img_dist(post_img_mean)
            reward_dist = _get_reward_dist(post_reward_mean)
            rec_loss = -img_dist.log_prob(obs_seq).mean() - reward_dist.log_prob(reward_seq).mean()
            
            post_dist = _get_latent_dist(post_stats)
            prior_dist = _get_latent_dist(prior_stats)
            
            if cfg.kl_balancing > 0:
                post_dist_sg = _get_latent_dist(post_stats, sg=True)
                prior_dist_sg = _get_latent_dist(prior_stats, sg=True)
                
                kl = cfg.kl_balancing * post_dist_sg.kl_divergence(prior_dist).mean() + (1.0 - cfg.kl_balancing) * prior_dist_sg.kl_divergence(post_dist).mean()
            else:
                kl = post_dist.kl_divergence(prior_dist).mean()
            
            metrics = {
                'rec': rec_loss,
                'kl': kl
            }
            
            return rec_loss + kl, metrics
        
        def byol_loss_fn(wm_params: hk.Params,
                         target_params: hk.Params,
                         obs_seq: jnp.ndarray,
                         action_seq: jnp.ndarray,
                         key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            T, B = obs_seq.shape[:2]

            def ws_body_fn(ws, curr_state):
                curr_loss, curr_loss_window, key = curr_state
                loss_key, moveon_key = jax.random.split(key)
                loss, loss_window = byol_loss_fn_window_size(wm_params, target_params, obs_seq, action_seq, loss_key, ws)
                return curr_loss + loss, curr_loss_window + loss_window, moveon_key
            
            init_state = (0.0, jnp.zeros((T, B)), key)
            total_loss, total_loss_window, _ = jax.lax.fori_loop(1, T + 1, ws_body_fn, init_state)
            return total_loss / T, total_loss_window / T # take avgs
        
        def total_loss_fn(wm_params: hk.Params,
                          target_params: hk.Params,
                          obs_seq: jnp.ndarray,
                          action_seq: jnp.ndarray,
                          reward_seq: jnp.ndarray,
                          key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
            '''Combining BYOL and Dreamer losses.'''
            byol_key, dreamer_key = jax.random.split(key)
            byol_loss, _ = byol_loss_fn(wm_params, target_params, obs_seq, action_seq, byol_key)
            dreamer_loss, metrics = dreamer_loss_fn(wm_params, obs_seq, action_seq, reward_seq, dreamer_key)
            metrics['byol'] = byol_loss
            
            total_loss = dreamer_loss + beta * byol_loss # switching it so model focuses more on dreamer
            metrics['total'] = total_loss
            return total_loss, metrics
        
        def update(train_state: BYOLTrainState,
                   obs: jnp.ndarray,
                   actions: jnp.ndarray,
                   rewards: jnp.ndarray,
                   step: int) -> Tuple[BYOLTrainState, Dict]:
            '''Updates the model.'''
            del step
            
            update_key, state_key = jax.random.split(train_state.rng_key)
            loss_grad_fn = jax.value_and_grad(total_loss_fn, has_aux=True)
            (loss, metrics), grads = loss_grad_fn(train_state.wm_params, train_state.target_params, obs, actions, rewards, update_key)
            
            if cfg.pmap:
                loss = jax.lax.pmean(loss, axis_name='i') # maybe use jax.tree_util.tree_map later if this doesn't work and is actually needed
                grads = jax.lax.pmean(grads, axis_name='i')
            
            update, new_opt_state = wm_opt.update(grads, train_state.wm_opt_state)
            new_params = optax.apply_updates(train_state.wm_params, update)
            
            new_target_params = target_update_fn(new_params, train_state.target_params, ema)
            new_train_state = BYOLTrainState(
                wm_params=new_params,
                target_params=new_target_params,
                wm_opt_state=new_opt_state,
                rng_key=state_key # TODO is this the right way to set keys?
            )
            
            return new_train_state, metrics
        
        def compute_uncertainty(obs_seq: jnp.ndarray,
                                action_seq: jnp.ndarray,
                                step: int):
            '''Computes transition uncertainties according to part (iv) in BYOL-Explore paper.
            
            :param obs_seq: Sequence of observations, of shape (seq_len, B, obs_dim)
            :param action_seq: Sequence of actions, of shape (seq_len, B, action_dim)
            
            :return uncertainties: Model uncertainties, of shape (seq_len, B).
            '''
            del step
            _, losses = byol_loss_fn(self.train_state.wm_params, self.train_state.target_params, obs_seq, action_seq)
            # losses are of shape (T, B), result of only BYOL loss accumulation
            return jax.lax.stop_gradient(losses)
        
        # ====== eval methods ======
        
        def eval(obs_seq: jnp.ndarray,
                 action_seq: jnp.ndarray,
                 post: bool = True):
            '''Evaluate from posterior on a test trajectory from the offline dataset.'''
            eval_key, state_key = jax.random.split(self.train_state.rng_key)
            post_img_means, _, prior_img_means, _, _, _ = dreamer_forward(self.train_state.wm_params, eval_key, obs_seq, action_seq)
            img_means = jnp.where(post, post_img_means, prior_img_means)
            img_means = (img_means + 0.5) * 255.0
            img_means = jnp.squeeze(img_means[:, :, :, :, :3]) # just the first image
            
            new_train_state = self.train_state._replace(rng_key=state_key)
            return new_train_state, jax.lax.stop_gradient(img_means)
        
        # whether to parallelize across devices, make sure to have multiple devices here for this for better performance
        # auto jits so don't need to do jax.jit before pmap
        if cfg.pmap:
            self._update = jax.pmap(update, axis_name='i')
            self._compute_uncertainty = jax.pmap(compute_uncertainty, axis_name='i')
            self._eval = jax.pmap(eval, axis_name='i')
        else:
            self._update = jax.jit(update)
            self._compute_uncertainty = jax.jit(compute_uncertainty)
            self._eval = jax.jit(eval)
    
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            dill.dump(self.train_state, f, protocol=2)
            
    def load(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                train_state = dill.load(f)
                self.train_state = train_state
        except FileNotFoundError:
            print('cannot load BYOL-Offline model')
            exit()