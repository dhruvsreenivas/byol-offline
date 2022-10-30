import warnings
warnings.filterwarnings("ignore")

import jax
from tqdm import trange
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from collections import defaultdict
import wandb
from datetime import datetime

from byol_offline.models import *
from byol_offline.agents import *
import envs.dmc as dmc
from memory.replay_buffer import *
from utils import *

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'current working directory: {self.work_dir}')
        
        self.cfg = cfg
        self.setup()
        print('Finished setting up')
        
        self.global_step = 0
    
    def setup(self):
        # offline directory
        if self.cfg.task not in MUJOCO_ENVS:
            self.offline_dir = Path(to_absolute_path('offline_data')) / self.cfg.task / self.cfg.level

        # assert that we're training byol model and using byol as reward aug at the same time
        assert (self.cfg.train_byol and self.cfg.reward_aug == 'byol') or (not self.cfg.train_byol and self.cfg.reward_aug == 'rnd'), "Can't train model and then not use said model in RL training."
        
        # model directories
        self.pretrained_byol_dir = Path(to_absolute_path('pretrained_models')) / 'byol' / self.cfg.task / self.cfg.level
        self.pretrained_byol_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_rnd_dir = Path(to_absolute_path('pretrained_models')) / 'rnd' / self.cfg.task / self.cfg.level / ('actions' if self.cfg.rnd.cat_actions else 'no_actions')
        self.pretrained_rnd_dir.mkdir(parents=True, exist_ok=True)
        
        # trained agent directory
        self.agent_dir = Path(to_absolute_path('trained_policies')) / self.cfg.task / self.cfg.level / self.cfg.learner
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cfg.task in MUJOCO_ENVS:
            self.train_env = make_gym_env(self.cfg.task, self.cfg.level)
            self.eval_env = make_gym_env(self.cfg.task, self.cfg.level)
            self.eval_env.seed(self.cfg.seed + 12345)
            
            self.cfg.obs_shape = self.train_env.observation_space.shape
            self.cfg.action_shape = self.train_env.action_space.shape
            self.cfg.max_action = float(self.train_env.action_space.high[0])
        else:
            self.train_env = dmc.make(
                self.cfg.task,
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.seed,
                self.cfg.img_size
            )
            self.eval_env = dmc.make(
                self.cfg.task,
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.seed,
                self.cfg.img_size
            )
            obs_spec = self.train_env.observation_spec()
            self.cfg.obs_shape = (obs_spec[0] * self.cfg.frame_stack, ) + obs_spec[1:]
            self.cfg.action_shape = self.train_env.action_spec().shape
        
        # RND model stuff
        self.rnd_trainer = RNDModelTrainer(self.cfg.rnd)
        if self.cfg.load_model and self.cfg.reward_aug == 'rnd':
            model_path = self.pretrained_rnd_dir / f'rnd_{self.cfg.model_train_epochs}.pkl'
            self.rnd_trainer.load(model_path)

        # BYOL model stuff
        self.byol_trainer = WorldModelTrainer(self.cfg.byol)
        if self.cfg.load_model and self.cfg.reward_aug == 'byol':
            model_path = self.pretrained_byol_dir / f'byol_{self.cfg.model_train_epochs}.pkl'
            self.byol_trainer.load(model_path)
            
        # RND dataloader
        if self.cfg.sample_batches:
            if self.cfg.task not in MUJOCO_ENVS:
                rnd_buffer = VD4RLTransitionReplayBuffer(self.offline_dir, self.cfg.frame_stack)
            else:
                lvl = 'medium-expert' if self.cfg.level == 'med_exp' else self.cfg.level # TODO make better
                rnd_buffer = D4RLTransitionReplayBuffer(self.cfg.task, lvl, normalize=self.cfg.normalize_inputs)
                if self.cfg.normalize_inputs:
                    self.dataset_stats = (
                        rnd_buffer.state_mean,
                        rnd_buffer.action_mean,
                        rnd_buffer.next_state_mean,
                        rnd_buffer.state_scale,
                        rnd_buffer.action_scale,
                        rnd_buffer.next_state_scale
                    )
            
            self.rnd_dataloader = rnd_sampling_dataloader(rnd_buffer, self.cfg.max_steps, self.cfg.model_batch_size)
        else:
            assert self.cfg.task in MUJOCO_ENVS, 'Do not currently have iterative support for DMC tasks.'
            lvl = 'medium-expert' if self.cfg.level == 'med_exp' else self.cfg.level # TODO make better
            self.rnd_dataloader = rnd_iterative_dataloader(self.cfg.task, lvl, self.cfg.model_batch_size, normalize=self.cfg.normalize_inputs)
        
        # BYOL dataloader
        if self.cfg.task not in MUJOCO_ENVS:
            byol_buffer = VD4RLSequenceReplayBuffer(self.offline_dir, self.cfg.seq_len)
        else:
            lvl = 'medium-expert' if self.cfg.level == 'med_exp' else self.cfg.level # TODO make better
            byol_buffer = D4RLSequenceReplayBuffer(self.cfg.task, lvl, self.cfg.seq_len, normalize=self.cfg.normalize_inputs)
            # already got stats by this point so we're good
            
        self.byol_dataloader = byol_sampling_dataloader(byol_buffer, self.cfg.max_steps, self.cfg.model_batch_size)
        # TODO make BYOL dataloader iterative if needed

        # RL agent dataloader
        if self.cfg.train_byol:
            self.agent_dataloader = byol_sampling_dataloader(byol_buffer, self.cfg.policy_rb_capacity, self.cfg.policy_batch_size)
        else:
            if self.cfg.sample_batches:
                self.agent_dataloader = rnd_sampling_dataloader(rnd_buffer, self.cfg.policy_rb_capacity, self.cfg.policy_batch_size)
            else:
                lvl = 'medium-expert' if self.cfg.level == 'med_exp' else self.cfg.level # TODO make better
                self.agent_dataloader = rnd_iterative_dataloader(self.cfg.task, lvl, self.cfg.policy_batch_size, normalize=self.cfg.normalize_inputs)
        
        # RL agent
        if self.cfg.learner == 'ddpg':
            self.agent = DDPG(self.cfg, self.byol_trainer, self.rnd_trainer)
        elif self.cfg.learner == 'sac':
            self.agent = SAC(self.cfg, self.byol_trainer, self.rnd_trainer)
        else:
            self.agent = TD3(self.cfg, self.byol_trainer, self.rnd_trainer)
            
        # sanity checking (BC + simple dynamics model)
        self.simple_dynamics_trainer = SimpleDynamicsTrainer(self.cfg.simple_dynamics)
        self.bc = BC(self.cfg)
        
        # rng (in case we actually need to use it later on)
        self.rng = jax.random.PRNGKey(self.cfg.seed)

    # ==================== MODEL TRAINING ====================
    
    def train_byol(self):
        '''Train BYOL-Explore latent world model offline.'''
        for epoch in trange(1, self.cfg.model_train_epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.byol_dataloader:
                obs, actions, _, _, _ = batch
                new_train_state, batch_metrics = self.byol_trainer._update(self.byol_trainer.train_state, obs, actions, self.global_step)
                self.byol_trainer.train_state = new_train_state
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, obs.shape[1]) # want to log per batch
                
            log_dump = {k: v.value() for k, v in epoch_metrics.items()}
            if self.cfg.wandb:
                wandb.log(log_dump)
            else:
                print_dict(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_byol_dir / f'byol_{epoch}.pkl'
                self.byol_trainer.save(model_path)
                
    def train_rnd(self):
        '''Train RND model offline.'''
        for epoch in trange(1, self.cfg.model_train_epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.rnd_dataloader:
                obs, actions, _, _, _ = batch
                new_train_state, batch_metrics = self.rnd_trainer._update(self.rnd_trainer.train_state, obs, actions, self.global_step)
                self.rnd_trainer.train_state = new_train_state
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, obs.shape[0])
            
            log_dump = {k: v.value() for k, v in epoch_metrics.items()}
            if self.cfg.wandb:
                wandb.log(log_dump)
            else:
                print_dict(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_rnd_dir / f'rnd_{epoch}.pkl'
                self.rnd_trainer.save(model_path)
                
    # ==================== AGENT TRAINING ====================
    
    def train_bc(self):
        eval_every = Every(self.cfg.policy_eval_every)
        
        for epoch in trange(1, self.cfg.policy_train_epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.agent_dataloader:
                transitions = Transition(*batch)
                
                new_train_state, batch_metrics = self.bc._update(self.bc.train_state, transitions, self.global_step)
                self.bc.train_state = new_train_state
                
                batch_size = transitions.obs.shape[1] if self.cfg.train_byol else transitions.obs.shape[0]
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, batch_size)
            
            log_dump = {k: v.value() for k, v in epoch_metrics.items()}
            if self.cfg.wandb:
                wandb.log(log_dump)
            else:
                print_dict(log_dump)

            # eval when necessary (in the beginning as well)
            if epoch == 1 or eval_every(epoch):
                # TODO write eval
                pass
                
    def eval_agent_mujoco(self):
        '''Evaluates agent in MuJoCo envs.'''
        episode_rewards = []
        episode_count = 0
        episode_until = Until(self.cfg.num_eval_episodes)

        while episode_until(episode_count):
            ob = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                # normalize if needed
                if self.cfg.normalize_inputs:
                    state_mean = self.dataset_stats[0]
                    state_std = self.dataset_stats[3]
                    ob = (ob - state_mean) / state_std
                
                action = self.agent._act(ob, self.global_step, eval_mode=True)
                action = np.asarray(action)

                n_ob, r, done, _ = self.eval_env.step(action)
                episode_reward += r
                
                ob = n_ob
                
            episode_rewards.append(episode_reward)
            episode_count += 1
        
        avg_reward = np.mean(episode_rewards)
        d4rl_normalized_score = self.eval_env.get_normalized_score(avg_reward) * 100
        metrics = {
            'eval_rew_mean': avg_reward,
            'eval_rew_std': np.std(episode_rewards),
            'd4rl_normalized_score': d4rl_normalized_score
        }
        if self.cfg.wandb:
            wandb.log(metrics)
        else:
            print_dict(metrics)

    def eval_agent_dmc(self):
        '''Evaluates agent in DMC envs.'''
        episode_rewards = []
        episode_count = 0
        episode_until = Until(self.cfg.num_eval_episodes)

        while episode_until(episode_count):
            time_step = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                ob = time_step.observation
                action = self.agent._act(ob, self.global_step, eval_mode=True)
                action = np.asarray(action)

                time_step = self.eval_env.step(action)
                reward = time_step.reward
                done = time_step.last()

                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            episode_count += 1
        
        metrics = {
            'eval_rew_mean': np.mean(episode_rewards),
            'eval_rew_std': np.std(episode_rewards)
        }
        
        if self.cfg.wandb:
            wandb.log(metrics)
        else:
            print_dict(metrics)

    def eval_agent(self):
        if self.cfg.task in MUJOCO_ENVS:
            self.eval_agent_mujoco()
        else:
            self.eval_agent_dmc()
            
    def train_agent(self):
        '''Train offline RL agent.'''
        eval_every = Every(self.cfg.policy_eval_every)
        save_every = Every(self.cfg.model_save_every)
        
        if self.cfg.bc_warmstart:
            print('====== STARTING BC WARMSTART ... ======')
            for epoch in trange(1, self.cfg.bc_epochs + 1):
                epoch_metrics = defaultdict(AverageMeter)
                for batch in self.agent_dataloader:
                    transitions = Transition(*batch)
                    new_train_state, bc_metrics = self.agent._bc_update(self.agent.train_state, transitions, self.global_step)
                    self.agent.train_state = new_train_state
                    
                    batch_size = transitions.obs.shape[1] if self.cfg.train_byol else transitions.obs.shape[0]
                    for k, v in bc_metrics.items():
                        epoch_metrics[k].update(v, batch_size)
                        
                    self.global_step += 1
                
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                if self.cfg.wandb:
                    wandb.log(log_dump)
                else:
                    print_dict(log_dump)
                    
            print('====== ENDING BC WARMSTART ... STARTING AGENT TRAINING ... ======')
        
        # reset global step for agent training (TODO do we need to do this?)
        self.global_step = 0
        
        for epoch in trange(1, self.cfg.policy_train_epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.agent_dataloader:
                transitions = Transition(*batch)
                
                new_train_state, batch_metrics = self.agent._update(self.agent.train_state, transitions, self.global_step)
                self.agent.train_state = new_train_state
                
                batch_size = transitions.obs.shape[1] if self.cfg.train_byol else transitions.obs.shape[0]
                for k, v in batch_metrics.items():
                    if v < jnp.inf:
                        epoch_metrics[k].update(v, batch_size)
                    
                self.global_step += 1
            
            log_dump = {k: v.value() for k, v in epoch_metrics.items()}
            if self.cfg.wandb:
                wandb.log(log_dump)
            else:
                print_dict(log_dump)
            
            # save when necessary
            if self.cfg.save_model and save_every(epoch) == 0:
                model_path = self.agent_dir / f'agent_{epoch}.pkl'
                self.agent.save(model_path)

            # eval when necessary (in the beginning as well)
            if epoch == 1 or eval_every(epoch):
                self.eval_agent()
                
    # ==================== SANITY CHECKING ====================
    
    def train_simple(self):
        '''Train simple dynamics model.'''
        for epoch in trange(1, self.cfg.model_train_epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.rnd_dataloader:
                transitions = Transition(*batch)
                obs = batch[0]
                new_train_state, metrics = self.simple_dynamics_trainer._update(self.simple_dynmaics_trainer.train_state, transitions, self.global_step)
                self.simple_dynamics_trainer.train_state = new_train_state

                for k, v in metrics.items():
                    epoch_metrics[k].update(v, obs.shape[0])

                if self.cfg.wandb:
                    log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                    wandb.log(log_dump)
                else:
                    log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                    print_dict(log_dump)
                    
    def train_one_datapoint(self):
        '''Train on one datapoint to make sure loss goes down.'''
        self.rng, subkey = jax.random.split(self.rng)
        rand_datapoint = jax.random.normal(key=subkey, shape=(1,) + tuple(self.cfg.obs_shape), dtype=jnp.float32)
        for epoch in trange(1, self.cfg.model_train_epochs + 1):
            new_train_state, metrics = self.rnd_trainer._update(self.rnd_trainer.train_state, rand_datapoint, self.global_step)
            self.rnd_trainer.train_state = new_train_state
            
            if self.cfg.wandb:
                wandb.log(metrics)
            else:
                print_dict(metrics)
                

@hydra.main(config_path='./cfgs', config_name='config')
def main(cfg):
    workspace = Workspace(cfg)
    
    print('=' * 50)
    from jax.lib import xla_bridge
    print(f'Experiment is running on {xla_bridge.get_backend().platform}')
    print('=' * 50)
    
    entity = 'dhruv_sreenivas'
    project_name = cfg.project_name
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    import sys

    name = f"{ts}"
    override_args = filter(lambda x: "=" in x, sys.argv[2:])
    for x in override_args:
        name += f"|{x}"
        
    # training functions
    def train_model():
        if cfg.train_byol:
            workspace.train_byol()
        else:
            workspace.train_rnd()

    def train():
        if cfg.load_model:
            workspace.train_agent()
        else:
            train_model()
            
    def sanity_check():
        if cfg.load_model:
            workspace.train_one_datapoint()
        else:
            workspace.train_simple()
    
    # actual training + sanity checking
    if cfg.wandb:
        with wandb.init(project=project_name, entity=entity, name=name) as run:
            train()
    else:
        train()
        
if __name__ == '__main__':
    main()