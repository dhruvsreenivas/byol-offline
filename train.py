import warnings
warnings.filterwarnings("ignore")

import jax
from tqdm import tqdm
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from collections import defaultdict
import wandb
import datetime

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
        self.pretrained_rnd_dir = Path(to_absolute_path('pretrained_models')) / 'rnd' / self.cfg.task / self.cfg.level
        self.pretrained_rnd_dir.mkdir(parents=True, exist_ok=True)
        
        # trained policy directory
        self.policy_dir = Path(to_absolute_path('trained_policies')) / self.cfg.task / self.cfg.level / self.cfg.learner
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cfg.task in MUJOCO_ENVS:
            self.train_env = make_gym_env(self.cfg.task)
            self.eval_env = make_gym_env(self.cfg.task)
            
            self.cfg.obs_shape = self.train_env.observation_space.shape
            self.cfg.action_shape = self.train_env.action_space.shape
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
            self.cfg.obs_shape = self.train_env.observation_spec().shape
            self.cfg.action_shape = self.train_env.action_spec().shape
        
        # RND model stuff
        self.rnd_trainer = RNDModelTrainer(self.cfg.rnd)
        if self.cfg.load_model:
            model_path = self.pretrained_rnd_dir / f'rnd_{self.cfg.model_train_epochs}.pkl' # TODO: don't hardcore
            self.rnd_trainer.load(model_path)

        # BYOL model stuff
        self.byol_trainer = WorldModelTrainer(self.cfg.byol)
        if self.cfg.load_model:
            model_path = self.pretrained_byol_dir / f'byol_{self.cfg.model_train_epochs}.pkl' # TODO: don't hardcore
            self.byol_trainer.load(model_path)
            
        # RND dataloader
        if self.cfg.task not in MUJOCO_ENVS:
            rnd_buffer = VD4RLTransitionReplayBuffer(self.offline_dir)
        else:
            lvl = 'medium-expert' if self.cfg.level == 'med_exp' else self.cfg.level # TODO make better
            rnd_buffer = D4RLTransitionReplayBuffer(self.cfg.task, lvl)
        self.rnd_dataloader = rnd_dataloader(rnd_buffer, self.cfg.max_steps, self.cfg.model_batch_size)
        
        # BYOL dataloader
        if self.cfg.task not in MUJOCO_ENVS:
            byol_buffer = VD4RLSequenceReplayBuffer(self.offline_dir, self.cfg.seq_len)
        else:
            lvl = 'medium-expert' if self.cfg.level == 'med_exp' else self.cfg.level # TODO make better
            byol_buffer = D4RLSequenceReplayBuffer(self.cfg.task, lvl, self.cfg.seq_len)
        self.byol_dataloader = byol_dataloader(byol_buffer, self.cfg.max_steps, self.cfg.model_batch_size)

        # RL agent dataloader
        self.agent_dataloader = self.byol_dataloader if self.cfg.train_byol else self.rnd_dataloader
        
        # policy
        if self.cfg.learner == 'ddpg':
            self.agent = DDPG(self.cfg, self.byol_trainer, self.rnd_trainer)
        else:
            self.agent = SAC(self.cfg, self.byol_trainer, self.rnd_trainer)
        
        # rng (in case we actually need to use it later on)
        self.rng = jax.random.PRNGKey(self.cfg.seed)

    def train_byol(self):
        '''Train BYOL-Explore latent world model offline.'''
        for epoch in tqdm(range(1, self.cfg.model_train_epochs + 1)):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.byol_dataloader:
                obs, actions, _, _, _ = batch
                batch_metrics = self.byol_trainer.update(obs, actions, self.global_step)
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, obs.shape[0] * obs.shape[1]) # want to log per example, not per batch avgs
                
            if self.cfg.wandb:
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                wandb.log(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_byol_dir / f'byol_{epoch}.pkl'
                self.byol_trainer.save(model_path)
                
    def train_rnd(self):
        '''Train RND model offline.'''
        for epoch in tqdm(range(1, self.cfg.model_train_epochs + 1)):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.rnd_dataloader:
                obs, _, _, _, _ = batch
                batch_metrics = self.rnd_trainer.update(obs, self.global_step)
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, obs.shape[0]) # want to log per example, not per batch avgs
            
            if self.cfg.wandb:
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                wandb.log(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_rnd_dir / f'rnd_{epoch}.pkl'
                self.rnd_trainer.save(model_path)
                
    def eval_agent(self):
        episode_rewards = []
        episode_count = 0
        episode_until = Until(self.cfg.num_eval_episodes)

        while episode_until(episode_count):
            ob = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.agent.act(ob, self.global_step, eval_mode=True)
                n_ob, r, done, _ = self.eval_env.step(action)
                episode_reward += r
                
                ob = n_ob
                
            episode_rewards.append(episode_reward)
            episode_count += 1
            
        metrics = {
            'eval_rew_mean': np.mean(episode_rewards),
            'eval_rew_std': np.std(episode_rewards)
        }
        if self.cfg.wandb:
            wandb.log(metrics)
            
    def train_agent(self):
        '''Train offline RL agent.'''
        for epoch in tqdm(range(1, self.cfg.policy_train_epochs + 1)):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.agent_dataloader:
                transitions = Transition(*batch)
                batch_metrics = self.agent.update(transitions, self.global_step)
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, transitions.obs.shape[0])

                self.global_step += 1
                
            if self.cfg.wandb:
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                wandb.log(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.policy_dir / f'policy_{epoch}.pkl'
                self.agent.save(model_path)
                
@hydra.main(config_path='./cfgs', config_name='config')
def main(cfg):
    workspace = Workspace(cfg)
    
    print('=' * 50)
    from jax.lib import xla_bridge
    print(f'Experiment is running on {xla_bridge.get_backend().platform}')
    print('=' * 50)
    
    entity = 'dhruv_sreenivas'
    project_name = cfg.project_name
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    import sys

    name = f"{ts}"
    override_args = filter(lambda x: "=" in x, sys.argv[2:])
    for x in override_args:
        name += f"|{x}"
        
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
    
    if cfg.wandb:
        with wandb.init(project=project_name, entity=entity, name=name) as run:
            train()
    else:
        train()
        
if __name__ == '__main__':
    main()