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
        self.offline_dir = Path(to_absolute_path('offline_data')) / self.cfg.task / self.cfg.level
        
        # world model directory
        final_dir = 'byol' if self.cfg.train_byol else 'rnd'
        self.pretrained_model_dir = Path(to_absolute_path('pretrained_models')) / final_dir / self.cfg.task / self.cfg.level
        self.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        
        # trained policy directory
        self.policy_dir = Path(to_absolute_path('trained_policies')) / self.cfg.task / self.cfg.level
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        
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
            model_path = self.pretrained_model_dir / 'rnd.pkl'
            self.rnd_trainer.load(model_path)
           
        # BYOL model stuff
        self.byol_trainer = WorldModelTrainer(self.cfg.byol)
        if self.cfg.load_model:
            model_path = self.pretrained_model_dir / 'byol.pkl'
            self.byol_trainer.load(model_path)
            
        # RND dataloader
        rnd_buffer = TransitionReplayBuffer(self.offline_dir)
        self.rnd_dataloader = rnd_dataloader(rnd_buffer, self.cfg.max_steps, self.cfg.model_batch_size)
        
        # BYOL dataloader
        byol_buffer = SequenceReplayBuffer(self.offline_dir, self.cfg.seq_len)
        self.byol_dataloader = model_dataloader(byol_buffer, self.cfg.max_steps, self.cfg.model_batch_size)
        
        # policy
        if self.cfg.learner == 'ddpg':
            self.agent = DDPG(self.cfg, self.byol_trainer, self.rnd_trainer)
        
        # rng (in case we actually need to use it later on)
        self.rng = jax.random.PRNGKey(self.cfg.seed)

    def train_byol(self):
        '''Train BYOL-Explore latent world model offline.'''
        for epoch in tqdm(range(1, self.cfg.model_train_epochs + 1)):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.byol_dataloader:
                obs, actions = batch
                batch_metrics = self.byol_trainer.update(obs, actions, self.global_step)
                
                for k, v in batch_metrics.items():
                    if epoch_metrics[k] is None:
                        epoch_metrics[k] = AverageMeter()
                    epoch_metrics[k].update(v, obs.shape[0]) # want to log per example, not per batch avgs
                
            if self.cfg.wandb:
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                wandb.log(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_model_dir / f'wm_{epoch}.pkl'
                self.byol_trainer.save(model_path)
                
    def train_rnd(self):
        '''Train RND model offline.'''
        for epoch in tqdm(range(1, self.cfg.model_train_epochs + 1)):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in self.rnd_dataloader:
                obs, _, _, _, _ = batch
                batch_metrics = self.rnd_trainer.update(obs, self.global_step)
                
                for k, v in batch_metrics.items():
                    if epoch_metrics[k] is None:
                        epoch_metrics[k] = AverageMeter()
                    epoch_metrics[k].update(v, obs.shape[0]) # want to log per example, not per batch avgs
            
            # print(epoch_metrics)
            if self.cfg.wandb:
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                wandb.log(log_dump)
            
            if self.cfg.save_model and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_model_dir / f'rnd_{epoch}.pkl'
                self.rnd_trainer.save(model_path)
                
    def train_agent(self):
        '''Train offline RL agent.'''
        pass
                
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
        
    def train_model_fn():
        if cfg.train_byol:
            workspace.train_byol()
        else:
            workspace.train_rnd()
    
    if cfg.wandb:
        with wandb.init(project=project_name, entity=entity, name=name) as run:
            train_model_fn()
    else:
        train_model_fn()
        
if __name__ == '__main__':
    main()