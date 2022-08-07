import jax
from tqdm import tqdm
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from collections import defaultdict
import wandb
import datetime
import numpy as np

from byol_offline.models.wm import *
from byol_offline.agents import *
import envs.dmc as dmc
from memory.replay_buffer import PolicyReplayBuffer, SequenceReplayBuffer, model_dataloader
from utils import *

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'current working directory: {self.work_dir}')
        
        self.cfg = cfg
        self.setup()
        
        self.global_step = 0
    
    def setup(self):
        # offline directory
        self.offline_dir = Path(to_absolute_path('offline_data')) / self.cfg.task / self.cfg.level
        
        # world model directory
        self.pretrained_wm_dir = Path(to_absolute_path('pretrained_wms')) / self.cfg.task / self.cfg.level
        self.pretrained_wm_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # world model stuff
        self.wm_trainer = WorldModelTrainer(self.cfg.wm)
        # rand_obs = np.random.uniform(size=(25, 20, 64, 64, 3))
        # rand_action = np.random.uniform(size=(25, 20, 6))
        # wm_params = self.wm_trainer.train_state.wm_params
        # print(f'latent, embedding shapes: {self.wm_trainer.wm.apply(wm_params, rand_obs, rand_action)[0].shape, self.wm_trainer.wm.apply(wm_params, rand_obs, rand_action)[1].shape}')
        
        if self.cfg.load_model:
            model_path = self.pretrained_wm_dir / 'wm.pkl'
            self.wm_trainer.load(model_path)
            
        # WM dataloader
        buffer = SequenceReplayBuffer(self.offline_dir, self.cfg.seq_len)
        self.model_dataloader = model_dataloader(buffer, self.cfg.max_steps, self.cfg.model_batch_size)
        
        # policy
        if self.cfg.learner == 'ddpg':
            self.agent = DDPG(self.cfg)
            
        self.policy_rb = PolicyReplayBuffer(self.cfg.policy_rb_capacity, self.cfg.obs_shape, self.cfg.action_shape)
        
        # rng (in case we actually need to use it later on)
        self.rng = jax.random.PRNGKey(self.cfg.seed)
            
    def train_wm(self):
        for epoch in range(1, self.cfg.model_train_epochs + 1):
            epoch_metrics = defaultdict(AverageMeter)
            for batch in tqdm(self.model_dataloader):
                obs, action = batch
                # print(obs.shape, action.shape)
                batch_metrics = self.wm_trainer.update(obs, action, self.global_step)
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k].update(v, 1)
                
            if self.cfg.wandb:
                log_dump = {k: v.value() for k, v in epoch_metrics.items()}
                wandb.log(log_dump)
            
            if self.cfg.save_wm and epoch % self.cfg.model_save_every == 0:
                model_path = self.pretrained_wm_dir / f'model_{epoch}.pkl'
                self.wm_trainer.save(model_path)
                
    def train_agent(self):
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
    
    if cfg.wandb:
        with wandb.init(project=project_name, entity=entity, name=name) as run:
            workspace.train_wm()
    else:
        workspace.train_wm()
        
if __name__ == '__main__':
    main()