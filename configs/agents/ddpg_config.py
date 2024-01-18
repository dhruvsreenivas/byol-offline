from ml_collections import ConfigDict

from configs.base_config import get_config as get_base_config


def get_config():
    """Gets config for DDPG training."""
    
    config = get_base_config()
    
    # ----- encoder -----
    
    config.encoder = ConfigDict()
    
    config.encoder.pixel = ConfigDict()
    config.encoder.pixel.dreamer = False
    config.encoder.pixel.depth = 32
    
    config.encoder.state = ConfigDict()
    config.encoder.state.hidden_dims = (256, 256)
    
    # ----- actor + critic -----
    
    config.feature_dim = 50
    config.ac_hidden_dim = 1024
    
    # ----- add learning parameters -----
    
    config.encoder_lr = 1e-4
    config.actor_lr = 1e-4
    config.critic_lr = 1e-4
    
    config.discount = 0.99
    config.ema = 0.01
    config.update_every_steps = 2
    
    config.init_std = 1.0
    config.final_std = 0.1
    config.std_duration = 500_000
    config.std_clip_val = 0.3
    
    return config