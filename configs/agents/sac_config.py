from ml_collections import ConfigDict

from configs.base_config import get_config as get_base_config


def get_config():
    """Gets the config for SAC training."""
    
    config = get_base_config()
    
    # ----- encoder -----
    
    config.encoder = ConfigDict()
    
    config.encoder.pixel = ConfigDict()
    config.encoder.pixel.dreamer = False
    config.encoder.pixel.depth = 32
    
    config.encoder.state = ConfigDict()
    config.encoder.state.hidden_dims = (256, 256)
    
    # ----- actor + critic -----
    
    config.ac_hidden_dim = 256
    
    # ----- add learning parameters -----
    
    config.encoder_lr = 1e-4
    config.actor_lr = 1e-4
    config.critic_lr = 1e-4
    config.alpha_lr = 1e-4
    
    config.discount = 0.99
    config.ema = 0.01
    config.target_update_frequency = 1
    
    return config