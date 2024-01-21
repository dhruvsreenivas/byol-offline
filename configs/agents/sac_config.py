from ml_collections import config_dict, ConfigDict

from configs.base_config import get_config as get_base_config


def get_config():
    """Gets the config for SAC training."""
    
    config = get_base_config()
    
    # ----- representation dim (undefined if just going from straight pixels or states) -----
    
    config.observation_repr_dim = config_dict.placeholder(int)
    
    # ----- encoder -----
    
    config.encoder = ConfigDict()
    
    config.encoder.pixel = ConfigDict()
    config.encoder.pixel.dreamer = True
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
    
    config.cql_alpha = 0.0
    config.cql_samples = 16
    config.discount = 0.99
    config.ema = 0.01
    config.target_update_frequency = 1
    
    return config