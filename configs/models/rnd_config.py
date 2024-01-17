from ml_collections import ConfigDict

from configs.base_config import get_config as get_base_config


def get_config():
    """Gets config for RND training."""
    
    config = get_base_config()
    
    # ----- add base parameters -----
    
    config.cat_actions = False
    config.hidden_dim = 256
    config.repr_dim = 50
    
    # ----- add pixel-based RND parameters ----
    
    config.pixel = ConfigDict()
    
    config.pixel.dreamer = True
    config.pixel.depth = 32
    
    config.pixel.hidden_dim = config.hidden_dim
    config.pixel.repr_dim = config.repr_dim
    
    # ----- add state-based RND parameters -----
    
    config.state = ConfigDict()
    
    config.state.encoder_dims = (400, 400, 400)
    config.state.hidden_dim = config.hidden_dim
    config.state.repr_dim = config.repr_dim

    # ----- learning parameters -----
    
    config.learning_rate = 1e-3
    config.l1 = False
    
    return config