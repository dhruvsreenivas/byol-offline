from ml_collections import ConfigDict

from configs.base_config import get_config as get_base_config


def get_config():
    """Gets config for BYOL model training."""

    config = get_base_config()

    # --- add pixel-based BYOL parameters ---

    config.pixel = ConfigDict()

    config.pixel.dreamer = True
    config.pixel.depth = 32

    # RSSM (pixel)
    config.pixel.rssm = ConfigDict()

    config.pixel.rssm.dreamer = config.pixel.dreamer
    config.pixel.rssm.gru_hidden_size = 256
    config.pixel.rssm.use_layer_norm = False
    config.pixel.rssm.stoch_dim = 64
    config.pixel.rssm.stoch_discrete_dim = 1
    config.pixel.rssm.hidden_dim = 256

    # reward MLP args
    config.pixel.reward_hidden_dims = (256, 256)

    # --- add state-based BYOL parameters ---

    config.state = ConfigDict()

    config.state.dreamer = True
    config.state.hidden_dims = (200, 200, 200, 200)
    config.state.repr_dim = 128

    # RSSM (state)
    config.state.rssm = ConfigDict()

    config.state.rssm.dreamer = config.state.dreamer
    config.state.rssm.gru_hidden_size = 64
    config.state.rssm.use_layer_norm = True
    config.state.rssm.stoch_dim = 32
    config.state.rssm.stoch_discrete_dim = 1
    config.state.rssm.hidden_dim = 512

    config.state.reward_hidden_dims = (200, 200, 200, 200)

    # --- more general training parameters ---

    config.initialize_target_with_online_params = True
    config.learning_rate = 6e-4
    config.ema = 0.95
    config.vae_beta = 1.0
    config.beta = 1.0

    config.kl_balance = 0.5
    config.kl_free_value = 1.0

    return config
