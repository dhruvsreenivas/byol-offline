from ml_collections import config_dict, ConfigDict


def get_config():
    """Base config."""

    config = ConfigDict()

    # whether to pmap the update function, set at training time
    config.pmap = config_dict.placeholder(bool)

    # optimizer class -- always needed
    config.optimizer_class = "adam"

    return config
