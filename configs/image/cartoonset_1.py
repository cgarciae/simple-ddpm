from configs.image.base import Config


def get_config() -> Config:
    config = Config()
    config.optimizer.lr_start = 1e-4
    config.optimizer.drop_1_mult = 1 / 2
    config.optimizer.drop_2_mult = 2 / 5
    return config
