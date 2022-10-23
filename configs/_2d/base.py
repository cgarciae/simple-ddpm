from dataclasses import dataclass


@dataclass
class EMAConfig:
    decay: float = 0.995
    update_every: int = 10
    update_after_step: int = 100


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 1e-5
    beta_end: float = 0.01
    timesteps: int = 1_000


@dataclass
class OptimizerConfig:
    lr_start: float = 1e-3
    drop_1_mult: float = 1
    drop_2_mult: float = 1


@dataclass
class ModelConfig:
    units: int = 128
    emb_dim: int = 32


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 10
    total_samples: int = 5_000_000
    loss_type: str = "mae"
    dataset: str = "moons"
    viz: str = "matplotlib"
    eval_every: int = 2000
    log_every: int = 200
    ema: EMAConfig = EMAConfig()
    schedule: DiffusionConfig = DiffusionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig = ModelConfig()

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.total_samples // self.batch_size


def get_config() -> Config:
    return Config()
