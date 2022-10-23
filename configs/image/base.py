from dataclasses import dataclass
from simple_ddpm.models.unet_stable import UNet2DConfig, UNet2DModule


@dataclass
class EMAConfig:
    decay: float = 0.995
    update_every: int = 10
    update_after_step: int = 100


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 1_000


@dataclass
class OptimizerConfig:
    lr_start: float = 2e-5
    drop_1_mult: float = 1.0
    drop_2_mult: float = 1.0


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 500
    total_samples: int = 5_000_000
    loss_type: str = "mae"
    dataset: str = "cartoonset"
    viz: str = "matplotlib"
    model: str = "stable_unet"
    viz_progress_every: int = 2000
    log_every: int = 50
    ema: EMAConfig = EMAConfig()
    schedule: DiffusionConfig = DiffusionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    stable_diffusion_unet: UNet2DConfig = UNet2DConfig(
        out_channels=3,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=(
            128,
            128,
            256,
            256,
        ),
        cross_attention_dim=256,
    )

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.total_samples // self.batch_size


def get_config() -> Config:
    return Config()
