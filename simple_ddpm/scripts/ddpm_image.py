# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

from dataclasses import dataclass
from functools import partial

import ciclo
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
from ciclo import Elapsed
from clu.metrics import Average, Collection
from datasets.load import load_dataset
from einop import einop
from flax import struct
from flax.training import train_state
from simple_ddpm.models.unet_lucid import UNet
from simple_ddpm.models.unet_stable import UNet2DConfig, UNet2DModule
from simple_ddpm.processes import GaussianDiffusion
from simple_ddpm.utils import EMA, get_wandb_run, parse_config, render_image, show
from simple_ddpm import schedules


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
    viz: str = "wandb"
    model: str = "stable_unet"
    eval_every: int = 2000
    log_every: int = 200
    ema: EMAConfig = EMAConfig()
    schedule: DiffusionConfig = DiffusionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.total_samples // self.batch_size


config = parse_config(Config)
wandb_run = get_wandb_run(config)
print(jax.devices())

if config.loss_type == "mse":
    loss_metric = lambda a, b: jnp.mean((a - b) ** 2)
elif config.loss_type == "mae":
    loss_metric = lambda a, b: jnp.mean(jnp.abs(a - b))
else:
    raise ValueError(f"Unknown loss type {config.loss_type}")


# %%
def get_data(dataset: str, batch_size: int):
    if dataset == "mnist":
        hfds = load_dataset("mnist", split="train")
        X = np.stack(hfds["image"])[..., None]
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    elif dataset == "pokemon":
        hfds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        hfds = hfds.map(
            lambda sample: {"image": sample["image"].resize((64 + 16, 64 + 16))},
            remove_columns=["text"],
            batch_size=96,
        )
        X = np.stack(hfds["image"])
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    elif dataset == "cartoonset":
        hfds = load_dataset("cgarciae/cartoonset", "10k", split="train")
        ds = tf.data.Dataset.from_generator(
            lambda: hfds,
            output_signature={
                "img_bytes": tf.TensorSpec(shape=(), dtype=tf.string),
            },
        )

        def process_fn(x):
            x = tf.image.decode_png(x["img_bytes"], channels=3)
            x = tf.cast(x, tf.float32)
            x = tf.image.resize(x, (128, 128))
            return x

        ds = ds.map(process_fn)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    ds = ds.map(lambda x: x / 127.5 - 1.0)
    ds = ds.repeat()
    ds = ds.shuffle(seed=42, buffer_size=1_000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def visualize_data_samples(x_sample):
    n_rows = 4
    n_cols = 7
    x = x_sample[: n_rows * n_cols]
    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    x = einop(x, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)
    render_image(x)
    show(config, "samples")


ds = get_data(config.dataset, config.batch_size)

# %%

x_sample: np.ndarray = ds.as_numpy_iterator().next()
num_channels = x_sample.shape[-1]

visualize_data_samples(x_sample)


# %%
def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


def visualize_schedule(config: Config, process: GaussianDiffusion):
    n_rows = 2
    n_cols = 7

    _, (ax_img, ax_plot) = plt.subplots(2, 1, figsize=(3 * n_cols, 3 * n_rows))

    t = jnp.linspace(0, config.diffusion.timesteps, n_cols).astype(int)
    x = einop(x_sample[0], "h w c -> b h w c", b=n_cols)
    x, _ = process.forward(jax.random.PRNGKey(0), x, t)
    x = einop(x, "col h w c -> h (col w) c", col=n_cols)
    render_image(x, ax=ax_img)

    linear = schedules.polynomial(
        process.betas.min(),
        process.betas.max(),
        config.diffusion.timesteps,
        exponent=1.0,
    )
    ax_plot.plot(linear, label="linear", color="black", linestyle="dotted")
    ax_plot.plot(process.betas)
    for s in ["top", "bottom", "left", "right"]:
        ax_plot.spines[s].set_visible(False)

    show(config, "betas_schedule")


# %%
if config.diffusion.schedule == "polynomial":
    schedule = schedules.polynomial
elif config.diffusion.schedule == "sigmoid":
    schedule = schedules.sigmoid
elif config.diffusion.schedule == "cosine":
    schedule = schedules.cosine
else:
    raise ValueError(f"Unknown schedule {config.diffusion.schedule}")

betas = schedule(
    config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps
)
process = GaussianDiffusion.create(betas)

visualize_schedule(config, process)

# %%


@struct.dataclass
class Metrics(Collection):
    loss: Average.from_output("loss")
    ema_loss: Average.from_output("ema_loss")

    def update(self, *, loss, ema_loss) -> "Metrics":
        updates = self.single_from_model_output(loss=loss, ema_loss=ema_loss)
        return self.merge(updates)


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray
    ema: EMA
    metrics: Metrics
    process: GaussianDiffusion

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema: EMA, **kwargs):
        return super().create(
            apply_fn=apply_fn, params=params, tx=tx, ema=ema, **kwargs
        )


# %%
if config.model == "stable_unet":
    module = UNet2DModule(
        UNet2DConfig(
            out_channels=num_channels,
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
    )
elif config.model == "lucid_unet":
    module = UNet(dim=64, dim_mults=(1, 2, 4), channels=num_channels)
else:
    raise ValueError(f"Unknown model: '{config.model}'")

variables = module.init(jax.random.PRNGKey(42), x_sample[:1], jnp.array([0]))
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        optax.piecewise_constant_schedule(
            config.optimizer.lr_start,
            {
                int(config.total_steps * 1 / 3): config.optimizer.drop_1_mult,
                int(config.total_steps * 2 / 3): config.optimizer.drop_2_mult,
            },
        )
    ),
)
state: TrainState = TrainState.create(
    apply_fn=module.apply,
    params=variables["params"],
    tx=tx,
    key=jax.random.PRNGKey(0),
    ema=EMA.create(
        params=variables["params"],
        decay=config.ema.decay,
    ),
    metrics=Metrics.empty(),
    process=process,
)


print(module.tabulate(jax.random.PRNGKey(42), x_sample[:1], jnp.array([0]), depth=1))


# %%


@jax.jit
def train_step(state: TrainState, x: jax.Array):
    print("compiling 'train_step' ...")
    key_t, key_diffusion, key = jax.random.split(state.key, 3)
    t = jax.random.uniform(
        key_t, (x.shape[0],), minval=0, maxval=config.diffusion.timesteps - 1
    ).astype(jnp.int32)
    xt, noise = state.process.forward(key_diffusion, x, t)

    def loss_fn(params):
        noise_hat = state.apply_fn({"params": params}, xt, t)
        return loss_metric(noise, noise_hat)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    ema_loss = loss_fn(state.ema.params)
    state = state.apply_gradients(grads=grads)
    metrics = state.metrics.update(loss=loss, ema_loss=ema_loss)
    logs = metrics.compute()
    return logs, state.replace(key=key, metrics=metrics)


def visualize_samples(state: TrainState, x: jax.Array, elapsed: Elapsed):
    print("Sampling...")
    n_rows = 3
    n_cols = 5
    viz_key = jax.random.PRNGKey(1)
    x = jax.random.normal(viz_key, (n_rows * n_cols, *x.shape[1:]))

    ts = np.arange(config.diffusion.timesteps)[::-1]
    xf = state.process.sample(
        viz_key,
        lambda x, t: state.apply_fn({"params": state.params}, x, t),
        x,
        ts,
        return_all=False,
    )
    xf = np.asarray(xf)
    xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    render_image(xf)
    show(config, "training_samples", step=elapsed.steps)


def reset_metrics(state: TrainState, x):
    return None, state.replace(metrics=Metrics.empty())


def ema_update(state: TrainState, batch, elapsed: Elapsed):
    ema = state.ema.update(state.params, elapsed.steps)
    return None, state.replace(ema=ema)


# %%

state, history, _ = ciclo.loop(
    state,
    ds.as_numpy_iterator(),
    {
        ciclo.every(1): [train_step],
        ciclo.every(config.ema.update_every): [ema_update],
        ciclo.every(config.eval_every): [visualize_samples],
        ciclo.every(config.log_every): [
            ciclo.wandb_logger(wandb_run),
            ciclo.checkpoint("logdir/mnist_full", monitor="ema_loss", mode="min"),
            reset_metrics,
        ],
        ciclo.every(1): [ciclo.keras_bar(total=ciclo.at(steps=config.total_steps))],
    },
    on_end=[visualize_samples],
    stop=ciclo.at(steps=config.total_steps),
)
