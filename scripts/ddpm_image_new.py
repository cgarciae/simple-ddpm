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
from time import time
from typing import Any, Callable, Union

import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import wandb
from absl import app, flags, logging
from ciclo import Elapsed, LoopState
from clu.metrics import Average, Collection
from configs.image.base import Config
from datasets.load import load_dataset
from einop import einop
from flax import struct
from flax.training import train_state
from flax.training.checkpoints import AsyncManager
from ml_collections import config_flags
from simple_ddpm import schedules
from simple_ddpm.models.unet_lucid import UNet
from simple_ddpm.models.unet_stable import UNet2DConfig, UNet2DModule
from simple_ddpm.processes import GaussianDiffusion
from simple_ddpm.utils import (
    ema_update,
    get_wandb_run,
    parse_config,
    print_compiling,
    render_image,
    show,
)


def get_data(config: Config):
    dataset = config.dataset

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
    ds = ds.batch(config.batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def visualize_data(config, x):
    n_rows = 4
    n_cols = 7
    x = x[: n_rows * n_cols]
    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    x = einop(x, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)
    render_image(x)
    show(config, "samples")


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


def visualize_schedule(config: Config, process: GaussianDiffusion, x_sample):
    n_rows = 2
    n_cols = 7

    _, (ax_img, ax_plot) = plt.subplots(2, 1, figsize=(3 * n_cols, 3 * n_rows))

    t = jnp.linspace(0, config.diffusion.timesteps, n_cols).astype(int)
    x = einop(x_sample[0], "h w c -> b h w c", b=n_cols)
    x, _ = process.forward(key=jax.random.PRNGKey(0), x=x, t=t)
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


def get_module(config: Config, num_channels: int):
    if config.model == "stable_unet":
        config.stable_diffusion_unet.out_channels = num_channels
        return UNet2DModule(config.stable_diffusion_unet)
    elif config.model == "lucid_unet":
        return UNet(dim=64, dim_mults=(1, 2, 4), channels=num_channels)
    else:
        raise ValueError(f"Unknown model: '{config.model}'")


@struct.dataclass
class Metrics(Collection):
    loss: Average.from_output("loss")
    ema_loss: Average.from_output("ema_loss")

    def update(self, *, loss, ema_loss) -> "Metrics":
        updates = self.single_from_model_output(loss=loss, ema_loss=ema_loss)
        return self.merge(updates)


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray
    ema_params: Any
    metrics: Metrics
    process: GaussianDiffusion
    config: Config = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        return super().create(
            apply_fn=apply_fn, params=params, tx=tx, ema_params=params, **kwargs
        )


@jax.jit
@print_compiling
def train_step(state: TrainState, x: jax.Array):
    config = state.config
    key_t, key_diffusion, key = jax.random.split(state.key, 3)
    t = jax.random.uniform(
        key_t, (x.shape[0],), minval=0, maxval=config.diffusion.timesteps - 1
    ).astype(jnp.int32)
    xt, noise = state.process.forward(key=key_diffusion, x=x, t=t)

    def loss_fn(params):
        noise_hat = state.apply_fn({"params": params}, xt, t)
        return state.loss_fn(noise, noise_hat)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    ema_loss = loss_fn(state.ema_params)
    state = state.apply_gradients(grads=grads)
    metrics = state.metrics.update(loss=loss, ema_loss=ema_loss)
    return None, state.replace(key=key, metrics=metrics)


@jax.jit
@print_compiling
def model_forward(state: TrainState, x: jax.Array, t: jax.Array) -> jax.Array:
    return state.apply_fn({"params": state.ema_params}, x, t)


def compute_metrics(state: TrainState):
    logs = state.metrics.compute()
    state = state.replace(metrics=Metrics.empty())
    return logs, state


def viz_model_samples(state: TrainState, x: jax.Array, elapsed: Elapsed):
    print("\nSampling ...")
    config = state.config
    process = state.process

    n_rows = 3
    n_cols = 6
    viz_key = jax.random.PRNGKey(1)
    x = jax.random.normal(viz_key, (n_rows * n_cols, *x.shape[1:]))

    ts = np.arange(config.diffusion.timesteps)[::-1]
    xf = process.sample(
        key=viz_key, model_fn=model_forward, state=state, x=x, t=ts, return_all=False
    )
    xf = np.asarray(xf)
    xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    render_image(xf)
    show(config, "training_samples", step=elapsed.steps)


@dataclass
class ema_step(ciclo.CallbackBase):
    decay: Union[float, Callable[[int], float]] = 0.995
    update_every: int = 10
    update_after_step: int = 100

    def __call__(self, state: TrainState, batch, elapsed: Elapsed, loop):
        if elapsed.steps < self.update_after_step:
            state = state.replace(ema_params=state.params)
            return None, state
        steps = elapsed.steps - self.update_after_step
        if steps >= 0 and steps % self.update_every == 0:
            decay = self.decay(elapsed.steps) if callable(self.decay) else self.decay
            ema_params = ema_update(decay, state.ema_params, state.params)
            return None, state.replace(ema_params=ema_params)


# %%


def main(_):
    # %%
    try:
        config: Config = flags.FLAGS.config
    except AttributeError:
        config = Config()

    # %%
    print(jax.devices())

    if config.viz == "wandb":
        wandb_run = get_wandb_run(config)
        run_id = wandb_run.id
    else:
        wandb_run = None
        run_id = str(int(time()))

    if config.diffusion.schedule == "polynomial":
        schedule = schedules.polynomial
    elif config.diffusion.schedule == "sigmoid":
        schedule = schedules.sigmoid
    elif config.diffusion.schedule == "cosine":
        schedule = schedules.cosine
    else:
        raise ValueError(f"Unknown schedule {config.diffusion.schedule}")

    if config.loss_type == "mse":
        loss_fn = lambda a, b: jnp.mean((a - b) ** 2)
    elif config.loss_type == "mae":
        loss_fn = lambda a, b: jnp.mean(jnp.abs(a - b))
    else:
        raise ValueError(f"Unknown loss type {config.loss_type}")

    # %%

    ds = get_data(config)
    x_sample: np.ndarray = ds.as_numpy_iterator().next()
    num_channels = x_sample.shape[-1]

    visualize_data(config, x_sample)

    # %%

    betas = schedule(
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.timesteps,
    )
    process = GaussianDiffusion.create(betas)

    visualize_schedule(config, process, x_sample)

    # %%

    module = get_module(config, num_channels)
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
        metrics=Metrics.empty(),
        process=process,
        config=config,
        loss_fn=loss_fn,
    )

    print(
        module.tabulate(jax.random.PRNGKey(42), x_sample[:1], jnp.array([0]), depth=1)
    )

    # %%
    state, history, _ = ciclo.loop(
        state,
        ds.as_numpy_iterator(),
        {
            ciclo.every(1): train_step,
            **ema_step(
                decay=config.ema.decay,
                update_every=config.ema.update_every,
                update_after_step=config.ema.update_after_step,
            ),
            ciclo.every(config.viz_progress_every, steps_offset=1): viz_model_samples,
            ciclo.every(config.log_every, steps_offset=1): [
                compute_metrics,
                ciclo.wandb_logger(wandb_run) if config.viz == "wandb" else ciclo.noop,
                ciclo.checkpoint(
                    f"logdir/{run_id}",
                    monitor="ema_loss",
                    mode="min",
                    async_manager=AsyncManager(),
                ),
            ],
            **ciclo.keras_bar(
                total=ciclo.at(steps=config.total_steps),
                always_stateful=True,
                interval=0.5,
            ),
        },
        on_end=[viz_model_samples],
        stop=ciclo.at(steps=config.total_steps),
    )
    # %%

    # ------------------
    # visualizations
    # ------------------
    # loss
    plt.figure()
    steps, ema_loss, loss = history["steps", "ema_loss", "loss"]
    ema_loss = np.asarray(ema_loss)
    plt.plot(steps, ema_loss, label="ema_loss")
    plt.plot(steps, loss, label="loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    show(config, "loss curve")

    # %%


# --------------------
# entry point
# --------------------
logging.set_verbosity("warning")

config_flags.DEFINE_config_file(
    "config", "configs/image/base.py", "Training configuration."
)

if __name__ == "__main__":
    app.run(main)

# %%
