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
from typing import Any, Callable, Union, Tuple, Optional

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
from ciclo import managed


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


class DebugModule(nn.Module):
    units: int

    @nn.compact
    def __call__(self, x, t):
        return nn.Conv(self.units, (3, 3), padding="SAME")(x)


def get_module(config: Config, num_channels: int):
    if config.debug:
        return DebugModule(num_channels)
    elif config.model == "stable_unet":
        config.stable_diffusion_unet.out_channels = num_channels
        return UNet2DModule(config.stable_diffusion_unet)
    elif config.model == "lucid_unet":
        return UNet(dim=64, dim_mults=(1, 2, 4), channels=num_channels)
    else:
        raise ValueError(f"Unknown model: '{config.model}'")


class State(managed.ManagedState):
    key: jax.random.KeyArray
    ema_params: Any
    process: GaussianDiffusion
    config: Config = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        ema_params = jax.tree_map(jnp.copy, params)
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            ema_params=ema_params,
            **kwargs,
        )


@managed.train_step
def train_step(state: State, x: jax.Array):
    print("Compiling train_step...")
    config = state.config
    key = jax.random.fold_in(state.key, state.step)
    t = jax.random.uniform(
        key, (x.shape[0],), minval=0, maxval=config.diffusion.timesteps - 1
    ).astype(jnp.int32)
    xt, noise = state.process.forward(key=key, x=x, t=t)

    def loss_fn(params):
        noise_hat = state.apply_fn({"params": params}, xt, t)
        return state.loss_fn(noise, noise_hat)

    loss = loss_fn(state.params)
    ema_loss = loss_fn(state.ema_params)

    # get logs
    logs = ciclo.logs()
    logs.add_loss("loss", loss)
    logs.add_metric("loss", loss)
    logs.add_metric("ema_loss", ema_loss)

    return logs


@managed.step
def reverse_sample(
    state: State, batch: Tuple[jax.Array, jax.Array, jax.random.KeyArray]
):
    print("Compiling reverse_sample...")
    x, t, key = batch
    key = key[0]
    t = t[0]
    process = state.process
    xf = process.sample(
        key=key, model_fn=model_forward, state=state, x=x, t=t, return_all=False
    )
    logs = ciclo.logs()
    logs.add_output("samples", xf)
    return logs


def model_forward(state: State, x: jax.Array, t: jax.Array) -> jax.Array:
    return state.apply_fn({"params": state.ema_params}, x, t)


def viz_model_samples(state: State, x: jax.Array, elapsed: Elapsed):
    print("\nSampling ...")
    config = state.config

    n_rows = 3
    n_cols = 8
    batch_size = n_rows * n_cols
    # create inputs
    viz_key = jax.random.PRNGKey(1)
    viz_key = einop(viz_key, "t -> b t", b=batch_size)
    ts = np.arange(config.diffusion.timesteps)[::-1]
    ts = einop(ts, "t -> b t", b=batch_size)
    x = jax.random.normal(jax.random.PRNGKey(2), (batch_size, *x.shape[1:]))

    logs, state = reverse_sample(state, (x, ts, viz_key))
    xf = logs["per_sample_outputs"]["samples"]
    xf = np.asarray(xf)
    xf = einop(xf, " (row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    render_image(xf)
    show(config, "training_samples", step=elapsed.steps)

    return state


@managed.step
def ema_update_step(state: State, _, decay):
    print("Compiling ema_update_step...")
    ema_params = ema_update(decay, state.ema_params, state.params)
    return state.replace(ema_params=ema_params)


@managed.step
def ema_copy_step(state: State):
    print("Compiling ema_copy_step...")
    ema_params = jax.tree_map(jnp.copy, state.params)
    return state.replace(ema_params=ema_params)


@dataclass
class ema_step(ciclo.LoopElement):
    decay: Union[float, Callable[[int], float]] = 0.995
    update_every: int = 10
    update_after_step: int = 100

    def __call__(self, state: State, _, elapsed: Elapsed):
        if elapsed.steps < self.update_after_step:
            return ema_copy_step(state)
        steps = elapsed.steps - self.update_after_step
        if steps >= 0 and steps % self.update_every == 0:
            decay = self.decay(elapsed.steps) if callable(self.decay) else self.decay
            return ema_update_step(state, None, decay)


def create_state(
    *,
    config: Config,
    module: nn.Module,
    process: GaussianDiffusion,
    loss_fn,
    x,
    strategy: str = "data_parallel_donate",
) -> State:
    variables = module.init(jax.random.PRNGKey(42), x[:1], jnp.array([0]))
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
    return State.create(
        apply_fn=module.apply,
        params=variables["params"],
        tx=tx,
        key=jax.random.PRNGKey(0),
        process=process,
        config=config,
        loss_fn=loss_fn,
        strategy=strategy,
    )


def create_checkpoint(*args, **kwargs):
    checkpoint = ciclo.checkpoint(*args, **kwargs)

    def checkpoint_fn(state: State, batch, elapsed, loop: ciclo.LoopState):
        logs = loop.logs
        state = state.with_strategy("eager")
        checkpoint(elapsed, state, logs)

    return checkpoint_fn


# %%


def main(_):
    # %%
    try:
        config: Config = flags.FLAGS.config
    except AttributeError:
        config = Config(debug=True)

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
    state = create_state(
        config=config, module=module, process=process, loss_fn=loss_fn, x=x_sample
    )

    print(
        module.tabulate(jax.random.PRNGKey(42), x_sample[:1], jnp.array([0]), depth=1)
    )

    # %%
    state, history, elapsed = ciclo.loop(
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
            ciclo.every(config.checkpoint_every, steps_offset=1): [
                create_checkpoint(
                    f"logdir/{run_id}",
                    monitor="ema_loss",
                    mode="min",
                    async_manager=AsyncManager(),
                ),
            ],
            **(ciclo.wandb_logger(wandb_run) if wandb_run is not None else ciclo.noop),
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
    steps, ema_loss, loss = history.collect("steps", "ema_loss", "metrics.loss")
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
