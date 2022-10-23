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
from absl import app, flags, logging
from ciclo import Elapsed, LoopState
from clu.metrics import Average, Collection
from configs._2d.base import Config
from einop import einop
from flax import struct
from flax.training import train_state
from flax.training.checkpoints import AsyncManager
from ml_collections import config_flags
from simple_ddpm import schedules
from simple_ddpm.processes import GaussianDiffusion
from simple_ddpm.utils import (
    ema_update,
    get_wandb_run,
    parse_config,
    render_image,
    show,
)
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler

import wandb


def get_data(config: Config):
    if config.dataset == "moons":
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
    elif config.dataset == "blobs":
        X = make_blobs(n_samples=1000, centers=6, cluster_std=0.5, random_state=6)[0]
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    X = MinMaxScaler((-1, 1)).fit_transform(X)
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.repeat()
    ds = ds.shuffle(seed=42, buffer_size=1_000)
    ds = ds.batch(config.batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return X, ds


def visualize_data(config, x):
    plt.figure()
    plt.axis("off")
    plt.scatter(x[:, 0], x[:, 1], s=1)
    show(config, "samples")


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


def visualize_schedule(config: Config, process: GaussianDiffusion, X):
    betas = process.betas
    timesteps = len(betas)
    n_rows = 2
    n_cols = 7

    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for i, ti in enumerate(jnp.linspace(0, timesteps, n_cols).astype(int)):
        t = jnp.full((X.shape[0],), ti)
        xt, noise = process.forward(key=jax.random.PRNGKey(ti), x=X, t=t)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(xt[:, 0], xt[:, 1], s=1)
        plt.axis("off")

    plt.subplot(2, 1, 2)
    linear = schedules.polynomial(betas.min(), betas.max(), timesteps, exponent=1.0)
    plt.plot(linear, label="linear", color="black", linestyle="dotted")
    plt.plot(betas)
    for s in ["top", "bottom", "left", "right"]:
        plt.gca().spines[s].set_visible(False)

    show(config, "betas_schedule")


class SinusoidalPosEmb(nn.Module):
    dim: int

    def __call__(self, t):
        half_dim = self.dim // 2
        mul = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(-mul * jnp.arange(half_dim))
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeConditionedDense(nn.Module):
    units: int
    emb_dim: int

    @nn.compact
    def __call__(self, *, x, t):
        t_embeddings = SinusoidalPosEmb(self.emb_dim)(t)
        x = jnp.concatenate([x, t_embeddings], axis=-1)
        x = nn.Dense(self.units)(x)
        return x


class Denoiser(nn.Module):
    units: int = 128
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x, t):
        inputs_units = x.shape[-1]
        dense = lambda units: TimeConditionedDense(units, self.emb_dim)
        x = nn.relu(dense(self.units)(x=x, t=t))
        x = nn.relu(dense(self.units)(x=x, t=t)) + x
        x = nn.relu(dense(self.units)(x=x, t=t)) + x
        x = dense(inputs_units)(x=x, t=t)
        return x


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
def train_step(state: TrainState, x: jax.Array):
    print("compiling 'train_step' ...")
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
def forward(state: TrainState, x: jax.Array, t: jax.Array) -> jax.Array:
    return state.apply_fn({"params": state.ema_params}, x, t)


def compute_metrics(state: TrainState):
    logs = state.metrics.compute()
    state = state.replace(metrics=Metrics.empty())
    return logs, state


def visualize_model_samples(state: TrainState, x: jax.Array, elapsed: Elapsed):
    config = state.config
    process = state.process

    n_cols = 7
    n_samples = 1000
    viz_key = jax.random.PRNGKey(1)
    x = jax.random.normal(viz_key, (n_samples, *x.shape[1:]))

    ts = np.arange(config.diffusion.timesteps)[::-1]
    xs = process.sample(
        key=viz_key, model_fn=forward, state=state, x=x, t=ts, return_all=True
    )
    xs = np.asarray(xs)
    _, axs_diffusion = plt.subplots(1, n_cols, figsize=(n_cols * 3, 3))

    ts = jnp.linspace(0, config.diffusion.timesteps - 1, n_cols).astype(int)
    for i, ti in enumerate(ts):
        axs_diffusion[i].clear()
        axs_diffusion[i].scatter(xs[ti, :, 0], xs[ti, :, 1], s=1)
        axs_diffusion[i].axis("off")
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


def compare_with_data(config: Config, state: TrainState, X: np.ndarray):
    n_samples = 1000
    viz_key = jax.random.PRNGKey(1)
    process = state.process

    xt = jax.random.normal(viz_key, (n_samples, *X.shape[1:]))
    ts = np.arange(config.diffusion.timesteps)[::-1]
    x0 = np.asarray(
        process.sample(key=viz_key, model_fn=forward, state=state, x=xt, t=ts)
    )

    # plot x and X side by side
    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(x0[:, 0], x0[:, 1], s=1)
    axs[0].axis("off")
    axs[0].set_title("model")
    axs[1].scatter(X[:, 0], X[:, 1], s=1)
    axs[1].axis("off")
    axs[1].set_title("data")


def plot_trajectory_2d(
    config: Config,
    state: TrainState,
    *,
    fig,
    interval: int = 10,
    repeat_delay: int = 1000,
    step_size: int = 1,
    end_pad: int = 500,
):
    from base64 import b64encode
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from IPython.display import HTML, display
    from matplotlib import animation

    process = state.process
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (1000, 2), minval=-1, maxval=1)
    ts = jnp.arange(config.diffusion.timesteps, 0, -1)
    xs = process.sample(
        key=key, model_fn=forward, state=state, x=x, t=ts, return_all=True
    )
    xs = np.asarray(xs)

    xs = np.concatenate([xs[::step_size], xs[-1:]], axis=0)
    # replace last sample to create a 'pause' effect
    pad_end = einop(xs[-1], "... -> batch ...", batch=end_pad)
    xs = np.concatenate([xs, pad_end], axis=0)
    N = len(xs)

    plt.axis("off")
    scatter = plt.scatter(xs[0][:, 0], xs[0][:, 1], s=1)

    def animate(i):
        scatter.set_offsets(xs[i])
        return [scatter]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: animate(0),
        frames=np.linspace(0, N - 1, N, dtype=int),
        interval=interval,
        repeat_delay=repeat_delay,
        blit=True,
    )

    with TemporaryDirectory() as tmpdir:
        img_name = Path(tmpdir) / f"diffusion.gif"
        anim.save(str(img_name), writer="pillow", fps=60)

        if config.viz == "wandb":
            wandb.log({"trajectory_2d": wandb.Video(str(img_name))})
        else:
            image_bytes = b64encode(img_name.read_bytes()).decode("utf-8")
            plt.axis("off")
            display(HTML(f"""<img src='data:image/gif;base64,{image_bytes}'>"""))
            plt.close()
            plt.show()


def plot_gradient_field(state: TrainState, X, *, alpha=0.9):
    # create a meshgrid between -1 and 1
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 40), np.linspace(-1.2, 1.2, 40))
    x = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    t = np.array(0, dtype=jnp.int32)
    t = einop(t, "-> batch", batch=len(x))

    # get predicted noise predictions, gradient is negative noise
    grad = -forward(state, x, t)

    grad_norm = np.linalg.norm(grad, axis=-1, ord=2, keepdims=True)
    grad_log1p = grad / (grad_norm + 1e-9) * np.log1p(grad_norm)

    plt.axis("off")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, s=40)
    plt.quiver(*x.T, *grad_log1p.T, width=0.002, alpha=alpha)


def plot_density(state: TrainState, X, ts):
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 40), np.linspace(-1.2, 1.2, 40))
    x = jnp.stack([xx, yy], axis=-1)

    def mass_fn(x, t):
        t_ = jnp.full((1,), t)
        x_ = x[None]
        noise_hat = forward(state, x_, t_)
        magnitud = jnp.linalg.norm(noise_hat, axis=-1, keepdims=False)
        mass = jnp.exp(-magnitud)
        return mass[0]

    mass_fn = jax.jit(
        jax.vmap(
            jax.vmap(jax.vmap(mass_fn, in_axes=(0, None)), in_axes=(0, None)),
            in_axes=(None, 0),
            out_axes=-1,
        )
    )
    mass = mass_fn(x, ts).mean(axis=-1)
    plt.axis("off")
    plt.contourf(xx, yy, mass, levels=100, cmap="Blues")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c="black", s=5)


def main(_):
    try:
        config: Config = flags.FLAGS.config
    except AttributeError:
        config = Config()

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

    module = Denoiser(
        units=config.model.units,
        emb_dim=config.model.emb_dim,
    )

    X, ds = get_data(config)

    visualize_data(config, X)

    betas = schedule(
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.timesteps,
    )
    process = GaussianDiffusion.create(betas)

    visualize_schedule(config, process, X)

    variables = module.init(jax.random.PRNGKey(42), X[:1], jnp.array([0]))
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

    print(module.tabulate(jax.random.PRNGKey(42), X[:1], jnp.array([0]), depth=1))

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
            ciclo.every(
                config.steps_per_epoch, steps_offset=1
            ): visualize_model_samples,
            ciclo.every(config.log_every): [
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
        on_end=[visualize_model_samples],
        stop=ciclo.at(steps=config.total_steps),
    )
    # ------------------
    # visualizations
    # ------------------
    # loss
    plt.figure()
    steps, ema_loss = history["steps", "ema_loss"]
    ema_loss = np.asarray(ema_loss)
    plt.plot(steps, ema_loss)
    plt.xlabel("steps")
    plt.ylabel("ema_loss")
    show(config, "ema_loss")

    # compare_with_data
    plt.figure()
    compare_with_data(config, state, X)
    show(config, "compare_with_data")

    # plot_trajectory_2d
    fig = plt.figure()
    plot_trajectory_2d(config, state, fig=fig, step_size=4)

    # plot_gradient_field
    fig = plt.figure()
    plot_gradient_field(state, X)
    show(config, "gradient_field")

    # plot_density
    fig = plt.figure()
    ts = np.array([0, 1, 10], dtype=np.int32)
    plot_density(state, X, ts)
    show(config, "density")


# --------------------
# entry point
# --------------------
logging.set_verbosity("warning")

config_flags.DEFINE_config_file(
    "config", "configs/_2d/base.py", "Training configuration."
)

if __name__ == "__main__":
    app.run(main)

# %%
