# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

from dataclasses import dataclass

import jax
from IPython import get_ipython

from utils import setup_config


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
class ModelConfig:
    units: int = 128
    emb_dim: int = 32


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 10
    total_samples: int = 2_000_000
    lr: float = 1e-3
    loss_type: str = "mae"
    dataset: str = "moons"
    viz: str = "matplotlib"
    eval_every: int = 2000
    log_every: int = 200
    ema: EMAConfig = EMAConfig()
    schedule: DiffusionConfig = DiffusionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.total_samples // self.batch_size


config = setup_config(Config)

print(jax.devices())


# %%

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler
from utils import show
import tensorflow as tf
import numpy as np


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


# %%

X, ds = get_data(config)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=1)
show("samples")


# %%
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


class GaussianDiffusion(PyTreeNode):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray

    @classmethod
    def create(cls, betas: jnp.ndarray) -> "GaussianDiffusion":
        return cls(
            betas=betas,
            alphas=1.0 - betas,
            alpha_bars=jnp.cumprod(1.0 - betas),
        )


@jax.jit
def forward_diffusion(process: GaussianDiffusion, key, x0, t):
    alpha_bars = expand_to(process.alpha_bars[t], x0)
    noise = jax.random.normal(key, x0.shape)
    xt = jnp.sqrt(alpha_bars) * x0 + jnp.sqrt(1.0 - alpha_bars) * noise
    return xt, noise


# %%
def polynomial_schedule(beta_start, beta_end, timesteps, exponent=2.0, **kwargs):
    betas = jnp.linspace(0, 1, timesteps) ** exponent
    return betas * (beta_end - beta_start) + beta_start


def sigmoid_schedule(beta_start, beta_end, timesteps, **kwargs):
    betas = jax.nn.sigmoid(jnp.linspace(-6, 6, timesteps))
    return betas * (beta_end - beta_start) + beta_start


def cosine_schedule(beta_start, beta_end, timesteps, s=0.008, **kwargs):
    x = jnp.linspace(0, timesteps, timesteps + 1)
    ft = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, 0.0001, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


# TODO: create a plot for each schedule

# %%
if config.diffusion.schedule == "polynomial":
    schedule = polynomial_schedule
elif config.diffusion.schedule == "sigmoid":
    schedule = sigmoid_schedule
elif config.diffusion.schedule == "cosine":
    schedule = cosine_schedule
else:
    raise ValueError(f"Unknown schedule {config.diffusion.schedule}")

betas = schedule(
    config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps
)
process = GaussianDiffusion.create(betas)
n_rows = 2
n_cols = 7

plt.figure(figsize=(n_cols * 3, n_rows * 3))
for i, ti in enumerate(jnp.linspace(0, config.diffusion.timesteps, n_cols).astype(int)):
    t = jnp.full((X.shape[0],), ti)
    xt, noise = forward_diffusion(process, jax.random.PRNGKey(ti), X, t)
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(xt[:, 0], xt[:, 1], s=1)
    plt.axis("off")

plt.subplot(2, 1, 2)
linear = polynomial_schedule(
    betas.min(), betas.max(), config.diffusion.timesteps, exponent=1.0
)
plt.plot(linear, label="linear", color="black", linestyle="dotted")
plt.plot(betas)
for s in ["top", "bottom", "left", "right"]:
    plt.gca().spines[s].set_visible(False)

show("betas_schedule")


# %%

from flax.training import train_state

from utils import EMA


class TrainState(train_state.TrainState):
    ema: EMA

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema: EMA, **kwargs):
        return super().create(
            apply_fn=apply_fn, params=params, tx=tx, ema=ema, **kwargs
        )

    def ema_update(self, step: int) -> "TrainState":
        ema = self.ema.update(step, self.params)
        return self.replace(ema=ema)


# %%
import flax.linen as nn


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
    def __call__(self, x, t):
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
        x = nn.relu(dense(self.units)(x, t))
        x = nn.relu(dense(self.units)(x, t)) + x
        x = nn.relu(dense(self.units)(x, t)) + x
        x = dense(inputs_units)(x, t)
        return x


# %%
import optax
from jax_metrics.metrics import Mean, Metrics

module = Denoiser(units=config.model.units, emb_dim=config.model.emb_dim)
variables = module.init(jax.random.PRNGKey(42), X[:1], jnp.array([0]))
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        optax.piecewise_constant_schedule(
            config.lr,
            {
                int(config.total_steps / 3): 1 / 3,
                int(config.total_steps * 2 / 3): 3 / 10,
            },
        )
    ),
)
state: TrainState = TrainState.create(
    apply_fn=module.apply,
    params=variables["params"],
    tx=tx,
    ema=EMA.create(
        params=variables["params"],
        decay=config.ema.decay,
        update_every=config.ema.update_every,
        update_after_step=config.ema.update_after_step,
    ),
)
metrics = Metrics(
    [
        Mean(name="loss").map_arg(loss="values"),
        Mean(name="ema_loss").map_arg(ema_loss="values"),
    ]
).init()

print(module.tabulate(jax.random.PRNGKey(42), X[:1], jnp.array([0]), depth=1))

# %%
from functools import partial
from einop import einop


@jax.jit
def reverse_diffusion(process: GaussianDiffusion, key, x, noise_hat, t):
    betas = expand_to(process.betas[t], x)
    alphas = expand_to(process.alphas[t], x)
    alpha_bars = expand_to(process.alpha_bars[t], x)

    z = jnp.where(
        expand_to(t, x) > 0, jax.random.normal(key, x.shape), jnp.zeros_like(x)
    )
    noise_scaled = betas / jnp.sqrt(1.0 - alpha_bars) * noise_hat
    x = (x - noise_scaled) / jnp.sqrt(alphas) + jnp.sqrt(betas) * z
    return x


@partial(jax.jit, static_argnames=["return_all"])
def sample(key, x0, ts, params, process, *, return_all=True):
    print("compiling 'sample' ...")
    keys = jax.random.split(key, len(ts))
    ts = einop(ts, "t -> t b", b=x0.shape[0])

    def scan_fn(x, inputs):
        t, key = inputs
        noise_hat = module.apply({"params": params}, x, t)
        x = reverse_diffusion(process, key, x, noise_hat, t)
        out = x if return_all else None
        return x, out

    x, xs = jax.lax.scan(scan_fn, x0, (ts, keys))
    return xs if return_all else x


# %%

if config.loss_type == "mse":
    loss_metric = lambda a, b: jnp.mean((a - b) ** 2)
elif config.loss_type == "mae":
    loss_metric = lambda a, b: jnp.mean(jnp.abs(a - b))
else:
    raise ValueError(f"Unknown loss type {config.loss_type}")


def loss_fn(params, xt, t, noise):
    noise_hat = state.apply_fn({"params": params}, xt, t)
    return loss_metric(noise, noise_hat)


@jax.jit
def train_step(key, x, state: TrainState, metrics: Metrics, process: GaussianDiffusion):
    print("compiling 'train_step' ...")
    key_t, key_diffusion, key = jax.random.split(key, 3)
    t = jax.random.uniform(
        key_t, (x.shape[0],), minval=0, maxval=config.diffusion.timesteps - 1
    ).astype(jnp.int32)
    xt, noise = forward_diffusion(process, key_diffusion, x, t)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, xt, t, noise)
    ema_loss = loss_fn(state.ema.params, xt, t, noise)
    state = state.apply_gradients(grads=grads)
    metrics = metrics.update(loss=loss, ema_loss=ema_loss)
    logs = metrics.compute()
    return logs, key, state, metrics


# %%
import numpy as np
from tqdm import tqdm

from utils import log_metrics

key = jax.random.PRNGKey(42)
axs_diffusion = None
axs_samples = None
ds_iterator = ds.as_numpy_iterator()
logs = {}
step = 0
history = []

# %%

for step in tqdm(
    range(step, config.total_steps), total=config.total_steps, unit="step"
):

    if step % config.steps_per_epoch == 0:
        # --------------------
        # visualize progress
        # --------------------
        n_cols = 7
        n_samples = 1000
        viz_key = jax.random.PRNGKey(1)
        x = jax.random.normal(viz_key, (n_samples, *X.shape[1:]))

        ts = np.arange(config.diffusion.timesteps)[::-1]
        xs = np.asarray(
            sample(viz_key, x, ts, state.ema.params, process, return_all=True)
        )
        if axs_diffusion is None or get_ipython():
            _, axs_diffusion = plt.subplots(1, n_cols, figsize=(n_cols * 3, 3))

        ts = jnp.linspace(0, config.diffusion.timesteps - 1, n_cols).astype(int)
        for i, ti in enumerate(ts):
            axs_diffusion[i].clear()
            axs_diffusion[i].scatter(xs[ti, :, 0], xs[ti, :, 1], s=1)
            axs_diffusion[i].axis("off")
        show("training_samples", step=step)

    if step % config.log_every == 0 and logs != {}:
        log_metrics(logs, step, do_print=False)
        history.append(logs)
        metrics = metrics.reset()

    # --------------------
    # trainig step
    # --------------------
    x = ds_iterator.next()
    logs, key, state, metrics = train_step(key, x, state, metrics, process)
    state = state.ema_update(step)
    logs["step"] = step

# %%
# plot history
plt.figure(figsize=(10, 5))
steps = np.array([h["step"] for h in history])
plt.plot(steps, [h["loss"] for h in history], label="loss")
plt.plot(steps, [h["ema_loss"] for h in history], label="ema_loss")
plt.legend()
plt.show()

# %%
n_rows = 3
n_cols = 5
viz_key = jax.random.PRNGKey(1)
x = jax.random.normal(viz_key, (n_rows * n_cols, *x_sample.shape[1:]))

ts = np.arange(config.diffusion.timesteps)[::-1]
xf = np.asarray(sample(viz_key, x, ts, state.ema.params, process, return_all=False))
xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

plt.figure(figsize=(3 * n_cols, 3 * n_rows))
render_image(xf)
show("final_samples")

# %%
