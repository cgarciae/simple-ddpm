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
#     display_name: 'Python 3.8.11 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
from dataclasses import dataclass
from imp import reload
from typing import List, Optional

from IPython import get_ipython

from ddpm_utils import update_config_from_args


@dataclass
class Config:
    dataset: str = "moons"
    batch_size: int = 32
    epochs: int = 10
    total_samples: int = 5_000_000
    lr: float = 1e-3
    num_steps: int = 50
    schedule_exponent: float = 2.0

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)


config = Config()

if not get_ipython():
    config = update_config_from_args(config)

# %%
import elegy as eg
import jax
import jax.numpy as jnp
import numpy as np


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


class GaussianDiffusion(eg.Pytree):
    betas: jnp.ndarray
    alpha_bars: jnp.ndarray

    def __init__(self, betas: jnp.ndarray) -> None:
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = jnp.cumprod(self.alphas)

    def get_time_index(self, t: jnp.ndarray) -> jnp.ndarray:
        return (t / len(self.betas)).astype(jnp.int32)

    def forward_diffusion(self, key, x0, t):
        alpha_bars = expand_to(self.alpha_bars[t], x0)
        noise = jax.random.normal(key, x0.shape)
        xt = jnp.sqrt(alpha_bars) * x0 + jnp.sqrt(1.0 - alpha_bars) * noise
        return xt, noise

    def backward_diffusion(self, key, x, pred_noise, t):
        betas = expand_to(self.betas[t], x)
        alphas = expand_to(self.alphas[t], x)
        alpha_bars = expand_to(self.alpha_bars[t], x)

        sampling_noise = jnp.sqrt(betas) * jax.random.normal(key, x.shape)
        pred_noise = betas / jnp.sqrt(1.0 - alpha_bars) * pred_noise
        x = (x - pred_noise) / jnp.sqrt(alphas)

        return jnp.where(t[:, None] == 0, x, x + sampling_noise)


# %%
def get_data(dataset: str = "moons"):
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.preprocessing import MinMaxScaler

    if dataset == "moons":
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
    elif dataset == "blobs":
        X = make_blobs(n_samples=1000, centers=6, cluster_std=0.5, random_state=6)[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X = MinMaxScaler((-1, 1)).fit_transform(X)
    return X


# %%
import matplotlib.pyplot as plt

from ddpm_utils import show_interactive

X = get_data(config.dataset)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=1)
show_interactive()

# %%
def polynomial_beta_schedule(beta_start, beta_end, timesteps, exponent=2.0):
    betas = jnp.linspace(0, 1, timesteps) ** exponent
    return betas * (beta_end - beta_start) + beta_start


# %%
betas = polynomial_beta_schedule(
    0.0001, 0.01, config.num_steps, exponent=config.schedule_exponent
)
diffusion = GaussianDiffusion(betas)

_, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, ti in enumerate(jnp.linspace(0, config.num_steps, 5).astype(int)):
    t = jnp.full((X.shape[0],), ti)
    xt, noise = diffusion.forward_diffusion(jax.random.PRNGKey(ti), X, t)
    axs[i].scatter(xt[:, 0], xt[:, 1], s=1)

show_interactive()

plt.figure(figsize=(15, 6))
plt.plot(betas)

show_interactive()

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
        x = TimeConditionedDense(self.units, self.emb_dim)(x, t)
        x = nn.relu(x)
        x = TimeConditionedDense(self.units, self.emb_dim)(x, t)
        x = nn.relu(x)
        x = TimeConditionedDense(inputs_units, self.emb_dim)(x, t)
        return x


# %%
import optax
from flax.training.train_state import TrainState
from matplotlib.axes import Axes


@dataclass
class Static:
    axes: Optional[List[Axes]] = None


@dataclass
class DDPM(eg.CoreModule):
    diffusion: GaussianDiffusion
    state: TrainState
    metrics: eg.Metrics
    key: jnp.ndarray
    steps: int = eg.static_field()
    static: Static = eg.static_field(default_factory=lambda: Static())

    def init_step(self, key, batch):
        return self

    def reset_step(self):
        return self.replace(metrics=self.metrics.reset())

    @jax.jit
    def predict_step(self, batch, batch_idx):
        x0, ts, key = batch
        keys = jax.random.split(key, len(ts))

        def scan_fn(x, inputs):
            t, key = inputs
            t = jnp.full((x.shape[0],), t)
            pred_noise = self.state.apply_fn({"params": self.state.params}, x, t)
            x = self.diffusion.backward_diffusion(key, x, pred_noise, t)
            return x, x

        _, xs = jax.lax.scan(scan_fn, x0, (ts, keys))
        return xs, self

    def loss_fn(self, params, key, x):
        key_t, key_diffusion = jax.random.split(key, 2)
        t = jax.random.uniform(
            key_t, (x.shape[0],), minval=0, maxval=self.steps
        ).astype(jnp.int32)
        xt, noise = self.diffusion.forward_diffusion(key_diffusion, x, t)
        noise_pred = self.state.apply_fn({"params": params}, xt, t)
        return jnp.mean((noise - noise_pred) ** 2)

    @jax.jit
    def train_step(self, batch, batch_idx, epoch_idx):
        x = batch
        loss_key, key = jax.random.split(self.key)
        loss, grads = jax.value_and_grad(self.loss_fn)(self.state.params, loss_key, x)
        state = self.state.apply_gradients(grads=grads)
        metrics = self.metrics.update(loss=loss)
        logs = metrics.compute()
        return logs, self.replace(state=state, metrics=metrics, key=key)

    def on_epoch_end(self, epoch: int, logs=None):

        x = jax.random.uniform(self.key, (1000, 2), minval=-1, maxval=1)
        ts = jnp.arange(self.steps, 0, -1)
        xs, _ = self.predict_step((x, ts, self.key), 0)
        if self.static.axes is None:
            _, self.static.axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, ti in enumerate(jnp.linspace(0, self.steps, 5).astype(int)):
            self.static.axes[i].clear()
            self.static.axes[i].scatter(xs[ti][:, 0], xs[ti][:, 1], s=1)
        show_interactive()
        return self


module = Denoiser(units=128)
variables = module.init(jax.random.PRNGKey(42), X[:1], jnp.array([0]))
tx = optax.adam(config.lr)
state = TrainState.create(apply_fn=module.apply, params=variables["params"], tx=tx)
metrics = eg.Metrics(eg.metrics.Mean(name="loss").map_arg(loss="values")).init()
ddpm = DDPM(
    diffusion=diffusion,
    state=state,
    metrics=metrics,
    steps=config.num_steps,
    key=jax.random.PRNGKey(42),
)

# %%
from elegy.utils import plot_history

trainer = eg.Trainer(ddpm)
history = trainer.fit(
    X,
    batch_size=config.batch_size,
    epochs=config.epochs,
    steps_per_epoch=config.steps_per_epoch,
)

plt.figure()
plot_history(history)
# %%
import importlib
from ddpm_utils import plot_trajectory_2d

ddpm: DDPM = trainer.module
x = jax.random.uniform(ddpm.key, (1000, 2), minval=-1, maxval=1)
ts = jnp.arange(config.num_steps, 0, -1)
xs = ddpm.predict_step((x, ts, ddpm.key), 0)[0]

if get_ipython():
    anim = plot_trajectory_2d(xs, step_size=2)
else:
    anim = plot_trajectory_2d(
        xs, step_size=2, interval=100, repeat_delay=1000, end_pad=0
    )


# %%
import ddpm_utils

importlib.reload(ddpm_utils)

from ddpm_utils import plot_density

plt.figure()
plot_density(
    model_fn=lambda x, t: ddpm.state.apply_fn({"params": ddpm.state.params}, x, t),
    ts=jnp.array([0, 10, 20]),
)
show_interactive()

# %%
plt.ioff()
plt.show()
