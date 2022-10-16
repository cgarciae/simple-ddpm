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
#     display_name: 'Python 3.8.11 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
from dataclasses import dataclass
from utils import setup_config
from IPython import get_ipython


@dataclass
class Config:
    dataset: str = "moons"
    batch_size: int = 32
    epochs: int = 10
    total_samples: int = 5_000_000
    lr: float = 1e-3
    num_steps: int = 50
    schedule_exponent: float = 2.0
    viz: str = "matplotlib"

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)


config = setup_config(Config)

# %%
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler
from utils import show
import tensorflow as tf


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

    return x, ds


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


# %%
betas = cosine_schedule(
    0.00001, 0.01, config.num_steps, exponent=config.schedule_exponent
)
# betas = betas_for_alpha_bar(config.num_steps, scale=0.01)
process = GaussianDiffusion.create(betas)

plt.figure(figsize=(15, 6))
for i, ti in enumerate(jnp.linspace(0, config.num_steps, 5).astype(int)):
    t = jnp.full((X.shape[0],), ti)
    xt, noise = forward_diffusion(process, jax.random.PRNGKey(ti), X, t)
    plt.subplot(2, 5, i + 1)
    plt.scatter(xt[:, 0], xt[:, 1], s=1)
    plt.axis("off")

plt.subplot(2, 1, 2)
linear = polynomial_schedule(betas.min(), betas.max(), config.num_steps, exponent=1.0)
plt.plot(linear, label="linear", color="black", linestyle="dotted")
plt.plot(betas)
for s in ["top", "bottom", "left", "right"]:
    plt.gca().spines[s].set_visible(False)

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
        dense = lambda units: TimeConditionedDense(units, self.emb_dim)
        x = nn.relu(dense(self.units)(x, t))
        x = nn.relu(dense(self.units)(x, t)) + x
        x = nn.relu(dense(self.units)(x, t)) + x
        x = dense(inputs_units)(x, t)
        return x


# %%
import optax
from flax.training.train_state import TrainState
from jax_metrics.metrics import Mean, Metrics

module = Denoiser(units=192, emb_dim=32)
variables = module.init(jax.random.PRNGKey(42), X[:1], jnp.array([0]))
tx = optax.adamw(
    optax.linear_onecycle_schedule(
        config.total_samples // config.batch_size,
        2 * config.lr,
    )
)
state = TrainState.create(apply_fn=module.apply, params=variables["params"], tx=tx)
metrics = Metrics(Mean(name="loss").map_arg(loss="values")).init()

print(module.tabulate(jax.random.PRNGKey(42), X[:1], jnp.array([0]), depth=1))

# %%
def reverse_diffusion(process, key, x, noise_hat, t):
    betas = expand_to(process.betas[t], x)
    alphas = expand_to(process.alphas[t], x)
    alpha_bars = expand_to(process.alpha_bars[t], x)

    sampling_noise = jnp.sqrt(betas) * jax.random.normal(key, x.shape)
    noise_hat = betas / jnp.sqrt(1.0 - alpha_bars) * noise_hat
    x = (x - noise_hat) / jnp.sqrt(alphas)

    return jnp.where(t[:, None] == 0, x, x + sampling_noise)


# %%
def loss_fn(params, key, x, process):
    key_t, key_diffusion = jax.random.split(key, 2)
    t = jax.random.uniform(
        key_t, (x.shape[0],), minval=0, maxval=config.num_steps
    ).astype(jnp.int32)
    xt, noise = forward_diffusion(process, key_diffusion, x, t)
    noise_hat = state.apply_fn({"params": params}, xt, t)
    return jnp.mean((noise - noise_hat) ** 2)


@jax.jit
def train_step(key, x, state, metrics, process):
    loss_key, key = jax.random.split(key)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, loss_key, x, process)
    state = state.apply_gradients(grads=grads)
    metrics = metrics.update(loss=loss)
    logs = metrics.compute()
    return logs, key, state, metrics


@jax.jit
def sample(key, x0, ts, state, process):
    keys = jax.random.split(key, len(ts))

    def scan_fn(x, inputs):
        t, key = inputs
        t = jnp.full((x.shape[0],), t)
        noise_hat = state.apply_fn({"params": state.params}, x, t)
        x = reverse_diffusion(process, key, x, noise_hat, t)
        return x, x

    _, xs = jax.lax.scan(scan_fn, x0, (ts, keys))
    return xs


# %%
import numpy as np
from pkbar import Kbar

key = jax.random.PRNGKey(42)
axs = None

for epoch in range(config.epochs):
    kbar = Kbar(
        target=config.steps_per_epoch,
        epoch=epoch,
        num_epochs=config.epochs,
        width=16,
        always_stateful=True,
    )
    metrics = metrics.reset()

    for step in range(config.steps_per_epoch):
        x = X[np.random.choice(np.arange(len(X)), config.batch_size)]
        logs, key, state, metrics = train_step(key, x, state, metrics, process)
        kbar.update(step, values=list(logs.items()))

    # --------------------
    # visualize progress
    # --------------------
    viz_key = jax.random.PRNGKey(0)
    x = jax.random.uniform(viz_key, (1000, 2), minval=-1, maxval=1)
    ts = jnp.arange(config.num_steps, 0, -1)
    xs = sample(viz_key, x, ts, state, process)
    if axs is None or get_ipython():
        _, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, ti in enumerate(jnp.linspace(0, config.num_steps, 5).astype(int)):
        axs[i].clear()
        axs[i].scatter(xs[ti, :, 0], xs[ti, :, 1], s=1)
        axs[i].axis("off")
    show_interactive()
    print()  # newline


# %%

from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory

from einop import einop
from IPython.display import HTML, display
from matplotlib import animation


def plot_trajectory_2d(
    xs: np.ndarray,
    interval: int = 10,
    repeat_delay: int = 1000,
    step_size: int = 1,
    end_pad: int = 500,
):

    xs = xs[::step_size]

    # replace last sample to create a 'pause' effect
    pad_end = einop(xs[-1], "... -> batch ...", batch=end_pad)
    xs = np.concatenate([xs, pad_end], axis=0)

    N = len(xs)

    fig = plt.figure()
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

    if get_ipython():
        with TemporaryDirectory() as tmpdir:
            img_name = Path(tmpdir) / f"diffusion.gif"
            anim.save(str(img_name), writer="pillow", fps=60)
            image_bytes = b64encode(img_name.read_bytes()).decode("utf-8")

        display(HTML(f"""<img src='data:image/gif;base64,{image_bytes}'>"""))
    else:
        pass

    return anim


# %%

x = jax.random.uniform(key, (1000, 2), minval=-1, maxval=1)
ts = jnp.arange(config.num_steps, 0, -1)
xs = sample(key, x, ts, state, process)

if get_ipython():
    anim = plot_trajectory_2d(xs, step_size=2)
else:
    anim = plot_trajectory_2d(
        xs, step_size=2, interval=100, repeat_delay=1000, end_pad=0
    )

# %%
from typing import Any, Callable


def plot_density(model_fn: Callable[[Any, Any], Any], ts):
    x = jnp.linspace(-1, 1, 100)
    y = jnp.linspace(-1, 1, 100)
    xx, yy = jnp.meshgrid(x, y)
    X = jnp.stack([xx, yy], axis=-1)

    def mass_fn(x, t):
        t_ = jnp.full((1,), t)
        x_ = x[None]
        noise_hat = model_fn(x_, t_)
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
    mass = mass_fn(X, ts).mean(axis=-1)
    plt.contourf(xx, yy, mass, levels=100)


# %%

plt.figure()
plot_density(
    model_fn=lambda x, t: state.apply_fn({"params": state.params}, x, t),
    ts=jnp.array([0, 10, 20]),
)
show_interactive()

# %%
plt.ioff()
plt.show()
