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

import utils


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 500
    total_samples: int = 5_000_000
    lr: float = 2e-4
    timesteps: int = 1000
    schedule_exponent: float = 2.0
    loss_type: str = "mae"
    dataset: str = "cartoonset"
    viz: str = "matplotlib"
    model: str = "stable_unet"
    eval_every: int = 2000
    log_every: int = 200
    ema_decay: float = 0.995
    ema_update_every: int = 10
    ema_update_after_step: int = 100

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.total_samples // self.batch_size


config = utils.setup_config(Config)

print(jax.devices())


# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets.load import load_dataset


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


ds = get_data(config.dataset, config.batch_size)

# %%
from einop import einop

from utils import render_image, show

x_sample = ds.as_numpy_iterator().next()
num_channels = x_sample.shape[-1]

n_rows = 4
n_cols = 7
x = x_sample[: n_rows * n_cols]
plt.figure(figsize=(3 * n_cols, 3 * n_rows))
x = einop(x, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)
render_image(x)
show("samples")


# %%
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
betas = cosine_schedule(1e-5, 0.5, config.timesteps, exponent=config.schedule_exponent)
# betas = polynomial_schedule(1e-5, 1e-2, config.timesteps, exponent=2)
process = GaussianDiffusion.create(betas)
n_rows = 2
n_cols = 7

_, (ax_img, ax_plot) = plt.subplots(2, 1, figsize=(3 * n_cols, 3 * n_rows))

t = jnp.linspace(0, config.timesteps, n_cols).astype(int)
x = einop(x_sample[0], "h w c -> b h w c", b=n_cols)
x, _ = forward_diffusion(process, jax.random.PRNGKey(0), x, t)
x = einop(x, "col h w c -> h (col w) c", col=n_cols)
render_image(x, ax=ax_img)

linear = polynomial_schedule(betas.min(), betas.max(), config.timesteps, exponent=1.0)
ax_plot.plot(linear, label="linear", color="black", linestyle="dotted")
ax_plot.plot(betas)
for s in ["top", "bottom", "left", "right"]:
    ax_plot.spines[s].set_visible(False)

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

    def update(self, *, grads, **kwargs):
        self = self.apply_gradients(grads=grads, **kwargs)
        ema = self.ema.update(self.params)
        return self.replace(ema=ema)


# %%
import optax
from jax_metrics.metrics import Mean, Metrics

from models.mlp_mixer import MLPMixer
from models.simple_cnn import SimpleCNN
from models.simple_unet import SimpleUNet
from models.unet_lucid import UNet
from models.unet_stable import UNet2DConfig, UNet2DModule

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
            config.lr,
            {
                int(config.total_steps / 3): 1 / 3,
                int(config.total_steps * 2 / 3): 3 / 10,
            },
        )
    ),
)
state = TrainState.create(
    apply_fn=module.apply,
    params=variables["params"],
    tx=tx,
    ema=EMA.create(
        params=variables["params"],
        decay=config.ema_decay,
        update_every=config.ema_update_every,
        update_after_step=config.ema_update_after_step,
    ),
)
metrics = Metrics(
    [
        Mean(name="loss").map_arg(loss="values"),
        Mean(name="ema_loss").map_arg(ema_loss="values"),
    ]
).init()

print(module.tabulate(jax.random.PRNGKey(42), x_sample[:1], jnp.array([0]), depth=1))

# %%
from functools import partial


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
        key_t, (x.shape[0],), minval=0, maxval=config.timesteps - 1
    ).astype(jnp.int32)
    xt, noise = forward_diffusion(process, key_diffusion, x, t)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, xt, t, noise)
    ema_loss = loss_fn(state.ema.params, xt, t, noise)
    state = state.update(grads=grads)
    metrics = metrics.update(loss=loss, ema_loss=ema_loss)
    logs = metrics.compute()
    return logs, key, state, metrics


# %%
import numpy as np
from tqdm import tqdm

from utils import log_metrics

print(jax.devices())

key = jax.random.PRNGKey(42)
axs_diffusion = None
axs_samples = None
ds_iterator = ds.as_numpy_iterator()
logs = {}

for step in tqdm(range(config.total_steps), total=config.total_steps, unit="step"):

    if step % config.eval_every == 0:
        # --------------------
        # visualize progress
        # --------------------
        print("Sampling...")
        n_rows = 3
        n_cols = 5
        viz_key = jax.random.PRNGKey(1)
        x = jax.random.normal(viz_key, (n_rows * n_cols, *x_sample.shape[1:]))

        ts = np.arange(config.timesteps)[::-1]
        xf = np.asarray(
            sample(viz_key, x, ts, state.ema.params, process, return_all=False)
        )
        xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

        if axs_diffusion is None or get_ipython() or config.viz == "wandb":
            plt.figure(figsize=(3 * n_cols, 3 * n_rows))
            axs_diffusion = plt.gca()

        axs_diffusion.clear()
        render_image(xf, ax=axs_diffusion)
        show("training_samples", step=step)

    if step % config.log_every == 0:
        print()  # newline
        log_metrics(logs, step)
        metrics = metrics.reset()

    # --------------------
    # trainig step
    # --------------------
    x = ds_iterator.next()
    logs, key, state, metrics = train_step(key, x, state, metrics, process)


# %%
n_rows = 3
n_cols = 5
viz_key = jax.random.PRNGKey(1)
x = jax.random.normal(viz_key, (n_rows * n_cols, *x_sample.shape[1:]))

ts = np.arange(config.timesteps)[::-1]
xf = np.asarray(sample(viz_key, x, ts, state.ema.params, process, return_all=False))
xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

plt.figure(figsize=(3 * n_cols, 3 * n_rows))
render_image(xf)
show("final_samples")
