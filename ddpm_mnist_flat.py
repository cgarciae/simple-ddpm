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
from typing import Any, Callable, Optional, Tuple

from IPython import get_ipython


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 200
    total_samples: int = 5_000_000
    lr: float = 1e-3
    num_steps: int = 200
    schedule_exponent: float = 2.0

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)


config = Config()

# %%

if not get_ipython():
    import sys

    from absl import flags
    from ml_collections import config_flags

    config_flag = config_flags.DEFINE_config_dataclass("config", config)
    flags.FLAGS(sys.argv)
    config = config_flag.value


def show_interactive():
    if not get_ipython():
        plt.ion()
        plt.pause(1)
        plt.ioff()
    else:
        plt.show()


# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets.load import load_dataset


def get_data():
    ds = load_dataset("mnist", split="train")
    X = np.stack(ds["image"])[..., None]
    return X / 127.5 - 1.0


def render_image(x, ax=None):
    if ax is None:
        ax = plt.gca()

    x = (x[..., 0] + 1) * 127.5
    x = np.clip(x, 0, 255).astype(np.uint8)
    ax.imshow(255 - x, cmap="gray")
    ax.axis("off")


X = get_data()

x = X[np.random.choice(len(X), 8)]
_, axs_diffusion = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axs_diffusion.flatten()):
    render_image(x[i], ax=ax)

show_interactive()


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


def forward_diffusion(process, key, x0, t):
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
    0.0001, 0.5, config.num_steps, exponent=config.schedule_exponent
)
process = GaussianDiffusion.create(betas)

x = X[:1]
plt.figure(figsize=(15, 6))
for i, ti in enumerate(jnp.linspace(0, config.num_steps, 5).astype(int)):
    t = jnp.full((x.shape[0],), ti)
    xt, noise = forward_diffusion(process, jax.random.PRNGKey(ti), x, t)
    ax = plt.subplot(2, 5, i + 1)
    render_image(xt[i], ax=ax)
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
from einop import einop
from typing import Optional, Tuple


class PositionalEmbedding(nn.Module):
    dim: int

    def __call__(self, t):
        half_dim = self.dim // 2
        mul = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(-mul * jnp.arange(half_dim))
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeConditioned(nn.Module):
    emb_dim: int
    module: nn.Module

    @nn.compact
    def __call__(self, x, t):
        t_embeddings = PositionalEmbedding(self.emb_dim)(t)
        axis = {f"a{i}": dim for i, dim in enumerate(x.shape[1:-1])}
        t_embeddings = einop(t_embeddings, f"b c -> b {' '.join(axis)} c", **axis)
        x = jnp.concatenate([x, t_embeddings], axis=-1)
        x = self.module(x)
        return x


class TimeConditionedMLP(nn.Module):
    units: int = 128
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x, t):
        input_shape = x.shape
        inputs_units = np.prod(input_shape[1:])
        x = x.reshape(-1, inputs_units)
        dense = lambda units: TimeConditioned(self.emb_dim, nn.Dense(units))
        x = nn.relu(dense(self.units)(x, t))
        x = nn.relu(dense(self.units)(x, t)) + x
        x = nn.relu(dense(self.units)(x, t)) + x
        x = dense(inputs_units)(x, t)
        x = x.reshape(*input_shape)
        return x


class SimpleUNet(nn.Module):
    units: int = 128
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x, t):
        # Downsample
        x = skip_0 = TimeConditioned(self.emb_dim, nn.Conv(32, (5, 5), padding="SAME"))(
            x, t
        )
        x = nn.GroupNorm(8)(x)
        x = nn.relu(x)
        x = TimeConditioned(
            self.emb_dim, nn.Conv(64, (5, 5), strides=(2, 2), padding="SAME")
        )(x, t)
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)
        x = skip_1 = TimeConditioned(self.emb_dim, nn.Conv(64, (3, 3), padding="SAME"))(
            x, t
        )
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)
        x = TimeConditioned(
            self.emb_dim, nn.Conv(128, (3, 3), strides=(2, 2), padding="SAME")
        )(x, t)
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)
        x = TimeConditioned(self.emb_dim, nn.Conv(128, (3, 3), padding="SAME"))(x, t)
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)

        # Upsample
        x = TimeConditioned(
            self.emb_dim, nn.ConvTranspose(128, (3, 3), strides=(2, 2))
        )(x, t)
        x = jnp.concatenate([x, skip_1], axis=-1)
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)
        x = TimeConditioned(self.emb_dim, nn.Conv(128, (3, 3), padding="SAME"))(x, t)
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)
        x = TimeConditioned(
            self.emb_dim, nn.ConvTranspose(128, (3, 3), strides=(2, 2))
        )(x, t)
        x = jnp.concatenate([x, skip_0], axis=-1)
        x = nn.GroupNorm(16)(x)
        x = nn.relu(x)
        x = TimeConditioned(self.emb_dim, nn.Conv(1, (5, 5), padding="SAME"))(x, t)


class SimpleCNN(nn.Module):
    units: int = 128
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x, t):
        input_units = x.shape[-1]
        conv = lambda kernel_size, stride=1, **kwargs: TimeConditioned(
            self.emb_dim,
            nn.Conv(
                self.units,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding="SAME",
                **kwargs,
            ),
        )
        conv_trans = lambda kernel_size, stride=1, **kwargs: TimeConditioned(
            self.emb_dim,
            nn.ConvTranspose(
                self.units,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding="SAME",
                **kwargs,
            ),
        )
        norm = lambda n: nn.GroupNorm(n)

        # Downsample
        x = conv(5)(x, t)
        x = norm(8)(x)
        x = nn.relu(x)

        x = conv(5, stride=2)(x, t)
        x = norm(8)(x)
        x = nn.relu(x)

        x = conv(3)(x, t)
        x = norm(8)(x)
        x = nn.relu(x)

        # Upsample
        x = conv_trans(5, stride=2)(x, t)
        x = norm(8)(x)
        x = nn.relu(x)

        x = conv(5)(x, t)
        x = norm(8)(x)
        x = nn.relu(x)

        x = nn.Dense(input_units)(x)

        return x


class MLP1D(nn.Module):
    out_dim: int
    units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class MixerBlock1D(nn.Module):
    mix_patch_size: int
    mix_hidden_size: int

    @nn.compact
    def __call__(self, x):
        _, num_patches, hidden_size = x.shape
        patch_mixer = MLP1D(num_patches, self.mix_patch_size)
        hidden_mixer = MLP1D(hidden_size, self.mix_hidden_size)
        norm1 = nn.LayerNorm()
        norm2 = nn.LayerNorm()

        x = einop(x, "... p c -> ... c p")
        x = x + patch_mixer(norm1(x))
        x = einop(x, "... c p -> ... p c")
        x = x + hidden_mixer(norm2(x))
        return x


class Mixer1D(nn.Module):
    patch_size: int
    hidden_size: int
    mix_patch_size: int
    mix_hidden_size: int
    num_blocks: int
    num_steps: int

    @nn.compact
    def __call__(self, x, t):
        input_size = x.shape[-1]
        batch_size = x.shape[0]
        height, width = x.shape[-3], x.shape[-2]
        # ----------------
        # setup
        # ----------------

        conv_in = nn.Conv(
            self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
        )
        conv_out = nn.ConvTranspose(
            input_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
        )
        blocks = [
            MixerBlock1D(self.mix_patch_size, self.mix_hidden_size)
            for _ in range(self.num_blocks)
        ]
        norm = nn.LayerNorm()

        ################
        t = t / self.num_steps
        t = einop(t, "b -> b h w 1", b=batch_size, h=height, w=width)
        x = jnp.concatenate([x, t], axis=-1)
        x = conv_in(x)
        _, patch_height, patch_width, _ = x.shape
        x = einop(x, "b h w c -> b (h w) c")
        for block in blocks:
            x = block(x)
        x = norm(x)
        x = einop(x, "b (h w) c -> b h w c", h=patch_height, w=patch_width)
        return conv_out(x)


class MLP2D(nn.Module):
    units: int
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x, t):
        out_dim = self.out_dim or x.shape[-1]
        x = TimeConditioned(32, nn.Dense(self.units))(x, t)
        x = nn.relu(x)
        x = nn.Dense(out_dim)(x)
        return x


class MixerBlock2D(nn.Module):
    units: int

    @nn.compact
    def __call__(self, x, t):
        mlp = lambda x, t: MLP2D(self.units)(x, t)
        norm = lambda x: nn.LayerNorm()(x)

        x = x + mlp(norm(x), t)  # mix channels
        x = einop(x, "... h w c -> ... c h w")
        x = x + mlp(norm(x), t)  # mix width
        x = einop(x, "... c h w -> ... w c h")
        x = x + mlp(norm(x), t)  # mix height
        x = einop(x, "... w c h -> ... h w c")
        return x


class Mixer2D(nn.Module):
    units: int
    hidden_units: int
    patch_size: Tuple[int, int]
    num_blocks: int

    @nn.compact
    def __call__(self, x, t):
        input_size = x.shape[-1]

        conv_in = nn.Conv(
            self.units,
            kernel_size=self.patch_size,
            strides=self.patch_size,
        )
        conv_out = nn.ConvTranspose(
            input_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
        )
        blocks = [MixerBlock2D(self.hidden_units) for _ in range(self.num_blocks)]
        norm = nn.LayerNorm()

        x = conv_in(x)
        for block in blocks:
            x = block(x, t)
        x = norm(x)
        x = conv_out(x)
        return x


# %%
from flax.training.train_state import TrainState
from flax.struct import field


class EMA(PyTreeNode):
    mu: float = field(pytree_node=False, default=0.999)
    params: Optional[Any] = None
    step: Optional[jnp.ndarray] = None

    def init(self, params) -> "EMA":
        return self.replace(params=params, step=jnp.array(0, dtype=jnp.int32))

    def update(self, new_params) -> Tuple[Any, "EMA"]:
        if self.params is None or self.step is None:
            raise ValueError("EMA must be initialized")

        updates = jax.tree_map(self._ema, self.params, new_params)
        return updates, self.replace(params=updates, step=self.step + 1)

    def _ema(self, params, new_params):
        return self.mu * params + (1.0 - self.mu) * new_params


class State(TrainState):
    ema: EMA

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema: EMA, **kwargs):
        return super().create(
            apply_fn=apply_fn, params=params, tx=tx, ema=ema.init(params), **kwargs
        )

    def apply_gradients(self, *, grads, **kwargs):
        self = super().apply_gradients(grads=grads, **kwargs)
        params, ema = self.ema.update(self.params)
        return self.replace(params=params, ema=ema)


# %%
import optax

from jax_metrics.metrics import Mean, Metrics
from models import UNet

# module = SimpleUNet(units=64, emb_dim=32)
# module = TimeConditionedMLP(2048)
module = SimpleCNN(128)
# module = Mixer1D(
#     patch_size=4,
#     hidden_size=64,
#     mix_patch_size=512,
#     mix_hidden_size=512,
#     num_blocks=4,
#     num_steps=config.num_steps,
# )
# module = Mixer2D(64, 128, (1, 1), 4)
# module = UNet(dim=32, dim_mults=(1, 2, 4), channels=1)
variables = module.init(jax.random.PRNGKey(42), X[:1], jnp.array([0]))
# tx = optax.adamw(
#     optax.linear_onecycle_schedule(
#         config.total_samples // config.batch_size,
#         2 * config.lr,
#     )
# )
# tx = optax.adamw(optax.linear_schedule(config.lr, config.lr / 10, config.num_steps))
tx = optax.adamw(config.lr)
state = State.create(
    apply_fn=module.apply, params=variables["params"], tx=tx, ema=EMA(mu=0.9)
)
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

    return jnp.where(expand_to(t, x) == 0, x, x + sampling_noise)


# %%
def loss_fn(params, xt, t, noise):
    noise_hat = state.apply_fn({"params": params}, xt, t)
    return jnp.mean((noise - noise_hat) ** 2)


@jax.jit
def train_step(key, x, state, metrics, process):
    key_t, key_diffusion, key = jax.random.split(key, 3)
    t = jax.random.uniform(
        key_t, (x.shape[0],), minval=0, maxval=config.num_steps
    ).astype(jnp.int32)
    xt, noise = forward_diffusion(process, key_diffusion, x, t)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, xt, t, noise)
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
axs_diffusion = None
axs_samples = None

for epoch in range(config.epochs):
    kbar = Kbar(
        target=config.steps_per_epoch,
        epoch=epoch,
        num_epochs=config.epochs,
        width=16,
        always_stateful=True,
        verbose=2 if get_ipython() else 1,
    )
    metrics = metrics.reset()

    for step in range(config.steps_per_epoch):
        x = X[np.random.choice(np.arange(len(X)), config.batch_size)]
        logs, key, state, metrics = train_step(key, x, state, metrics, process)
        kbar.update(step, values=list(logs.items()))

    print(logs)

    # --------------------
    # visualize progress
    # --------------------
    n_rows = 3
    viz_key = jax.random.PRNGKey(0)
    x = jax.random.uniform(viz_key, (n_rows, *X.shape[1:]), minval=-1, maxval=1)

    ts = jnp.arange(config.num_steps)[::-1]
    xs = sample(viz_key, x, ts, state, process)
    xs = np.concatenate([x[None], xs], axis=0)

    if axs_diffusion is None or get_ipython():
        _, axs_diffusion = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
    for r, ax_row in enumerate(axs_diffusion):
        for i, ti in enumerate(jnp.linspace(0, len(xs) - 1, 5).astype(int)):
            ax_row[i].clear()
            render_image(xs[ti, r], ax_row[i])
            ax_row[i].axis("off")
    show_interactive()
    print()  # newline


# %%

from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory

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
ts = jnp.arange(config.num_steps)[::-1]
xs = sample(key, x, ts, state, process)

if get_ipython():
    anim = plot_trajectory_2d(xs, step_size=2)
else:
    anim = plot_trajectory_2d(
        xs, step_size=2, interval=100, repeat_delay=1000, end_pad=0
    )

# %%


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
