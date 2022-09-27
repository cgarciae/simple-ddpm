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

import sys
from dataclasses import asdict, dataclass
from os import stat
from typing import Any, Callable, Dict, Optional, Tuple

import jax
from absl import flags
from IPython import get_ipython
from ml_collections import config_flags
import wandb


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 500
    total_samples: int = 5_000_000
    lr: float = 3e-4
    timesteps: int = 1000
    schedule_exponent: float = 2.0
    loss_type: str = "mae"
    dataset: str = "cartoonset"
    viz: str = "matplotlib"
    model: str = "stable_unet"
    log_every: int = 200
    initial_ema_decay: float = 0.0

    @property
    def steps_per_epoch(self) -> int:
        return self.total_samples // (self.epochs * self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.total_samples // self.batch_size


config = Config()

if not get_ipython():
    config_flag = config_flags.DEFINE_config_dataclass("config", config)
    flags.FLAGS(sys.argv)
    config = config_flag.value

if config.viz == "wandb":
    run = wandb.init(
        project=f"ddpm-{config.dataset}",
        config=asdict(config),
        save_code=True,
    )

    from gitignore_parser import parse_gitignore

    ignored = parse_gitignore(".gitignore")

    def include_fn(path: str) -> bool:
        try:
            return not ignored(path) and "_files" not in path
        except:
            return False

    run.log_code(include_fn=include_fn)


print(jax.devices())

# %%
def show_interactive(name: str, step: int = 0):
    if config.viz == "wandb":
        wandb.log({name: plt}, step=step)
    elif not get_ipython():
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
            # x = tf.image.convert_image_dtype(x, tf.float32)
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


def render_image(x, ax=None):
    if ax is None:
        ax = plt.gca()

    if x.shape[-1] == 1:
        x = x[..., 0]
        cmap = "gray"
    else:
        cmap = None
    x = (x / 2.0 + 0.5) * 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    # ax.imshow(255 - x, cmap="gray")
    ax.imshow(x, cmap=cmap)
    ax.axis("off")


x_sample = ds.as_numpy_iterator().next()
num_channels = x_sample.shape[-1]

n_rows = 4
n_cols = 7
x = x_sample[: n_rows * n_cols]
plt.figure(figsize=(3 * n_cols, 3 * n_rows))
x = einop(x, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)
render_image(x)

show_interactive("samples")


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

show_interactive("betas_schedule")


from flax.struct import field

# %%
from flax.training.train_state import TrainState
from typing import Callable, Union


class EMA(PyTreeNode):
    decay_fn: Callable[[jnp.ndarray], jnp.ndarray] = field(pytree_node=False)
    params: Optional[Any]
    step: jnp.ndarray

    @classmethod
    def create(cls, decay: Union[float, Callable[[jnp.ndarray], jnp.ndarray]]):
        if isinstance(decay, (float, int, np.ndarray, jnp.ndarray)):
            decay_fn = lambda _: jnp.asarray(decay, dtype=jnp.float32)
        else:
            decay_fn = decay

        return cls(decay_fn=decay_fn, params=None, step=jnp.array(0, dtype=jnp.int32))

    def init(self, params) -> "EMA":
        return self.replace(params=params)

    def update(self, new_params) -> Tuple[Any, "EMA"]:
        if self.params is None:
            raise ValueError("EMA must be initialized")

        ema_params = jax.tree_map(self._ema, self.params, new_params)
        return ema_params, self.replace(params=ema_params, step=self.step + 1)

    def _ema(self, ema_params, new_params):
        decay = self.decay_fn(self.step)
        return decay * ema_params + (1.0 - decay) * new_params


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


def piecewise_constant_schedule(
    init_value: float, boundaries_and_values: Optional[Dict[int, float]] = None
):
    if boundaries_and_values is None:
        boundaries_and_values = {}

    boundaries_and_values_ = sorted(boundaries_and_values.items())
    boundaries = jnp.array([b for b, _ in boundaries_and_values_], dtype=jnp.int32)
    values = jnp.array(
        [init_value] + [v for _, v in boundaries_and_values_], dtype=jnp.float32
    )

    def schedule(step):
        index = jnp.sum(step >= boundaries)
        return values[index]

    return schedule


# %%
import optax
from jax_metrics.metrics import Mean, Metrics

from models.mlp_mixer import MLPMixer
from models.simple_cnn import SimpleCNN
from models.simple_unet import SimpleUNet
from models.unet_lucid import UNet
from models.unet_stable import UNet2DConfig, UNet2DModule

if config.model == "simple_unet":
    module = SimpleUNet(units=64, emb_dim=32)
elif config.model == "simple_cnn":
    module = SimpleCNN(128)
elif config.model == "mlp_mixer":
    module = MLPMixer(
        patch_size=4,
        hidden_size=64,
        mix_patch_size=512,
        mix_hidden_size=512,
        num_blocks=4,
        num_steps=config.timesteps,
    )
elif config.model == "mlp_mixer_2d":
    module = Mixer2D(64, 128, (1, 1), 4)
elif config.model == "stable_unet":
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
state = State.create(
    apply_fn=module.apply,
    params=variables["params"],
    tx=tx,
    ema=EMA.create(
        decay=config.initial_ema_decay
        # decay=piecewise_constant_schedule(
        #     config.initial_ema_decay, {10_000: 0.5, 10_000: 0.99}
        # )
    ),
)
metrics = Metrics(Mean(name="loss").map_arg(loss="values")).init()

print(module.tabulate(jax.random.PRNGKey(42), x_sample[:1], jnp.array([0]), depth=1))

# %%
from functools import partial


@jax.jit
def reverse_diffusion(process, key, x, noise_hat, t):
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
def sample(key, x0, ts, state, process, *, return_all=True):
    print("JITTING sample")
    keys = jax.random.split(key, len(ts))
    ts = einop(ts, "t -> t b", b=x0.shape[0])

    def scan_fn(x, inputs):
        t, key = inputs
        noise_hat = state.apply_fn({"params": state.params}, x, t)
        x = reverse_diffusion(process, key, x, noise_hat, t)
        out = x if return_all else None
        return x, out

    x, xs = jax.lax.scan(scan_fn, x0, (ts, keys))
    return xs if return_all else x


# %%
def loss_fn(params, xt, t, noise):
    noise_hat = state.apply_fn({"params": params}, xt, t)
    if config.loss_type == "mse":
        return jnp.mean((noise - noise_hat) ** 2)
    elif config.loss_type == "mae":
        return jnp.mean(jnp.abs(noise - noise_hat))
    else:
        raise ValueError(f"Unknown loss type {config.loss_type}")


@jax.jit
def train_step(key, x, state, metrics, process):
    print("JITTING train_step")
    key_t, key_diffusion, key = jax.random.split(key, 3)
    t = jax.random.uniform(
        key_t, (x.shape[0],), minval=0, maxval=config.timesteps - 1
    ).astype(jnp.int32)
    xt, noise = forward_diffusion(process, key_diffusion, x, t)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, xt, t, noise)
    state = state.apply_gradients(grads=grads)
    metrics = metrics.update(loss=loss)
    logs = metrics.compute()
    return logs, key, state, metrics


# %%

from time import time


class Timer:
    def __init__(self, steps_until_eval: float, deltas: Dict[float, float] = {}):
        self.steps_until_eval = steps_until_eval
        self.deltas = list(deltas.items())
        self._last_eval_step: Optional[int] = None

    def is_ready(self, step: int) -> bool:
        if self._last_eval_step is None:
            self._last_eval_step = step
            return True
        elapsed = step - self._last_eval_step
        if elapsed >= self.steps_until_eval:
            if len(self.deltas) > 0:
                update_time, steps_until_eval = self.deltas[0]
                if step >= update_time:
                    self.deltas.pop(0)
                    self.steps_until_eval = steps_until_eval
            self._last_eval_step = step
            return True
        else:
            return False


def log_metrics(logs, step):
    if logs is None:
        return

    if config.viz == "wandb":
        wandb.log(logs, step=step)

    print(f"step: {step}, " + ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()))


# %%
import numpy as np
from tqdm import tqdm

print(jax.devices())

key = jax.random.PRNGKey(42)
axs_diffusion = None
axs_samples = None
ds_iterator = ds.as_numpy_iterator()
epoch_timer = Timer(500, {10_000: 1000, 70_000: 5000})
logs = {}

for step in tqdm(range(config.total_steps), total=config.total_steps, unit="step"):

    if epoch_timer.is_ready(step):
        # --------------------
        # visualize progress
        # --------------------
        print("Sampling...")
        n_rows = 3
        n_cols = 5
        viz_key = jax.random.PRNGKey(1)
        x = jax.random.normal(viz_key, (n_rows * n_cols, *x_sample.shape[1:]))

        ts = np.arange(config.timesteps)[::-1]
        xf = np.asarray(sample(viz_key, x, ts, state, process, return_all=False))
        xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

        if axs_diffusion is None or get_ipython() or config.viz == "wandb":
            plt.figure(figsize=(3 * n_cols, 3 * n_rows))
            axs_diffusion = plt.gca()

        axs_diffusion.clear()
        render_image(xf, ax=axs_diffusion)
        show_interactive("training_samples", step=step)

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
xf = np.asarray(sample(viz_key, x, ts, state, process, return_all=False))
xf = einop(xf, "(row col) h w c -> (row h) (col w) c", row=n_rows, col=n_cols)

plt.figure(figsize=(3 * n_cols, 3 * n_rows))
render_image(xf)
show_interactive("final_samples")


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

x = jax.random.normal(key, (1000, 2))
ts = jnp.arange(config.timesteps)[::-1]
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
