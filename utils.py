import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
from absl import flags
from ml_collections import config_flags
import wandb
import matplotlib.pyplot as plt
from IPython import get_ipython
from gitignore_parser import parse_gitignore
from dataclasses import asdict, dataclass
import numpy as np
from typing_extensions import Protocol
from flax.struct import PyTreeNode, field


class Config(Protocol):
    viz: str
    dataset: str


CONFIG: Optional[Config]
CONFIG = None

A = TypeVar("A")
C = TypeVar("C", bound=Config)


class EMA(Generic[A], PyTreeNode):
    decay_fn: Callable[[jnp.ndarray], jnp.ndarray] = field(pytree_node=False)
    params: A
    update_after_step: int = field(pytree_node=False)
    update_every: int = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        params: A,
        decay: Union[float, Callable[[jnp.ndarray], jnp.ndarray]],
        update_after_step: int = -1,
        update_every: int = 10,
    ):
        if isinstance(decay, (float, int, np.ndarray, jnp.ndarray)):
            decay_fn = lambda _: jnp.asarray(decay, dtype=jnp.float32)
        else:
            decay_fn = decay

        return cls(
            decay_fn=decay_fn,
            params=params,
            update_after_step=update_after_step,
            update_every=update_every,
        )

    def update(self, step: int, new_params) -> "EMA":
        if step <= self.update_after_step:
            ema_params = new_params
        elif step % self.update_every == 0:
            ema_params = self._ema_update(step, new_params)
        else:
            ema_params = self.params

        return self.replace(params=ema_params)

    @jax.jit
    def _ema_update(self, step, new_params):
        decay = self.decay_fn(step)

        def _ema(ema_params, new_params):
            return decay * ema_params + (1.0 - decay) * new_params

        return jax.tree_map(_ema, self.params, new_params)


def setup_config(config_class: Type[C]) -> C:
    global CONFIG

    config = config_class()

    if not get_ipython():
        config_flag = config_flags.DEFINE_config_dataclass("config", config)
        flags.FLAGS(sys.argv)
        config = config_flag.value

    CONFIG = config

    if config.viz == "wandb":
        run = wandb.init(
            project=f"ddpm-{config.dataset}",
            config=asdict(config),
            save_code=True,
        )

        ignored = parse_gitignore(".gitignore")

        def include_fn(path: str) -> bool:
            try:
                return not ignored(path) and "_files" not in path
            except:
                return False

        run.log_code(include_fn=include_fn)

    return config


def show(name: str, step: int = 0):
    if CONFIG.viz == "wandb":
        wandb.log({name: plt}, step=step)
    elif not get_ipython():
        plt.ion()
        plt.pause(1)
        plt.ioff()
    else:
        plt.show()


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


def switch_case(
    cond_branches: Sequence[Tuple[jnp.ndarray, Callable]], *, otherwise: Callable
):
    for cond, _ in cond_branches:
        if cond.shape != ():
            raise ValueError("Conditions must be scalars")
        if cond.dtype != jnp.bool_:
            raise ValueError("Conditions must be booleans")

    conditions = jnp.stack(
        [c.astype(jnp.int32) for c, b in cond_branches]
        + [jnp.array(1, dtype=jnp.int32)]
    )
    braches = [b for c, b in cond_branches] + [otherwise]
    index = jnp.argmax(jnp.cumsum(conditions) == 1)

    def _switch(*operands, **kwargs):
        return jax.lax.switch(index, braches, *operands, **kwargs)

    return _switch


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


def log_metrics(logs, step):
    if logs is None:
        return

    if CONFIG.viz == "wandb":
        wandb.log(logs, step=step)

    print(f"step: {step}, " + ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()))
