import functools
from pathlib import Path
import sys
from dataclasses import asdict, dataclass
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
import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.wandb_run import Run
from absl import flags
from flax import traverse_util
from flax.struct import PyTreeNode, field
from gitignore_parser import parse_gitignore
from IPython import get_ipython
from ml_collections import config_flags
from typing_extensions import Protocol

A = TypeVar("A")


@jax.jit
def ema_update(decay, ema_params, new_params):
    def _ema(ema_params, new_params):
        return decay * ema_params + (1.0 - decay) * new_params

    return jax.tree_map(_ema, ema_params, new_params)


def parse_config(config_class: Type[A]) -> A:
    config = config_class()
    if not get_ipython():
        config_flag = config_flags.DEFINE_config_dataclass("config", config)
        flags.FLAGS(sys.argv)
        config = config_flag.value
    return config


def get_wandb_run(config) -> Run:

    run = wandb.init(
        project=f"ddpm-{config.dataset}",
        config={
            ".".join(p): v
            for p, v in traverse_util.flatten_dict(asdict(config)).items()
        },
        save_code=True,
    )
    assert run is not None

    ignored = parse_gitignore(".gitignore")

    def include_fn(path: str) -> bool:
        try:
            return (
                not ignored(path)
                and "notebooks" not in path
                and not path.endswith(".ipynb")
                and not path.endswith(".png")
                and not path.endswith(".gif")
            )
        except:
            return False

    run.log_code(include_fn=include_fn)
    return run


def show(config, name: str, step: int = 0):
    if config.viz == "wandb":
        wandb.log({name: wandb.Image(plt)}, step=step)
    elif not get_ipython():
        plt.ion()
        plt.pause(1)
        plt.ioff()
    else:
        plt.show()


def print_compiling(f: A) -> A:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Compiling '{f.__name__}' ...")
        return f(*args, **kwargs)

    return wrapper


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


# ----------------------------------
# Lookahead patch
# ----------------------------------

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from absl import logging
from optax import apply_updates
from optax._src import base

# pylint:disable=no-value-for-parameter


class LookaheadState(NamedTuple):
    """State of the `GradientTransformation` returned by `lookahead`.

    Attributes:
      fast_state: Optimizer state of the fast optimizer.
      steps_since_sync: Number of fast optimizer steps taken since slow and fast
        parameters were synchronized.
    """

    fast_state: base.OptState
    slow_params: base.Params
    steps_since_sync: jnp.ndarray


class LookaheadParams(NamedTuple):
    """Holds a pair of slow and fast parameters for the lookahead optimizer.

    Gradients should always be calculated with the fast parameters. The slow
    parameters should be used for testing and inference as they generalize better.
    See the reference for a detailed discussion.

    References:
      [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

    Attributes:
      fast: Fast parameters.
      slow: Slow parameters.
    """

    fast: base.Params
    slow: base.Params

    @classmethod
    def init_synced(cls, params: base.Params) -> "LookaheadParams":
        """Initialize a pair of synchronized lookahead parameters."""
        return cls(slow=params, fast=params)


def lookahead(
    fast_optimizer: base.GradientTransformation,
    sync_period: int,
    slow_step_size: float,
    reset_state: bool = False,
) -> base.GradientTransformation:
    """Lookahead optimizer.

    Performs steps with a fast optimizer and periodically updates a set of slow
    parameters. Optionally resets the fast optimizer state after synchronization
    by calling the init function of the fast optimizer.

    Updates returned by the lookahead optimizer should not be modified before they
    are applied, otherwise fast and slow parameters are not synchronized
    correctly.

    References:
      [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

    Args:
      fast_optimizer: The optimizer to use in the inner loop of lookahead.
      sync_period: Number of fast optimizer steps to take before synchronizing
        parameters. Must be >= 1.
      slow_step_size: Step size of the slow parameter updates.
      reset_state: Whether to reset the optimizer state of the fast opimizer after
        each synchronization.

    Returns:
      A `GradientTransformation` with init and update functions. The updates
      passed to the update function should be calculated using the fast lookahead
      parameters only.
    """
    if sync_period < 1:
        raise ValueError("Synchronization period must be >= 1.")

    def init_fn(params: base.Params) -> LookaheadState:

        return LookaheadState(
            fast_state=fast_optimizer.init(params),
            slow_params=params,
            steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
        )

    def update_fn(
        updates: base.Updates, state: LookaheadState, params: base.Params
    ) -> Tuple[base.Params, LookaheadState]:
        fast_params = params
        slow_params = state.slow_params
        updates, fast_state = fast_optimizer.update(updates, state.fast_state, params)

        sync_next = state.steps_since_sync == sync_period - 1
        updates, slow_params = _lookahead_update(
            updates, sync_next, slow_params, fast_params, slow_step_size
        )
        if reset_state:
            # Jittable way of resetting the fast optimizer state if parameters will be
            # synchronized after this update step.
            initial_state = fast_optimizer.init(params)
            fast_state = jax.tree_util.tree_map(
                lambda current, init: (1 - sync_next) * current + sync_next * init,
                fast_state,
                initial_state,
            )

        steps_since_sync = (state.steps_since_sync + 1) % sync_period
        return updates, LookaheadState(fast_state, slow_params, steps_since_sync)

    return base.GradientTransformation(init_fn, update_fn)


def _lookahead_update(
    updates: base.Updates,
    sync_next: bool,
    slow_params: base.Params,
    fast_params: base.Params,
    slow_step_size: float,
) -> Tuple[base.Updates, base.Params]:
    """Returns the updates corresponding to one lookahead step.

    References:
      [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

    Args:
      updates: Updates returned by the fast optimizer.
      sync_next: Wether fast and slow parameters should be synchronized after the
        fast optimizer step.
      params: Current fast and slow parameters as `LookaheadParams` object.
      slow_step_size: Step size of the slow optimizer.

    Returns:
      The updates for the lookahead parameters.
    """
    # In the paper, lookahead is presented as two nested loops. To write lookahead
    # as optax wrapper, these loops have to be broken into successive updates.
    # This leads to two types of update steps:
    #
    # Non-synchronization steps (sync_next == False):
    # The updates returned by the fast optimizer are used for the fast parameters
    # without change and the slow parameter updates are zero (i.e. fast_updates =
    # updates, slow_updates = 0).
    #
    # Synchronisation step (sync_next == True):
    # This consists of two substeps: a last fast optimizer step and the
    # synchronization.
    #   Substep 1 (last fast optimizer step):
    #     last_fast_params = fast_params + updates
    #   Substep 2 (synchronization):
    #     new_slow_params = slow_params + slow_step_size * (
    #                       last_fast_params - slow_params)
    #     new_fast_params = new_slow_params
    #
    #   Merging into a single update step we get the update rules:
    #     slow_updates = slow_step_size * (fast_params + updates - slow_params)
    #     fast_updates = new_slow_params - fast_params = updates - (1 -
    #       slow_step_size) * (fast_params + updates - slow_params)
    #
    # To make the equations jittable, the two types of steps are merged. Defining
    # last_difference = fast_params + updates - slow_params, this yields the
    # following equtions which are implemented below:
    #   slow_updates = slow_step_size * sync_next * last_difference
    #   fast_updates = updates - (
    #                  1 - slow_step_size) * sync_next * last_difference
    last_difference = jax.tree_util.tree_map(
        lambda f, u, s: f + u - s, fast_params, updates, slow_params
    )
    slow_updates = jax.tree_util.tree_map(
        lambda diff: slow_step_size * sync_next * diff, last_difference
    )
    fast_updates = jax.tree_util.tree_map(
        lambda up, diff: up - sync_next * (1 - slow_step_size) * diff,
        updates,
        last_difference,
    )
    slow_params = jax.tree_util.tree_map(apply_updates, slow_params, slow_updates)

    return fast_updates, slow_params
