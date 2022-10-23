from functools import partial
from typing import Callable
from flax.struct import PyTreeNode
import jax.numpy as jnp
import jax
from einop import einop
from simple_ddpm.utils import print_compiling


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

    @print_compiling
    def forward(self: "GaussianDiffusion", *, key, x, t):
        alpha_bars = expand_to(self.alpha_bars[t], x)
        noise = jax.random.normal(key, x.shape)
        xt = jnp.sqrt(alpha_bars) * x + jnp.sqrt(1.0 - alpha_bars) * noise
        return xt, noise

    @print_compiling
    def reverse(
        self: "GaussianDiffusion",
        *,
        key: jax.random.KeyArray,
        noise: jax.Array,
        x: jax.Array,
        t: jax.Array
    ) -> jax.Array:
        betas = expand_to(self.betas[t], x)
        alphas = expand_to(self.alphas[t], x)
        alpha_bars = expand_to(self.alpha_bars[t], x)

        z = jnp.where(
            expand_to(t, x) > 0, jax.random.normal(key, x.shape), jnp.zeros_like(x)
        )
        a = 1.0 / jnp.sqrt(alphas)
        b = betas / jnp.sqrt(1.0 - alpha_bars)
        x = a * (x - b * noise) + jnp.sqrt(betas) * z
        return x

    @print_compiling
    def sample(self, *, key, model_fn, state, x, t, return_all: bool = False):
        keys = jax.random.split(key, len(t))
        t = einop(t, "t -> t b", b=x.shape[0])

        def scan_fn(x, inputs):
            t, key = inputs
            noise_hat = model_fn(state, x, t)
            x = self.reverse(key=key, noise=noise_hat, x=x, t=t)
            out = x if return_all else None
            return x, out

        xf, xt = jax.lax.scan(scan_fn, x, (t, keys))
        return xt if xt is not None else xf


GaussianDiffusion.forward = jax.jit(GaussianDiffusion.forward)
GaussianDiffusion.reverse = jax.jit(GaussianDiffusion.reverse)
GaussianDiffusion.sample = jax.jit(
    GaussianDiffusion.sample, static_argnames=("model_fn", "return_all")
)
