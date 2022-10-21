from functools import partial
from typing import Callable
from flax.struct import PyTreeNode
import jax.numpy as jnp
import jax
from einop import einop


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
    def forward(self: "GaussianDiffusion", key, x0, t):
        alpha_bars = expand_to(self.alpha_bars[t], x0)
        noise = jax.random.normal(key, x0.shape)
        xt = jnp.sqrt(alpha_bars) * x0 + jnp.sqrt(1.0 - alpha_bars) * noise
        return xt, noise

    @jax.jit
    def reverse(
        self: "GaussianDiffusion",
        key: jax.random.KeyArray,
        noise_hat: jax.Array,
        x: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        betas = expand_to(self.betas[t], x)
        alphas = expand_to(self.alphas[t], x)
        alpha_bars = expand_to(self.alpha_bars[t], x)

        z = jnp.where(
            expand_to(t, x) > 0, jax.random.normal(key, x.shape), jnp.zeros_like(x)
        )
        noise_scaled = betas / jnp.sqrt(1.0 - alpha_bars) * noise_hat
        x = (x - noise_scaled) / jnp.sqrt(alphas) + jnp.sqrt(betas) * z
        return x

    def sample(
        self,
        key: jax.random.KeyArray,
        model_fn: Callable[[jax.Array, jax.Array], jax.Array],
        x0: jax.Array,
        ts: jax.Array,
        *,
        return_all=True
    ):
        print("compiling 'sample' ...")
        keys = jax.random.split(key, len(ts))
        ts = einop(ts, "t -> t b", b=x0.shape[0])

        def scan_fn(x, inputs):
            t, key = inputs
            noise_hat = model_fn(x, t)
            x = self.reverse(key, x, noise_hat, t)
            out = x if return_all else None
            return x, out

        x, xs = jax.lax.scan(scan_fn, x0, (ts, keys))
        return xs if return_all else x


GaussianDiffusion.sample = jax.jit(
    GaussianDiffusion.sample, static_argnames=["return_all"]
)
