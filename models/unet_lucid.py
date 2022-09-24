import typing as tp

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einop import einop

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="nn.Module")
ArrayFn = tp.Callable[[jnp.ndarray], jnp.ndarray]


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def conv_padding(*args: int) -> tp.List[tp.Tuple[int, int]]:
    return [(p, p) for p in args]


class Residual(nn.Module):
    fn: nn.Module

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalEmb(nn.Module):
    dim: int

    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


def Upsample(dim: int):
    return nn.ConvTranspose(dim, (4, 4), strides=(2, 2), padding=conv_padding(2, 2))


def Downsample(dim: int):
    return nn.Conv(dim, (4, 4), strides=(2, 2), padding=conv_padding(1, 1))


class PreNorm(nn.Module):
    fn: ArrayFn

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm()(x)
        return self.fn(x)


class Sequential(nn.Module):
    modules: tp.Tuple[ArrayFn, ...]

    @classmethod
    def new(cls, *modules):
        return cls(modules)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for module in self.modules:
            x = module(x)
        return x


class Identity(nn.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    dim: int
    dim_out: int
    time_emb_dim: tp.Optional[int] = None
    mult: int = 2
    norm: bool = True

    def setup(self):
        self.mlp = (
            Sequential.new(nn.gelu, nn.Dense(self.dim))
            if self.time_emb_dim is not None
            else None
        )

        self.ds_conv = nn.Conv(
            self.dim, [7, 7], padding=conv_padding(3, 3), feature_group_count=self.dim
        )

        self.net = Sequential.new(
            nn.LayerNorm() if self.norm else Identity(),
            nn.Conv(self.dim_out * self.mult, [3, 3], padding=conv_padding(1, 1)),
            nn.gelu,
            nn.Conv(self.dim_out, [3, 3], padding=conv_padding(1, 1)),
        )

        self.res_conv = (
            nn.Conv(self.dim_out, [1, 1], padding=conv_padding(0, 0))
            if self.dim != self.dim_out
            else Identity()
        )

    def __call__(self, x: jnp.ndarray, time_emb: tp.Optional[jnp.ndarray] = None):
        h = self.ds_conv(x)

        if self.mlp is not None:
            assert time_emb is not None, "time emb must be passed in"
            condition = self.mlp(time_emb)
            h = h + einop(condition, "b c -> b 1 1 c")

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    def setup(self):
        hidden_dim = self.dim_head * self.heads
        self.to_qkv = nn.Conv(hidden_dim * 3, [1, 1], use_bias=False)
        self.to_out = nn.Conv(self.dim, [1, 1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        qkv = jnp.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(
            lambda t: einop(t, "b x y (h c) -> b (x y) h c", h=self.heads),
            qkv,
        )

        scale = self.dim_head**-0.5
        q = q * scale

        k = nn.softmax(k, axis=-1)
        context = jnp.einsum("b n h d , b n h e -> b h d e", k, v)

        out = jnp.einsum("b h d e, b n h d -> b n h e ", context, q)
        out = einop(out, "b (x y) h c -> b x y (h c)", h=self.heads, x=h, y=w)
        return self.to_out(out)


def Resize(
    sample_shape: tp.Sequence[int],
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
    antialias: bool = True,
    precision=jax.lax.Precision.HIGHEST,
) -> ArrayFn:
    def _resize(x: jnp.ndarray) -> jnp.ndarray:
        batch_dims = len(x.shape) - len(sample_shape) - 1  # 1 = channel dim
        shape = (*x.shape[:batch_dims], *sample_shape, x.shape[-1])
        output = jax.image.resize(
            x,
            shape=shape,
            method=method,
            antialias=antialias,
            precision=precision,
        )

        return output

    return _resize



class UNet(nn.Module):
    dim: int
    out_dim: tp.Optional[int] = None
    dim_mults: tp.Tuple[int, ...] = (1, 2, 4, 8)
    channels: int = 3
    with_time_emb: bool = True

    def setup(self):
        dims = [self.channels, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if self.with_time_emb:
            time_dim = self.dim
            self.time_mlp = Sequential.new(
                SinusoidalEmb(self.dim),
                nn.Dense(self.dim * 4),
                nn.gelu,
                nn.Dense(self.dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        downs = []
        ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            downs.append(
                [
                    ConvNextBlock(
                        dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0
                    ),
                    ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else Identity(),
                ]
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            ups.append(
                [
                    ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else Identity(),
                ]
            )

        self.downs = downs
        self.ups = ups

        out_dim = default(self.out_dim, self.channels)
        self.final_conv = Sequential.new(
            ConvNextBlock(self.dim, self.dim),
            nn.Conv(out_dim, [1, 1]),
        )

    def __call__(self, x: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        t = self.time_mlp(time) if self.time_mlp is not None else None

        hs = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            hs.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for (convnext, convnext2, attn, upsample), h in zip(self.ups, reversed(hs)):
            x = jnp.concatenate([x, h], axis=-1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
            x

        return self.final_conv(x)
