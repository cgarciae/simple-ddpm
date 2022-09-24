import flax.linen as nn
import jax
import jax.numpy as jnp
from einop import einop

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
