
import flax.linen as nn
import jax
import jax.numpy as jnp


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
