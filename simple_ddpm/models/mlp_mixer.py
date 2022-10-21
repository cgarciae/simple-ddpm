import flax.linen as nn
import jax
import jax.numpy as jnp
from einop import einop


class MLP(nn.Module):
    out_dim: int
    units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class MixerBlock(nn.Module):
    mix_patch_size: int
    mix_hidden_size: int

    @nn.compact
    def __call__(self, x):
        _, num_patches, hidden_size = x.shape
        patch_mixer = MLP(num_patches, self.mix_patch_size)
        hidden_mixer = MLP(hidden_size, self.mix_hidden_size)
        norm1 = nn.LayerNorm()
        norm2 = nn.LayerNorm()

        x = einop(x, "... p c -> ... c p")
        x = x + patch_mixer(norm1(x))
        x = einop(x, "... c p -> ... p c")
        x = x + hidden_mixer(norm2(x))
        return x


class MLPMixer(nn.Module):
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
            MixerBlock(self.mix_patch_size, self.mix_hidden_size)
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
