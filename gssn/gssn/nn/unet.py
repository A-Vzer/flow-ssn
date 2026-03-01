"""UNet model for Flow-SSN in JAX/Flax.

All spatial tensors use NHWC (B, H, W, C) internally — JAX/Flax convention.
External API uses NCHW (B, C, H, W) for compatibility with the model layer.
Ported from PyTorch channel-first UNet (guided-diffusion style).
"""

from typing import Optional, Sequence, Tuple

import math
import jax
import jax.numpy as jnp
import flax.linen as nn

from .utils import timestep_embedding


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def group_norm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with automatic group count."""
    num_groups = max(1, num_channels // 16)
    return nn.GroupNorm(num_groups=num_groups)


# --------------------------------------------------------------------------- #
# ResBlock (NHWC)
# --------------------------------------------------------------------------- #

class ResBlock(nn.Module):
    """Residual block with optional timestep embedding and up/down sampling."""
    out_channels: int
    emb_channels: int = 0
    dropout: float = 0.1
    up: bool = False
    down: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        emb: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, C) input
            emb: (B, emb_dim) timestep embedding
        """
        in_channels = x.shape[-1]

        h = group_norm(in_channels)(x)
        h = nn.silu(h)

        if self.up:
            b, hi, wi, c = h.shape
            h = jax.image.resize(h, (b, hi * 2, wi * 2, c), method="nearest")
            x = jax.image.resize(x, (b, hi * 2, wi * 2, in_channels), method="nearest")
        elif self.down:
            h = nn.avg_pool(h, (2, 2), strides=(2, 2))
            x = nn.avg_pool(x, (2, 2), strides=(2, 2))

        h = nn.Conv(self.out_channels, (3, 3), padding="SAME")(h)

        if self.emb_channels > 0 and emb is not None:
            emb_out = nn.silu(emb)
            emb_out = nn.Dense(2 * self.out_channels)(emb_out)
            emb_out = emb_out[:, None, None, :]  # (B, 1, 1, 2C)
            scale, shift = jnp.split(emb_out, 2, axis=-1)
            h = group_norm(self.out_channels)(h) * (1 + scale) + shift
            h = nn.silu(h)
            h = nn.Dropout(rate=self.dropout, deterministic=deterministic)(h)
            h = nn.Conv(
                self.out_channels, (3, 3), padding="SAME",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(h)
        else:
            h = group_norm(self.out_channels)(h)
            h = nn.silu(h)
            h = nn.Dropout(rate=self.dropout, deterministic=deterministic)(h)
            h = nn.Conv(
                self.out_channels, (3, 3), padding="SAME",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(h)

        if in_channels != self.out_channels:
            x = nn.Conv(self.out_channels, (3, 3), padding="SAME")(x)

        return x + h


# --------------------------------------------------------------------------- #
# Attention (NHWC)
# --------------------------------------------------------------------------- #

class AttentionBlock(nn.Module):
    """Multi-head self-attention on (B, H, W, C) feature maps."""
    num_heads: int = 1
    num_head_channels: int = -1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        num_heads = self.num_heads if self.num_head_channels == -1 else c // self.num_head_channels
        head_dim = c // num_heads

        x_flat = x.reshape(b, h * w, c)  # (B, L, C)
        x_norm = nn.GroupNorm(num_groups=max(1, c // 16))(x)
        x_norm = x_norm.reshape(b, h * w, c)

        qkv = nn.Dense(3 * c)(x_norm)  # (B, L, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        scale = 1.0 / math.sqrt(head_dim)

        def to_heads(t):
            return t.reshape(b, -1, num_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        weight = jnp.einsum("bhld,bhsd->bhls", q * scale, k * scale)
        weight = jax.nn.softmax(weight, axis=-1)
        out = jnp.einsum("bhls,bhsd->bhld", weight, v)

        out = out.transpose(0, 2, 1, 3).reshape(b, h * w, c)
        out = nn.Dense(
            c, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
        )(out)

        return (x_flat + out).reshape(b, h, w, c)


# --------------------------------------------------------------------------- #
# UNetModel
# --------------------------------------------------------------------------- #

class UNetModel(nn.Module):
    """UNet with timestep + image conditioning.

    External API: NCHW (B, C, H, W) — transposed to/from NHWC internally.

    Attributes:
        input_shape: (C_in, H, W) — config format, channels-first.
        model_channels: base channel width.
        out_channels: number of output channels.
        num_res_blocks: residual blocks per encoder level.
        attention_resolutions: downsample factors at which to add attention.
        dropout: dropout rate.
        channel_mult: channel multiplier per level.
        num_heads: number of attention heads.
        num_head_channels: channels per attention head (-1 = use num_heads directly).
    """
    input_shape: Tuple[int, int, int] = (2, 128, 128)
    model_channels: int = 32
    out_channels: int = 4
    num_res_blocks: int = 1
    attention_resolutions: Sequence[int] = ()
    dropout: float = 0.0
    channel_mult: Sequence[int] = (1, 2, 4, 8)
    num_heads: int = 1
    num_head_channels: int = 64

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        y: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: (B, C_in, H, W) channel-first input
            t: (B,) timesteps in [0, 1]
            y: (B, C_ctx, H, W) optional context image
            deterministic: disable dropout if True

        Returns:
            (B, out_channels, H, W) channel-first output
        """
        emb_channels = self.model_channels * 4
        ch0 = int(self.channel_mult[0] * self.model_channels)

        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        # --- Time embedding ---
        emb = None
        if t is not None:
            t_emb = timestep_embedding(t, self.model_channels)
            emb = nn.Dense(emb_channels, name="time_dense1")(t_emb)
            emb = nn.silu(emb)
            emb = nn.Dense(emb_channels, name="time_dense2")(emb)

        # --- Context embedding ---
        y_emb = None
        if y is not None:
            y_nhwc = jnp.transpose(y, (0, 2, 3, 1))
            if y_nhwc.shape[-1] == 1:
                y_nhwc = jnp.repeat(y_nhwc, 3, axis=-1)

            if not deterministic:
                drop_mask = (
                    jax.random.uniform(
                        self.make_rng("dropout"), (y_nhwc.shape[0], 1, 1, 1),
                    ) > 0.1
                ).astype(jnp.float32)
                y_nhwc = y_nhwc * drop_mask

            y_emb = nn.Conv(ch0, (1, 1), name="label_conv1")(y_nhwc)
            y_emb = group_norm(ch0)(y_emb)
            y_emb = nn.silu(y_emb)
            y_emb = nn.Conv(ch0, (1, 1), name="label_conv2")(y_emb)

        # --- Stem ---
        h = nn.Conv(ch0, (3, 3), padding="SAME", name="stem")(x)
        if y_emb is not None:
            h = h + y_emb

        # --- Encoder ---
        hs = [h]
        ch = ch0
        res_factor = 1

        for level, mult in enumerate(self.channel_mult):
            out_ch = int(mult * self.model_channels)
            for block_idx in range(self.num_res_blocks):
                h = ResBlock(
                    out_ch, emb_channels, self.dropout,
                    name=f"enc_{level}_{block_idx}",
                )(h, emb, deterministic)
                ch = out_ch

                if res_factor in self.attention_resolutions:
                    h = AttentionBlock(
                        self.num_heads, self.num_head_channels,
                        name=f"enc_attn_{level}_{block_idx}",
                    )(h)

                hs.append(h)

            if level != len(self.channel_mult) - 1:
                h = ResBlock(
                    ch, emb_channels, self.dropout, down=True,
                    name=f"enc_down_{level}",
                )(h, emb, deterministic)
                hs.append(h)
                res_factor *= 2

        # --- Middle ---
        if len(self.attention_resolutions) > 0 and self.attention_resolutions[0] == -1:
            h = ResBlock(ch, emb_channels, self.dropout, name="mid_0")(h, emb, deterministic)
        else:
            h = ResBlock(ch, emb_channels, self.dropout, name="mid_0")(h, emb, deterministic)
            h = AttentionBlock(
                self.num_heads, self.num_head_channels, name="mid_attn",
            )(h)
            h = ResBlock(ch, emb_channels, self.dropout, name="mid_1")(h, emb, deterministic)

        # --- Decoder ---
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            for i in range(self.num_res_blocks + 1):
                skip = hs.pop()
                h = jnp.concatenate([h, skip], axis=-1)  # NHWC concat
                out_ch = int(mult * self.model_channels)

                h = ResBlock(
                    out_ch, emb_channels, self.dropout,
                    name=f"dec_{level}_{i}",
                )(h, emb, deterministic)
                ch = out_ch

                if res_factor in self.attention_resolutions:
                    h = AttentionBlock(
                        self.num_heads, self.num_head_channels,
                        name=f"dec_attn_{level}_{i}",
                    )(h)

                if level > 0 and i == self.num_res_blocks:
                    h = ResBlock(
                        ch, emb_channels, self.dropout, up=True,
                        name=f"dec_up_{level}",
                    )(h, emb, deterministic)
                    res_factor //= 2

        # --- Head ---
        h = group_norm(ch)(h)
        h = nn.silu(h)
        h = nn.Conv(
            self.out_channels, (3, 3), padding="SAME",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="head",
        )(h)

        # NHWC -> NCHW
        return jnp.transpose(h, (0, 3, 1, 2))
