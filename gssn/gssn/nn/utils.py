"""Utility functions: timestep embedding, zero-init, etc."""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn


def zero_init(module_fn, *args, **kwargs):
    """Returns a module with zero-initialized parameters.
    In Flax linen, we handle this via custom initializers instead of post-hoc zeroing."""
    return module_fn(*args, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros, **kwargs)


def timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10000) -> jnp.ndarray:
    """Sinusoidal timestep embedding.

    Args:
        t: (B,) timesteps in [0, 1]
        dim: embedding dimension
        max_period: controls frequency range

    Returns:
        (B, dim) embedding
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
    )
    t_scaled = t * 1000.0  # [0, 1] -> [0, 1000]
    args = t_scaled[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding  # (B, dim)
