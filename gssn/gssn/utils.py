"""Shared utilities: seeding, parameter counting, EMA."""

from typing import Any

import copy
import random

import jax
import jax.numpy as jnp
import numpy as np
import optax


def seed_all(seed: int) -> jax.Array:
    """Seed all RNGs and return a JAX PRNG key."""
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def count_params(params: Any) -> int:
    """Count total number of parameters in a pytree."""
    return sum(p.size for p in jax.tree.leaves(params))


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, params: Any, rate: float = 0.999):
        self.rate = rate
        self.ema_params = copy.deepcopy(params)

    def update(self, params: Any) -> None:
        """Update EMA parameters."""
        self.ema_params = jax.tree.map(
            lambda ema, p: self.rate * ema + (1 - self.rate) * p,
            self.ema_params,
            params,
        )

    def get(self) -> Any:
        """Return current EMA parameters."""
        return self.ema_params


def create_lr_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """Linear warmup then constant learning rate."""
    warmup = optax.linear_schedule(
        init_value=base_lr / max(warmup_steps, 1),
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    return warmup
