"""ODE solvers for continuous normalizing flows in JAX."""

from typing import Callable

import jax
import jax.numpy as jnp


def euler_solve(
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    y0: jnp.ndarray,
    t0: float = 0.0,
    t1: float = 1.0,
    num_steps: int = 10,
) -> jnp.ndarray:
    """Forward Euler ODE solver.

    Args:
        fn: vector field function(t, y) -> dy/dt
        y0: initial state (B, C, H, W)
        t0: start time
        t1: end time
        num_steps: number of Euler steps

    Returns:
        y1: final state at t=t1
    """
    dt = (t1 - t0) / num_steps
    y = y0

    def step(y, i):
        t = t0 + i * dt
        t_arr = jnp.full((y.shape[0],), t)
        dy = fn(t_arr, y)
        y_next = y + dt * dy
        return y_next, None

    y, _ = jax.lax.scan(step, y, jnp.arange(num_steps))
    return y


def ode_solve(
    model_apply: Callable,
    params: dict,
    u: jnp.ndarray,
    context: jnp.ndarray | None = None,
    field: str = "categorical",
    num_steps: int = 10,
    t0: float = 0.0,
    t1: float = 1.0,
) -> jnp.ndarray:
    """Solve ODE for flow-SSN.

    Args:
        model_apply: function that applies the flow network: (params, x, t, context) -> output
        params: flow network parameters
        u: (B, K, H, W) initial noise sample
        context: (B, C, H, W) optional conditioning image
        field: "categorical" applies softmax - u correction, "unconstrained" uses raw output
        num_steps: Euler steps
        t0: start time
        t1: end time

    Returns:
        y_hat: (B, K, H, W) predicted output at t=t1
    """
    if field == "categorical":
        def fn(t, y):
            out = model_apply(params, y, t, context)
            return jax.nn.softmax(out, axis=1) - u
    elif field == "unconstrained":
        def fn(t, y):
            return model_apply(params, y, t, context)
    else:
        raise ValueError(f"Unknown field type: {field}")

    return euler_solve(fn, u, t0, t1, num_steps)
