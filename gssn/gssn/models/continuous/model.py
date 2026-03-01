"""Continuous Flow-SSN model in JAX/Flax.

Port of ContinuousFlowSSN from PyTorch to JAX, using Flax linen modules.
The flow operates in NCHW space (B, K, H, W) where K = num_classes.
"""

from typing import Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def _euler_solve_categorical(
    flow_net: nn.Module,
    u: jnp.ndarray,
    context: jnp.ndarray | None,
    num_steps: int,
    deterministic: bool = True,
) -> jnp.ndarray:
    """Euler ODE solver for categorical flow field, called within nn.Module scope.

    Args:
        flow_net: the flow network module (must be called in a valid Flax scope)
        u: (B, K, H, W) base noise
        context: (B, C, H, W) optional conditioning
        num_steps: number of Euler steps
        deterministic: whether to disable dropout

    Returns:
        y1: (B, K, H, W) solved state at t=1
    """
    dt = 1.0 / num_steps
    y = u
    for i in range(num_steps):
        t_val = i * dt
        t_arr = jnp.full((y.shape[0],), t_val)
        out = flow_net(y, t_arr, context, deterministic=deterministic)
        velocity = jax.nn.softmax(out, axis=1) - u
        y = y + dt * velocity
    return y


class ContinuousFlowSSN(nn.Module):
    """Continuous-time Flow Stochastic Segmentation Network.

    Attributes:
        flow_net: velocity field network (UNet), NCHW in/out
        base_net: optional base distribution network (UNet), NCHW in/out
        num_classes: number of segmentation classes
        cond_base: whether base distribution is conditioned on input
        cond_flow: whether flow is conditioned on input
        base_std: fixed std for base distribution (0 = learned)
    """
    flow_net: nn.Module
    base_net: Optional[nn.Module] = None
    num_classes: int = 2
    cond_base: bool = False
    cond_flow: bool = False
    base_std: float = 1.0

    def setup(self):
        if not self.cond_base:
            input_shape = self.flow_net.input_shape  # (K, H, W)
            self.base_loc = self.param(
                "base_loc", nn.initializers.zeros, input_shape,
            )
            self.base_scale = self.param(
                "base_scale", nn.initializers.ones, input_shape,
            )

    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        mc_samples: int = 1,
        rng: Optional[jax.Array] = None,
        eval_T: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass.

        Args:
            batch: dict with 'x' (B, C, H, W) and optionally 'y' (B, H, W, K)
            mc_samples: number of Monte Carlo samples
            rng: JAX PRNG key
            eval_T: number of ODE solver steps (test only)
            deterministic: if True, disable dropout

        Returns:
            dict with 'loss', 'probs', 'std'
        """
        if rng is None:
            rng = self.make_rng("sample")

        x = batch["x"]  # (B, C, H, W)
        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]

        rng_base, rng_t = jax.random.split(rng)

        # --- Sample from base distribution ---
        base_loc, base_scale = self._get_base_params(x, deterministic)

        if self.cond_base:
            # base_loc: (B, K, H, W)
            noise = jax.random.normal(rng_base, (mc_samples, batch_size, self.num_classes, h, w))
            u = base_loc[None] + base_scale * noise  # broadcast scale
        else:
            # base_loc: (K, H, W)
            noise = jax.random.normal(rng_base, (mc_samples, batch_size, *base_loc.shape))
            u = base_loc[None, None] + base_scale[None, None] * noise

        # (mc*B, K, H, W)
        u = u.reshape(mc_samples * batch_size, self.num_classes, h, w)

        # Context for flow conditioning
        context = None
        if self.cond_flow:
            context = _maybe_expand(x, mc_samples)  # (mc*B, C, H, W)

        loss, probs, std = None, None, 0.0

        if "y" in batch:
            # --- Training ---
            y = batch["y"].astype(jnp.float32)   # (B, H, W, K)
            y = jnp.transpose(y, (0, 3, 1, 2))   # (B, K, H, W)
            y = _maybe_expand(y, mc_samples)      # (mc*B, K, H, W)

            t = jax.random.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)
            t = _maybe_expand(t, mc_samples)      # (mc*B,)

            # Stochastic interpolant: y_t = (1-t)*u + t*y
            t_exp = t.reshape(-1, *([1] * (u.ndim - 1)))
            y_t = (1 - t_exp) * u + t_exp * y

            # Predict and compute loss
            loss, std = self._logit_pred_loss(
                batch["y"], y_t, t, context, mc_samples, deterministic,
            )
        else:
            # --- Inference ---
            y_hat = _euler_solve_categorical(
                self.flow_net, u, context, num_steps=eval_T, deterministic=True,
            )

            # Normalize to probabilities
            probs = y_hat - jnp.min(y_hat, axis=1, keepdims=True)
            probs = probs / jnp.clip(jnp.sum(probs, axis=1, keepdims=True), a_min=1e-7)
            # (mc, B, H, W, K)
            probs = jnp.transpose(probs, (0, 2, 3, 1)).reshape(
                mc_samples, batch_size, h, w, -1
            )

        return {"loss": loss, "probs": probs, "std": std}

    def _get_base_params(
        self, x: jnp.ndarray, deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Any]:
        """Get base distribution parameters (loc, scale)."""
        if not self.cond_base:
            return self.base_loc, self.base_scale
        else:
            out = self.base_net(x, deterministic=deterministic)
            loc, log_scale = jnp.split(out, 2, axis=1)
            if self.base_std:
                scale = self.base_std
            else:
                scale = jnp.exp(jnp.clip(log_scale, a_min=jnp.log(1e-5)))
            return loc, scale

    def _logit_pred_loss(
        self,
        y: jnp.ndarray,       # (B, H, W, K) one-hot
        y_t: jnp.ndarray,     # (mc*B, K, H, W)
        t: jnp.ndarray,       # (mc*B,)
        context: Optional[jnp.ndarray],
        mc_samples: int,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, float]:
        """Compute cross-entropy loss on predicted logits."""
        # (mc*B, K, H, W)
        logits = self.flow_net(y_t, t, context, deterministic=deterministic)

        # (mc, B, H, W, K)
        logits = logits.reshape(mc_samples, -1, *logits.shape[1:])
        logits = jnp.transpose(logits, (0, 1, 3, 4, 2))

        # log_prob of one-hot categorical
        log_softmax = jax.nn.log_softmax(logits, axis=-1)
        log_py = jnp.sum(y * log_softmax, axis=-1)  # (mc, B, H, W)

        # Average over spatial dims -> (mc, B)
        log_prob = jnp.mean(log_py, axis=(-2, -1))

        std = 0.0
        if mc_samples > 1:
            std = float(jnp.exp(log_prob).std(axis=0).mean())
            log_prob = jax.nn.logsumexp(log_prob, axis=0) - jnp.log(mc_samples)

        loss = -jnp.mean(log_prob)
        return loss, std


def _maybe_expand(x: jnp.ndarray, mc_samples: int) -> jnp.ndarray:
    """Expand tensor for MC samples: (B, ...) -> (mc*B, ...)."""
    if mc_samples == 1:
        return x
    expanded = jnp.broadcast_to(x[None, ...], (mc_samples, *x.shape))
    return expanded.reshape(-1, *x.shape[1:])
