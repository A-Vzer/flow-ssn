"""Base Segmentation Model with Banded Gaussian Distribution.

A simplified version of Flow-SSN that predicts segmentation directly from
a learned banded Gaussian distribution without flow augmentation.
The banded structure introduces spatial correlations between nearby pixels.
"""

from typing import Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


class GaussianSegmentationNetwork(nn.Module):
    """Gaussian Segmentation Network with banded covariance (no flow).

    Predicts segmentation masks by sampling from a banded Gaussian distribution
    parameterized by a base network (UNet). The banded structure introduces
    spatial correlations between pixels within a specified bandwidth.

    Attributes:
        base_net: backbone network (UNet) that predicts distribution parameters
        num_classes: number of segmentation classes
        cond_base: whether base distribution is conditioned on input
        base_std: fixed std for base distribution (0 = learned from network)
        band_width: spatial bandwidth for covariance structure (0 = diagonal only)
                    band_width=1 includes immediate neighbors (3x3 kernel)
                    band_width=2 includes 5x5 neighborhood, etc.
    """
    base_net: nn.Module
    num_classes: int = 2
    cond_base: bool = True
    base_std: float = 1.0
    band_width: int = 1

    def setup(self):
        if not self.cond_base:
            # Unconditional: learnable parameters for entire spatial distribution
            input_shape = getattr(self.base_net, 'input_shape', None)
            if input_shape is None:
                # Fallback if base_net doesn't have input_shape
                raise ValueError("base_net must have input_shape attribute for unconditional base")
            # Assume input_shape is (C, H, W), we need (K, H, W)
            spatial_shape = (self.num_classes, input_shape[1], input_shape[2])
            self.base_loc = self.param(
                "base_loc", nn.initializers.zeros, spatial_shape,
            )
            self.base_scale = self.param(
                "base_scale", nn.initializers.ones, spatial_shape,
            )

        # Banded covariance: learnable spatial correlation kernel
        if self.band_width > 0:
            kernel_size = 2 * self.band_width + 1
            bw = self.band_width  # Capture in closure
            
            # One correlation kernel per class (depthwise convolution)
            # Initialize with Gaussian-like decay from center
            def correlation_init(key, shape, dtype=jnp.float32):
                """Initialize correlation kernel with Gaussian-like decay.
                
                For depthwise conv: shape = (kernel_h, kernel_w, in_per_group, out_channels)
                where in_per_group = in_channels // feature_group_count = 1
                """
                k_size, _, in_per_group, out_channels = shape
                center = k_size // 2
                # Create base 2D kernel with Gaussian decay
                kernel_2d = jnp.zeros((k_size, k_size), dtype=dtype)
                for i in range(k_size):
                    for j in range(k_size):
                        dist = jnp.sqrt((i - center)**2 + (j - center)**2)
                        kernel_2d = kernel_2d.at[i, j].set(jnp.exp(-dist**2 / (2 * bw**2)))
                # Normalize
                kernel_2d = kernel_2d / kernel_2d.sum()
                # Expand to (k_size, k_size, 1, out_channels)
                kernel_4d = kernel_2d[:, :, None, None]  # (k_size, k_size, 1, 1)
                kernel_4d = jnp.tile(kernel_4d, (1, 1, 1, out_channels))  # (k_size, k_size, 1, out_channels)
                return kernel_4d
            
            self.correlation_kernel = self.param(
                "correlation_kernel",
                correlation_init,
                (kernel_size, kernel_size, 1, self.num_classes)
            )

    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        mc_samples: int = 1,
        rng: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass.

        Samples from a banded Gaussian distribution where spatial correlations
        are introduced via a learnable convolution kernel with bandwidth determined
        by band_width parameter.

        Args:
            batch: dict with 'x' (B, C, H, W) and optionally 'y' (B, H, W, K)
            mc_samples: number of Monte Carlo samples
            rng: JAX PRNG key
            deterministic: if True, disable dropout

        Returns:
            dict with 'loss', 'probs', 'std'
        """
        if rng is None:
            rng = self.make_rng("sample")

        x = batch["x"]  # (B, C, H, W)
        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]

        # --- Get base distribution parameters ---
        base_loc, base_scale = self._get_base_params(x, deterministic)

        # --- Sample from base distribution ---
        if self.cond_base:
            # base_loc: (B, K, H, W)
            noise = jax.random.normal(rng, (mc_samples, batch_size, self.num_classes, h, w))
            
            # Apply banded covariance via spatial correlation
            if self.band_width > 0:
                # Apply depthwise convolution for spatial correlations per class
                # Reshape for convolution: (mc*B, K, H, W)
                noise_flat = noise.reshape(-1, self.num_classes, h, w)
                # Transpose to NHWC for JAX conv
                noise_nhwc = jnp.transpose(noise_flat, (0, 2, 3, 1))  # (mc*B, H, W, K)
                
                # Apply correlation kernel via depthwise convolution
                # Use feature_group_count for depthwise conv (one kernel per class)
                # Explicitly specify dimension_numbers for NHWC format
                noise_corr = jax.lax.conv_general_dilated(
                    noise_nhwc,
                    self.correlation_kernel,
                    window_strides=(1, 1),
                    padding='SAME',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                    feature_group_count=self.num_classes,
                )
                # Transpose back to NCHW
                noise_corr = jnp.transpose(noise_corr, (0, 3, 1, 2))  # (mc*B, K, H, W)
                # Reshape back to (mc, B, K, H, W)
                noise = noise_corr.reshape(mc_samples, batch_size, self.num_classes, h, w)
            
            u = base_loc[None] + base_scale * noise  # broadcast scale
        else:
            # base_loc: (K, H, W)
            noise = jax.random.normal(rng, (mc_samples, batch_size, *base_loc.shape))
            
            # Apply banded covariance via spatial correlation
            if self.band_width > 0:
                noise_flat = noise.reshape(-1, self.num_classes, h, w)
                noise_nhwc = jnp.transpose(noise_flat, (0, 2, 3, 1))
                noise_corr = jax.lax.conv_general_dilated(
                    noise_nhwc,
                    self.correlation_kernel,
                    window_strides=(1, 1),
                    padding='SAME',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                    feature_group_count=self.num_classes,
                )
                noise_corr = jnp.transpose(noise_corr, (0, 3, 1, 2))
                noise = noise_corr.reshape(mc_samples, batch_size, self.num_classes, h, w)
            
            u = base_loc[None, None] + base_scale[None, None] * noise

        # --- Convert to probabilities via softmax ---
        # u: (mc, B, K, H, W)
        probs = jax.nn.softmax(u, axis=2)  # softmax over class dimension

        # Transpose to (mc, B, H, W, K) for consistency with evaluation format
        probs = jnp.transpose(probs, (0, 1, 3, 4, 2))

        loss, std = None, 0.0

        if "y" in batch:
            # --- Training: compute cross-entropy loss ---
            y = batch["y"].astype(jnp.float32)  # (B, H, W, K) one-hot

            # Compute log probabilities
            log_probs = jax.nn.log_softmax(u, axis=2)  # (mc, B, K, H, W)
            log_probs = jnp.transpose(log_probs, (0, 1, 3, 4, 2))  # (mc, B, H, W, K)

            # Cross-entropy: sum over classes (one-hot * log_prob)
            log_py = jnp.sum(y[None] * log_probs, axis=-1)  # (mc, B, H, W)

            # Average over spatial dimensions -> (mc, B)
            log_prob = jnp.mean(log_py, axis=(-2, -1))

            # Monte Carlo averaging
            if mc_samples > 1:
                std = float(jnp.exp(log_prob).std(axis=0).mean())
                log_prob = jax.nn.logsumexp(log_prob, axis=0) - jnp.log(mc_samples)

            loss = -jnp.mean(log_prob)

        return {"loss": loss, "probs": probs, "std": std}

    def _get_base_params(
        self, x: jnp.ndarray, deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Any]:
        """Get base distribution parameters (loc, scale).

        The returned parameters define the diagonal components of the distribution.
        Spatial correlations (banded structure) are introduced by the correlation
        kernel during sampling.

        Args:
            x: (B, C, H, W) input images
            deterministic: whether to disable dropout

        Returns:
            loc: (B, K, H, W) if conditional, else (K, H, W) - mean per pixel
            scale: scalar or array matching loc shape - diagonal std per pixel
        """
        if not self.cond_base:
            # Unconditional: return learnable parameters
            return self.base_loc, self.base_scale
        else:
            # Conditional: predict from base_net
            out = self.base_net(x, deterministic=deterministic)  # (B, 2*K, H, W)
            loc, log_scale = jnp.split(out, 2, axis=1)  # each (B, K, H, W)

            if self.base_std:
                # Use fixed standard deviation
                scale = self.base_std
            else:
                # Learn standard deviation from network
                scale = jnp.exp(jnp.clip(log_scale, a_min=jnp.log(1e-5)))

            return loc, scale
