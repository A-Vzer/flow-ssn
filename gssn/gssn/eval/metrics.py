"""Evaluation metrics in JAX: energy distance, Hungarian-matched IoU, Dice score."""

from typing import Optional, List, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

BackgroundFilter = Optional[Union[bool, List[int]]]


def is_one_hot(x: jnp.ndarray) -> bool:
    return x.ndim >= 2 and bool(jnp.all((x == 0) | (x == 1))) and bool(jnp.all(x.sum(axis=-1) == 1))


def check_inputs(x: jnp.ndarray, filter_bg: BackgroundFilter = None) -> jnp.ndarray:
    if not is_one_hot(x):
        x = x[..., None]
    if filter_bg is not None:
        if isinstance(filter_bg, list):
            x = x[..., filter_bg]
        else:
            x = x[..., :-1]
    return x


def intersection_over_union(
    x: jnp.ndarray,
    y: jnp.ndarray,
    axes: tuple,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """IoU between x and y, reduced over given axes.

    Args:
        x, y: (..., H, W, K) one-hot masks
        axes: spatial axes to reduce over

    Returns:
        (..., ) IoU values
    """
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    intersection = jnp.sum(x * y, axis=axes)
    total_area = jnp.sum(x + y, axis=axes)
    union = total_area - intersection

    if x.shape[-1] > 1:
        iou = (intersection + eps) / (union + eps)
        # Set IoU to NaN where total area is 0
        iou = jnp.where(total_area == 0, jnp.nan, iou)
        return jnp.nanmean(iou, axis=-1)
    else:
        iou = intersection / jnp.maximum(union, eps)
        iou = jnp.where(total_area.squeeze(-1) == 0, 1.0, iou.squeeze(-1))
        return iou


def jaccard_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Pairwise Jaccard distance.

    Args:
        x: (M, B, H, W, K) predictions
        y: (N, B, H, W, K) ground truth

    Returns:
        (M, N, B) Jaccard distances
    """
    m, n = x.shape[0], y.shape[0]
    b = x.shape[1]
    jd = jnp.zeros((m, n, b))

    for i in range(m):
        for j in range(n):
            iou = intersection_over_union(x[i], y[j], axes=(-3, -2))
            jd = jd.at[i, j].set(1 - iou)
    return jd


def energy_distance(
    x: jnp.ndarray,
    y: jnp.ndarray,
    filter_bg: BackgroundFilter = None,
) -> tuple:
    """Compute generalised energy distance (GED).

    Args:
        x: (M, B, H, W, K) MC predictions (one-hot)
        y: (N, B, H, W, K) ground truth raters (one-hot)
        filter_bg: background class filter

    Returns:
        (ged_sq, diversity) both shape (B,)
    """
    x = check_inputs(x, filter_bg)
    y = check_inputs(y, filter_bg)

    d_xy = jaccard_distance(x, y)
    d_xx = jaccard_distance(x, x)
    d_yy = jaccard_distance(y, y)

    d_xy_mean = jnp.mean(d_xy, axis=(0, 1))
    d_xx_mean = jnp.mean(d_xx, axis=(0, 1))
    d_yy_mean = jnp.mean(d_yy, axis=(0, 1))

    ged_sq = 2 * d_xy_mean - d_xx_mean - d_yy_mean
    return ged_sq, d_xx_mean


def hungarian_matched_iou(
    x: jnp.ndarray,
    y: jnp.ndarray,
    filter_bg: BackgroundFilter = None,
) -> jnp.ndarray:
    """Hungarian-matched IoU.

    Args:
        x: (M, B, H, W, K) MC predictions (one-hot)
        y: (N, B, H, W, K) ground truth raters (one-hot)
        filter_bg: background class filter

    Returns:
        (B,) matched IoU scores
    """
    x = check_inputs(x, filter_bg)
    y = check_inputs(y, filter_bg)
    jd = jaccard_distance(x, y)
    cost = np.array(jd)  # move to numpy for scipy

    hm_ious = []
    for i in range(cost.shape[-1]):
        cost_i = cost[:, :, i]
        if np.isnan(cost_i).any():
            continue
        row_idx, col_idx = linear_sum_assignment(cost_i)
        hm_ious.append(np.mean(1 - cost_i[row_idx, col_idx]))

    return jnp.array(hm_ious) if hm_ious else jnp.array([jnp.nan])


def dice_score(
    x: jnp.ndarray,
    y: jnp.ndarray,
    filter_bg: BackgroundFilter = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Dice score.

    Args:
        x: (B, H, W, K) predicted mask (one-hot)
        y: (B, H, W, K) ground truth mask (one-hot)
        filter_bg: background class filter

    Returns:
        (B,) Dice scores
    """
    x = check_inputs(x, filter_bg).astype(jnp.float32)
    y = check_inputs(y, filter_bg).astype(jnp.float32)
    intersection = jnp.sum(x * y, axis=(-3, -2))
    total_area = jnp.sum(x + y, axis=(-3, -2))
    dice = (2 * intersection + eps) / (total_area + eps)
    return jnp.mean(dice, axis=-1)
