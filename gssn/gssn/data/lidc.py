"""LIDC-IDRI dataset loading and preprocessing for JAX."""

from typing import Optional, Callable, Dict, Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np


class LIDC:
    """LIDC-IDRI dataset (in-memory)."""

    def __init__(self, root: str, split: str, resolution: int = 128):
        self.split = split[:3] if split == "valid" else split
        print(f"Loading LIDC {self.split}:")

        with h5py.File(root, "r") as f:
            data = f[self.split]
            images = np.array(data["images"][:], dtype=np.float32)  # type: ignore
            labels = np.array(data["labels"][:], dtype=np.uint8)  # type: ignore

        # images: (N, H, W) -> (N, 1, H, W), in [0, 1]
        self.images = (images + 0.5)[:, None, ...]
        # labels: (N, 4, H, W) -> keep as (N, 4, H, W) raters-first
        self.labels = np.moveaxis(labels, 3, 1)
        print(f"images: {self.images.shape}, masks: {self.labels.shape}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {"x": self.images[idx], "y": self.labels[idx]}


def get_lidc(args: Any) -> Dict[str, LIDC]:
    """Build train/valid/test LIDC datasets."""
    return {
        k: LIDC(root=args.data_dir, split=k, resolution=args.resolution)
        for k in ["train", "valid", "test"]
    }


def make_dataloader(
    dataset: LIDC,
    batch_size: int,
    rng: jax.Array,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Generator that yields batches of JAX arrays from a LIDC dataset."""
    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        perm = jax.random.permutation(rng, n)
        indices = np.array(perm)

    num_batches = n // batch_size if drop_last else (n + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        x = dataset.images[batch_idx]
        y = dataset.labels[batch_idx]
        yield {"x": jnp.array(x), "y": jnp.array(y)}


def augment_lidc_batch(
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
) -> Dict[str, jnp.ndarray]:
    """Random augmentation: rotation (0/90/180/270), horizontal/vertical flip.

    Args:
        batch: dict with 'x' (B, C, H, W) and 'y' (B, R, H, W)
        rng: JAX PRNG key

    Returns:
        Augmented batch.
    """
    rng_rot, rng_hflip, rng_vflip = jax.random.split(rng, 3)
    x, y = batch["x"], batch["y"]
    b = x.shape[0]

    # Random rotation in {0, 1, 2, 3} * 90 degrees, per sample
    # Use jax.lax.switch instead of jnp.rot90 (which can't handle traced k)
    rot_k = jax.random.randint(rng_rot, (b,), 0, 4)

    def _rot90_fixed(arr, k, axes=(1, 2)):
        """rot90 with traced k using lax.switch."""
        branches = [
            lambda a: a,                                          # k=0
            lambda a: jnp.flip(jnp.swapaxes(a, *axes), axes[0]), # k=1
            lambda a: jnp.flip(jnp.flip(a, axes[0]), axes[1]),   # k=2
            lambda a: jnp.flip(jnp.swapaxes(a, *axes), axes[1]), # k=3
        ]
        return jax.lax.switch(k, branches, arr)

    def rotate_sample(x_i, y_i, k):
        # x_i: (C, H, W), y_i: (R, H, W)
        x_i = _rot90_fixed(x_i, k)
        y_i = _rot90_fixed(y_i, k)
        return x_i, y_i

    x, y = jax.vmap(rotate_sample)(x, y, rot_k)

    # Random horizontal flip per sample
    hflip = jax.random.bernoulli(rng_hflip, 0.5, (b,))

    def maybe_hflip(x_i, y_i, do_flip):
        x_i = jnp.where(do_flip, jnp.flip(x_i, axis=2), x_i)
        y_i = jnp.where(do_flip, jnp.flip(y_i, axis=2), y_i)
        return x_i, y_i

    x, y = jax.vmap(maybe_hflip)(x, y, hflip)

    # Random vertical flip per sample
    vflip = jax.random.bernoulli(rng_vflip, 0.5, (b,))

    def maybe_vflip(x_i, y_i, do_flip):
        x_i = jnp.where(do_flip, jnp.flip(x_i, axis=1), x_i)
        y_i = jnp.where(do_flip, jnp.flip(y_i, axis=1), y_i)
        return x_i, y_i

    x, y = jax.vmap(maybe_vflip)(x, y, vflip)

    return {"x": x, "y": y}


def preprocess_lidc_fn(
    batch: Dict[str, jnp.ndarray],
    rng: Optional[jax.Array] = None,
) -> Dict[str, jnp.ndarray]:
    """Preprocess LIDC batch: scale images to [-1,1], sample random rater, one-hot encode.

    Args:
        batch: dict with 'x' (B, C, H, W) in [0,1] and 'y' (B, R, H, W)
        rng: JAX PRNG key (used to sample random rater)

    Returns:
        Preprocessed batch with keys: 'x', 'y' (one-hot), 'y_all'.
    """
    x = batch["x"] * 2 - 1  # [0,1] -> [-1,1]
    y = batch["y"]  # (B, R, H, W)
    # (B, H, W, R) - raters last
    y = jnp.transpose(y, (0, 2, 3, 1))
    b = y.shape[0]
    r = y.shape[-1]
    y_all = y.astype(jnp.float32)

    # sample a random rater per element
    if rng is None:
        rng = jax.random.PRNGKey(0)
    idx = jax.random.randint(rng, (b,), 0, r)

    # (B, H, W) - one rater per sample
    y_single = y[jnp.arange(b), :, :, idx]
    # (B, H, W, K) one-hot
    y_oh = jax.nn.one_hot(y_single.astype(jnp.int32), 2)

    return {"x": x, "y": y_oh, "y_all": y_all}
