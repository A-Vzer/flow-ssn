"""Training script for Continuous Flow-SSN in JAX."""

from typing import Optional, Dict, Any

import os
import time
import argparse
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
from tqdm import tqdm

from gssn.utils import seed_all, count_params, EMA, create_lr_schedule
from gssn.factory import parse_ssn_args, parse_nn_args, build_nn
from gssn.eval.metrics import energy_distance, hungarian_matched_iou, dice_score
from gssn.models.continuous.model import ContinuousFlowSSN
from gssn.models.gnn.model import GaussianSegmentationNetwork
from gssn.data.lidc import get_lidc, make_dataloader, preprocess_lidc_fn, augment_lidc_batch


# --------------------------------------------------------------------------- #
# Train / Eval Step
# --------------------------------------------------------------------------- #

@functools.partial(jax.jit, static_argnames=("mc_samples", "deterministic"))
def train_step(
    params: Any,
    opt_state: Any,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    mc_samples: int,
    deterministic: bool,
    *,
    model: nn.Module,  # Can be ContinuousFlowSSN or GaussianSegmentationNetwork
    optimizer: optax.GradientTransformation,
):
    """Single training step.

    Returns:
        (new_params, new_opt_state, loss, std, grad_norm)
    """
    def loss_fn(params):
        out = model.apply(
            params,
            batch,
            mc_samples=mc_samples,
            rng=rng,
            deterministic=False,
            rngs={"sample": rng, "dropout": jax.random.fold_in(rng, 1)},
        )
        return out["loss"], out["std"]

    (loss, std), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))

    # Clip gradients
    grads = jax.tree.map(
        lambda g: jnp.where(grad_norm > 10.0, g * 10.0 / grad_norm, g), grads
    )

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, std, grad_norm


def eval_batch(
    batch: Dict[str, jnp.ndarray],
    probs: jnp.ndarray,
    num_classes: int = 2,
) -> Dict[str, jnp.ndarray]:
    """Evaluate a batch of predictions.

    Args:
        batch: must contain 'y' and 'y_all'
        probs: (M, B, H, W, K) probabilities
        num_classes: number of classes

    Returns:
        dict of metric name -> (B,) values
    """
    batch_size = probs.shape[1]
    metrics = {}

    # Mean prediction -> hard mask -> one-hot
    preds_oh = jax.nn.one_hot(jnp.argmax(probs.mean(0), axis=-1), num_classes)
    mc_preds_oh = jax.nn.one_hot(jnp.argmax(probs, axis=-1), num_classes)

    # y_all: (B, H, W, R) -> modes: (R, B, H, W)
    modes = jnp.transpose(batch["y_all"], (3, 0, 1, 2))
    modes_oh = jax.nn.one_hot(modes.astype(jnp.int32), num_classes)

    # Fused ground truth
    fused = jnp.sum(modes_oh, axis=0)
    fused_oh = jax.nn.one_hot(jnp.argmax(fused, axis=-1), num_classes)

    idx = [1]  # foreground class

    ged, div = energy_distance(mc_preds_oh, modes_oh, filter_bg=idx)
    metrics["energy_distance"] = ged
    metrics["diversity"] = div
    metrics["hmiou"] = hungarian_matched_iou(mc_preds_oh, modes_oh, filter_bg=idx)
    metrics["dice"] = dice_score(preds_oh, fused_oh, filter_bg=idx)
    return metrics


def run_eval_epoch(
    model: nn.Module,  # Can be ContinuousFlowSSN or GaussianSegmentationNetwork
    params: Any,
    dataset: Any,
    rng: jax.Array,
    batch_size: int,
    eval_samples: int,
    eval_T: Optional[int] = None,  # Only used by ContinuousFlowSSN
    num_classes: int = 2,
) -> Dict[str, float]:
    """Run evaluation over an entire dataset split."""
    keys = ["energy_distance", "diversity", "hmiou", "dice"]
    metrics = {k: 0.0 for k in keys}
    counts = {k: 0.0 for k in keys}

    rng_loader, rng_eval = jax.random.split(rng)
    loader = make_dataloader(dataset, batch_size, rng_loader, shuffle=False, drop_last=False)
    batches = list(loader)
    pbar = tqdm(batches, total=len(batches), mininterval=1.0)

    for batch in pbar:
        rng_eval, rng_step = jax.random.split(rng_eval)
        rng_pre, rng_model = jax.random.split(rng_step)

        batch = preprocess_lidc_fn(batch, rng=rng_pre)
        bs = batch["x"].shape[0]

        # Micro-batching for memory efficiency
        micro_bs = max(1, min((4 * bs) // eval_samples, bs))

        for i in range(0, bs, micro_bs):
            rng_model, rng_micro = jax.random.split(rng_model)
            micro_batch = {k: v[i:i + micro_bs] for k, v in batch.items()}

            # Build kwargs - only pass eval_T if provided (for ContinuousFlowSSN)
            apply_kwargs = {
                "mc_samples": eval_samples,
                "rng": rng_micro,
                "deterministic": True,
            }
            if eval_T is not None:
                apply_kwargs["eval_T"] = eval_T

            out = model.apply(
                params,
                {"x": micro_batch["x"]},
                **apply_kwargs,
                rngs={"sample": rng_micro},
            )
            probs = out["probs"]
            batch_metrics = eval_batch(micro_batch, probs, num_classes)

            for k, v in batch_metrics.items():
                valid = v[~jnp.isnan(v)]
                metrics[k] += float(jnp.sum(valid))
                counts[k] += int(valid.size)

        desc = ""
        for k, v in metrics.items():
            if counts[k] > 0:
                m = v / counts[k]
                desc += f", {k}: {m:.4f}" if m > 0 else ""
        pbar.set_description(f"[eval]{desc}")

    return {
        k: v / counts[k]
        for k, v in metrics.items()
        if counts[k] > 0 and v / counts[k] != 0
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument("--dataset", type=str, default="lidc")
    parser.add_argument("--data_dir", type=str, default="./datasets/lidc/data_lidc.hdf5")
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    # TRAIN
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup", type=int, default=1000)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    # EVAL
    parser.add_argument("--eval_freq", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=32)
    # MODEL
    parser.add_argument("--model", type=str, choices=["c-flowssn", "gauss"], default="c-flowssn")
    parser.add_argument("--net", type=str, choices=["unet"], default="unet")
    parser.add_argument("--base_net", type=str, choices=["unet", ""], default="")
    parser.add_argument("--band_width", type=int, default=1, help="Banded covariance width (for gauss model)")

    args = parser.parse_known_args()[0]
    
    # Only parse SSN args for flow-based models
    if args.model == "c-flowssn":
        args = parse_ssn_args(args.model, parser)
    
    args = parse_nn_args(args.net, parser)

    if args.base_net:
        args = parse_nn_args(args.base_net, parser, prefix="base_")

    # --- Seed ---
    rng = seed_all(args.seed)

    # --- Data ---
    if args.dataset == "lidc":
        datasets = get_lidc(args)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    # --- Build model ---
    if args.model == "c-flowssn":
        # Flow-based model
        flow_net = build_nn(args.net, args=args)[0]

        base_net = None
        if args.base_net and args.cond_base:
            base_net = build_nn(args.base_net, args=args, prefix="base_")[0]

        model = ContinuousFlowSSN(
            flow_net=flow_net,
            base_net=base_net,
            num_classes=args.num_classes,
            cond_base=args.cond_base,
            cond_flow=args.cond_flow,
            base_std=args.base_std,
        )
    elif args.model == "gauss":
        # Gaussian Segmentation Network (no flow)
        base_net = None
        if args.base_net and hasattr(args, 'cond_base') and args.cond_base:
            base_net = build_nn(args.base_net, args=args, prefix="base_")[0]
        elif args.base_net:
            # Default to conditional if not specified
            base_net = build_nn(args.base_net, args=args, prefix="base_")[0]
            args.cond_base = True
        else:
            args.cond_base = True  # Default for gauss model

        # Set defaults for gauss model if not present
        if not hasattr(args, 'base_std'):
            args.base_std = 1.0

        model = GaussianSegmentationNetwork(
            base_net=base_net,
            num_classes=args.num_classes,
            cond_base=args.cond_base,
            base_std=args.base_std,
            band_width=args.band_width,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --- Initialize parameters ---
    rng, init_rng = jax.random.split(rng)
    dummy_batch = {
        "x": jnp.zeros((1, args.img_channels, args.resolution, args.resolution)),
        "y": jnp.zeros((1, args.resolution, args.resolution, args.num_classes)),
    }
    params = model.init(
        {"params": init_rng, "sample": init_rng, "dropout": init_rng},
        dummy_batch,
        mc_samples=1,
        rng=init_rng,
        deterministic=False,
    )

    print(f"Total params: {count_params(params):,}")

    # --- Optimizer ---
    lr_schedule = create_lr_schedule(args.lr, args.lr_warmup, total_steps=0)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(lr_schedule, weight_decay=args.wd),
    )
    opt_state = optimizer.init(params)

    # --- EMA ---
    ema = EMA(params, rate=args.ema_rate) if args.ema_rate > 0 else None

    # --- WandB ---
    wandb.init(project="gssn", name=args.exp_name, config=vars(args))

    for k, v in vars(args).items():
        print(f"--{k}={v}")

    # --- Save dir ---
    save_path = f"checkpoints/{args.exp_name}/"
    os.makedirs(save_path, exist_ok=True)

    best_ged = 1.0
    best_metric = 0.0
    track_metric = "dice"

    start_t = time.time()

    # --- Training loop ---
    mininterval = float(os.environ.get("TQDM_MININTERVAL", "1"))

    for epoch in range(args.epochs):
        rng, rng_epoch = jax.random.split(rng)
        rng_loader, rng_train = jax.random.split(rng_epoch)

        loader = make_dataloader(
            datasets["train"], args.bs, rng_loader, shuffle=True, drop_last=True,
        )
        batches = list(loader)
        pbar = tqdm(batches, total=len(batches), mininterval=mininterval)

        epoch_loss = 0.0
        epoch_count = 0

        for batch in pbar:
            rng_train, rng_step = jax.random.split(rng_train)
            rng_aug, rng_pre, rng_model = jax.random.split(rng_step, 3)

            # Augment + preprocess
            batch = augment_lidc_batch(batch, rng_aug)
            batch = preprocess_lidc_fn(batch, rng=rng_pre)

            bs = batch["x"].shape[0]

            # JIT-compiled train step
            _train_step = functools.partial(
                train_step, model=model, optimizer=optimizer,
            )
            params, opt_state, loss, std, grad_norm = _train_step(
                params, opt_state, batch, rng_model,
                mc_samples=args.mc_samples, deterministic=False,
            )

            epoch_loss += float(loss) * bs
            epoch_count += bs

            if ema is not None:
                ema.update(params)

            avg_loss = epoch_loss / epoch_count
            desc = f"[train] loss: {avg_loss:.4f}, std: {float(std):.1e}, gnorm: {float(grad_norm):.2f}"
            pbar.set_description(desc)

            wandb.log({
                "loss": float(loss),
                "grad_norm": float(grad_norm),
                "mc_std": float(std),
            })

        train_loss = epoch_loss / max(epoch_count, 1)
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        # --- Evaluation ---
        if epoch > 0 and (epoch % args.eval_freq) == 0:
            rng, rng_eval = jax.random.split(rng)
            eval_params = ema.get() if ema is not None else params

            # Only pass eval_T for flow-based models
            eval_kwargs = {
                "batch_size": args.bs,
                "eval_samples": args.eval_samples,
                "num_classes": args.num_classes,
            }
            if args.model == "c-flowssn" and hasattr(args, 'eval_T'):
                eval_kwargs["eval_T"] = args.eval_T
            
            valid_metrics = run_eval_epoch(
                model, eval_params, datasets["valid"], rng_eval,
                **eval_kwargs,
            )
            wandb.log({"valid_" + k: v for k, v in valid_metrics.items()})

            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))
            t = time.strftime("[%H:%M:%S]")
            print(
                f"\n{t} epoch: {epoch + 1}, elapsed: {elapsed}"
                f"\n{t} train loss: {train_loss:.5f}"
                f"\n{t} valid {', '.join(f'{k}: {v:.5f}' for k, v in valid_metrics.items())}"
            )

            # Save checkpoints (as numpy for portability)
            import pickle
            save_dict = {
                "args": vars(args),
                "epoch": epoch,
                "params": jax.device_get(eval_params),
            }

            if track_metric in valid_metrics and valid_metrics[track_metric] > best_metric:
                best_metric = valid_metrics[track_metric]
                path = os.path.join(save_path, f"checkpoint_{track_metric}.pkl")
                with open(path, "wb") as f:
                    pickle.dump({**save_dict, track_metric: best_metric}, f)
                print(f"{t} model saved: {path}")

            if "energy_distance" in valid_metrics and valid_metrics["energy_distance"] < best_ged:
                best_ged = valid_metrics["energy_distance"]
                path = os.path.join(save_path, "checkpoint_ged.pkl")
                with open(path, "wb") as f:
                    pickle.dump({**save_dict, "ged": best_ged}, f)
                print(f"{t} model saved: {path}")

    # --- Final test ---
    import pickle
    t = time.strftime("[%H:%M:%S]")
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))

    for ckpt_metric in [track_metric, "ged"]:
        load_path = os.path.join(save_path, f"checkpoint_{ckpt_metric}.pkl")
        if not os.path.exists(load_path):
            print(f"{t} checkpoint not found: {load_path}")
            continue

        print(f"\n{t} Loading checkpoint: {load_path}")
        with open(load_path, "rb") as f:
            ckpt = pickle.load(f)

        rng, rng_test = jax.random.split(rng)
        
        # Only pass eval_T for flow-based models
        test_kwargs = {
            "batch_size": args.bs,
            "eval_samples": 100,
            "num_classes": args.num_classes,
        }
        if args.model == "c-flowssn":
            test_kwargs["eval_T"] = 50
        
        test_metrics = run_eval_epoch(
            model, ckpt["params"], datasets["test"], rng_test,
            **test_kwargs,
        )
        print(f"{t} test ({ckpt_metric}): {', '.join(f'{k}: {v:.5f}' for k, v in test_metrics.items())}")
        print(f"{t} training_time: {elapsed}")


if __name__ == "__main__":
    main()
