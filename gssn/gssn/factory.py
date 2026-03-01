"""Factory functions for building models and argument parsing."""

from typing import Optional, Tuple, Any
import argparse

import flax.linen as nn

from gssn.nn.unet import UNetModel


def continuous_flowssn_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval_T", type=int, default=10)
    parser.add_argument("--cond_base", action="store_true", default=False)
    parser.add_argument("--cond_flow", action="store_true", default=False)
    parser.add_argument("--base_std", type=float, default=0.0)


def unet_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    p = prefix
    parser.add_argument(f"--{p}input_shape", type=int, nargs=3, default=(2, 128, 128))
    parser.add_argument(f"--{p}model_channels", type=int, default=32)
    parser.add_argument(f"--{p}out_channels", type=int, default=192)
    parser.add_argument(f"--{p}num_res_blocks", type=int, default=1)
    parser.add_argument(f"--{p}attention_resolutions", nargs="+", type=int, default=[])
    parser.add_argument(f"--{p}dropout", type=float, default=0.1)
    parser.add_argument(f"--{p}channel_mult", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(f"--{p}num_heads", type=int, default=1)
    parser.add_argument(f"--{p}num_head_channels", type=int, default=64)


def parse_ssn_args(
    name: str,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    if name == "c-flowssn":
        continuous_flowssn_args(parser)
    else:
        raise NotImplementedError(f"Unknown model: {name}")
    return parser.parse_known_args()[0]


def parse_nn_args(
    name: str,
    parser: argparse.ArgumentParser,
    prefix: str = "",
) -> argparse.Namespace:
    if name == "unet":
        unet_args(parser, prefix)
    else:
        raise NotImplementedError(f"Unknown net: {name}")
    return parser.parse_known_args()[0]


def build_unet(args: Any, prefix: str = "") -> UNetModel:
    """Build a UNet from parsed args."""
    p = prefix
    return UNetModel(
        input_shape=tuple(getattr(args, f"{p}input_shape")),
        model_channels=getattr(args, f"{p}model_channels"),
        out_channels=getattr(args, f"{p}out_channels"),
        num_res_blocks=getattr(args, f"{p}num_res_blocks"),
        attention_resolutions=tuple(getattr(args, f"{p}attention_resolutions")),
        dropout=getattr(args, f"{p}dropout"),
        channel_mult=tuple(getattr(args, f"{p}channel_mult")),
        num_heads=getattr(args, f"{p}num_heads"),
        num_head_channels=getattr(args, f"{p}num_head_channels"),
    )


def build_nn(
    name: str,
    parser: Optional[argparse.ArgumentParser] = None,
    args: Optional[argparse.Namespace] = None,
    prefix: str = "",
) -> Tuple[nn.Module, argparse.Namespace]:
    """Build a neural network module from args.

    Args:
        name: network type ("unet")
        parser: argument parser (used if args is None)
        args: pre-parsed args
        prefix: argument prefix (e.g. "base_")

    Returns:
        (model, args) tuple
    """
    if args is None and parser is not None:
        args = parse_nn_args(name, parser, prefix)
    assert args is not None

    if name == "unet":
        model = build_unet(args, prefix)
    else:
        raise NotImplementedError(f"Unknown net: {name}")

    return model, args
