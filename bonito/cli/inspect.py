"""
Inspect a model defined in a TOML config using torchinfo.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import toml
import torch
from torchinfo import summary

from bonito.util import load_symbol


def main(args):
    config_path = Path(args.config).resolve()
    config = toml.load(config_path)
    config["__config_dir__"] = str(config_path.parent)

    model = load_symbol(config, "Model")(config)
    model.to(args.device)

    input_size = (
        args.batch_size,
        args.input_features or config["input"]["features"],
        args.signal_length,
    )

    summary(
        model,
        input_size=input_size,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=args.depth,
        device=args.device,
        verbose=1,
    )


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("config", help="Path to a config.toml defining the model.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--signal-length", type=int, default=4000)
    parser.add_argument(
        "--input-features",
        type=int,
        default=None,
        help="Override input feature size; defaults to config.input.features",
    )
    parser.add_argument("--depth", type=int, default=4)
    return parser
