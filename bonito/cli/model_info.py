"""
Torchinfo helper to inspect Bonito basecaller models.
"""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from torchinfo import summary

from bonito.cli.download import Downloader, __models_dir__, models
from bonito.util import load_model


def _ensure_model_available(model_directory: str) -> None:
    """Download a named model if it is not already cached locally."""

    if model_directory in models and not (__models_dir__ / model_directory).exists():
        sys.stderr.write("> downloading model\n")
        Downloader(__models_dir__).download(model_directory)


def main(args):
    _ensure_model_available(args.model_directory)
    sys.stderr.write(f"> loading model {args.model_directory}\n")
    model = load_model(
        args.model_directory,
        args.device,
        weights=args.weights if args.weights > 0 else None,
        half=args.half,
        chunksize=args.chunksize,
        overlap=args.overlap,
        batchsize=args.batchsize,
        quantize=args.quantize,
        use_koi=args.use_koi,
        compile=False,
    )

    input_size = (
        args.batch_size,
        model.config["input"]["features"],
        args.signal_length,
    )

    summary(
        model,
        input_size=input_size,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=args.depth,
        device=args.device,
        verbose=1,
        kwargs={"return_all": args.return_all_heads} if args.return_all_heads else {},
    )


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("model_directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", default=0, type=int, help="Use a specific checkpoint number; defaults to latest.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for the torchinfo input shape.")
    parser.add_argument("--signal-length", type=int, default=4000, help="Signal length for the torchinfo input shape.")
    parser.add_argument("--depth", type=int, default=4, help="Maximum module depth shown in the summary.")
    parser.add_argument(
        "--return-all-heads",
        action="store_true",
        help="Request all model heads (if supported) instead of just the basecall head during summary.",
    )
    parser.add_argument("--half", action="store_true", help="Load the model in half precision.")
    parser.add_argument("--use-koi", action="store_true", help="Enable k-mer only inference mode adjustments.")
    quant_group = parser.add_mutually_exclusive_group(required=False)
    quant_group.add_argument("--quantize", dest="quantize", action="store_true")
    quant_group.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--chunksize", default=None, type=int, help="Override model chunksize before summary.")
    parser.add_argument("--overlap", default=None, type=int, help="Override model overlap before summary.")
    parser.add_argument("--batchsize", default=None, type=int, help="Override basecaller batchsize before summary.")
    return parser
