#!/usr/bin/env python3
"""
Generate mod_targets.npy for Bonito train_mod from an existing Bonito dataset.

Assumptions:
- references.npy uses Bonito base encoding: 0=pad, 1=A, 2=C, 3=G, 4=T/U
- all valid A positions are labeled as m6A-positive
- padding is written as ignore_value

By default, non-A valid positions are also ignored. This is usually the safer
choice for m6A because only A positions are chemically valid candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BASE_A = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing references.npy and reference_lengths.npy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for mod_targets.npy. Defaults to <dataset-dir>/mod_targets.npy",
    )
    parser.add_argument(
        "--non-a-policy",
        choices=["ignore", "zero"],
        default="ignore",
        help="How to label valid non-A positions: ignore -> -100, zero -> unmodified",
    )
    parser.add_argument(
        "--ignore-value",
        type=int,
        default=-100,
        help="Value used for padding and ignored positions",
    )
    return parser.parse_args()


def load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.load(path)


def build_mod_targets(
    references: np.ndarray,
    lengths: np.ndarray,
    non_a_policy: str,
    ignore_value: int,
) -> np.ndarray:
    if references.ndim != 2:
        raise ValueError(f"references.npy must be 2D, got shape {references.shape}")
    if lengths.ndim != 1:
        raise ValueError(f"reference_lengths.npy must be 1D, got shape {lengths.shape}")
    if references.shape[0] != lengths.shape[0]:
        raise ValueError(
            "references.npy and reference_lengths.npy must share the same first dimension, "
            f"got {references.shape[0]} and {lengths.shape[0]}"
        )
    if lengths.size and int(lengths.max()) > references.shape[1]:
        raise ValueError(
            "reference_lengths.npy exceeds references.npy width: "
            f"max length {int(lengths.max())}, width {references.shape[1]}"
        )

    mod_targets = np.full(references.shape, ignore_value, dtype=np.int16)
    non_a_value = ignore_value if non_a_policy == "ignore" else 0

    for row_idx, length in enumerate(lengths.astype(np.int64)):
        if length <= 0:
            continue
        valid_refs = references[row_idx, :length]
        valid_targets = np.full(valid_refs.shape, non_a_value, dtype=np.int16)
        valid_targets[valid_refs == BASE_A] = 1
        mod_targets[row_idx, :length] = valid_targets

    return mod_targets


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output) if args.output else dataset_dir / "mod_targets.npy"

    references = load_array(dataset_dir / "references.npy", "references.npy")
    lengths = load_array(dataset_dir / "reference_lengths.npy", "reference_lengths.npy")

    mod_targets = build_mod_targets(
        references=references,
        lengths=lengths,
        non_a_policy=args.non_a_policy,
        ignore_value=args.ignore_value,
    )
    np.save(output_path, mod_targets)

    valid_mask = np.arange(references.shape[1])[None, :] < lengths[:, None]
    a_count = int(((references == BASE_A) & valid_mask).sum())
    positive_count = int(((mod_targets == 1) & valid_mask).sum())
    ignored_count = int((mod_targets == args.ignore_value).sum())

    print(f"Saved mod targets to: {output_path}")
    print(f"Shape: {mod_targets.shape}, dtype: {mod_targets.dtype}")
    print(f"Valid A positions labeled modified: {positive_count}")
    print(f"Valid A positions observed in references: {a_count}")
    print(f"Ignored positions: {ignored_count}")
    if args.non_a_policy == "ignore":
        print("Non-A valid positions were ignored.")
    else:
        print("Non-A valid positions were labeled as 0 (unmodified).")


if __name__ == "__main__":
    main()