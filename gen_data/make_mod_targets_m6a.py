#!/usr/bin/env python3
"""
Generate mod_targets.npy for Bonito train_mod from an existing Bonito dataset.

Assumptions:
- references.npy uses Bonito base encoding: 0=pad, 1=A, 2=C, 3=G, 4=T/U
- global labels follow the multi-head RNA mod scheme:
  -100=ignore, 0=canonical_A, 1=canonical_C, 2=canonical_G, 3=canonical_T/U, 4=m6A
- full-mod mode labels all valid A positions as m6A-positive (4)
- canonical mode labels all valid A positions as canonical_A (0)
- padding is written as ignore_value

By default, valid non-A positions are labeled with their canonical base ids so the
result matches the global mod target label space used by the multi-head model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BASE_A = 1
BASE_C = 2
BASE_G = 3
BASE_T = 4
MODE_FULL_MOD = "full-mod"
MODE_CANONICAL = "canonical"
NON_A_POLICY_IGNORE = "ignore"
NON_A_POLICY_CANONICAL = "canonical"
NON_A_POLICY_LEGACY_ZERO = "zero"
CANONICAL_LABELS = {
    BASE_A: 0,
    BASE_C: 1,
    BASE_G: 2,
    BASE_T: 3,
}
M6A_LABEL = 4


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
        "--mode",
        choices=[MODE_FULL_MOD, MODE_CANONICAL],
        default=MODE_FULL_MOD,
        help=(
            "Target labeling mode: full-mod -> label all valid A positions as modified, "
            "canonical -> label all valid A positions as unmodified"
        ),
    )
    parser.add_argument(
        "--non-a-policy",
        choices=[NON_A_POLICY_IGNORE, NON_A_POLICY_CANONICAL, NON_A_POLICY_LEGACY_ZERO],
        default=NON_A_POLICY_CANONICAL,
        help=(
            "How to label valid non-A positions: ignore -> -100, canonical -> canonical_C/G/T "
            "(legacy alias: zero)"
        ),
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
    mode: str,
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
    canonical_non_a_policy = NON_A_POLICY_CANONICAL if non_a_policy == NON_A_POLICY_LEGACY_ZERO else non_a_policy
    a_value = M6A_LABEL if mode == MODE_FULL_MOD else CANONICAL_LABELS[BASE_A]

    for row_idx, length in enumerate(lengths.astype(np.int64)):
        if length <= 0:
            continue
        valid_refs = references[row_idx, :length]
        valid_targets = np.full(valid_refs.shape, ignore_value, dtype=np.int16)
        if canonical_non_a_policy == NON_A_POLICY_CANONICAL:
            for base_value, label_value in CANONICAL_LABELS.items():
                valid_targets[valid_refs == base_value] = label_value
        valid_targets[valid_refs == BASE_A] = a_value
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
        mode=args.mode,
        non_a_policy=args.non_a_policy,
        ignore_value=args.ignore_value,
    )
    np.save(output_path, mod_targets)

    valid_mask = np.arange(references.shape[1])[None, :] < lengths[:, None]
    a_count = int(((references == BASE_A) & valid_mask).sum())
    base_counts = {
        "A": int(((references == BASE_A) & valid_mask).sum()),
        "C": int(((references == BASE_C) & valid_mask).sum()),
        "G": int(((references == BASE_G) & valid_mask).sum()),
        "T/U": int(((references == BASE_T) & valid_mask).sum()),
    }
    positive_count = int(((mod_targets == M6A_LABEL) & valid_mask).sum())
    ignored_count = int((mod_targets == args.ignore_value).sum())
    canonical_counts = {
        "canonical_A": int(((mod_targets == CANONICAL_LABELS[BASE_A]) & valid_mask).sum()),
        "canonical_C": int(((mod_targets == CANONICAL_LABELS[BASE_C]) & valid_mask).sum()),
        "canonical_G": int(((mod_targets == CANONICAL_LABELS[BASE_G]) & valid_mask).sum()),
        "canonical_T": int(((mod_targets == CANONICAL_LABELS[BASE_T]) & valid_mask).sum()),
    }

    print(f"Saved mod targets to: {output_path}")
    print(f"Shape: {mod_targets.shape}, dtype: {mod_targets.dtype}")
    print(f"Valid A positions observed in references: {a_count}")
    print(f"Valid A positions labeled modified: {positive_count}")
    print(f"Canonical label counts: {canonical_counts}")
    print(f"Reference base counts: {base_counts}")
    print(f"Ignored positions: {ignored_count}")
    if args.mode == MODE_FULL_MOD:
        print("Mode: full-mod (all valid A positions were labeled as m6A / global id 4).")
    else:
        print("Mode: canonical (all valid A positions were labeled as canonical_A / global id 0).")
    if args.non_a_policy == NON_A_POLICY_IGNORE:
        print("Non-A valid positions were ignored.")
    else:
        print("Non-A valid positions were labeled with canonical_C/G/T global ids.")
        if args.non_a_policy == NON_A_POLICY_LEGACY_ZERO:
            print("Note: --non-a-policy zero is treated as a legacy alias for canonical.")


if __name__ == "__main__":
    main()
