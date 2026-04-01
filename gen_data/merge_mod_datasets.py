#!/usr/bin/env python3
"""
Merge full-mod and canonical Bonito mod-training datasets into one mixed dataset.

Design goals:
- keep memory usage low for large datasets
- read inputs with mmap
- write outputs directly with memmap
- avoid a second full-array shuffle pass by emitting indices.npy

Usage:
    python gen_data/merge_mod_datasets.py \
        --full-mod-dir /path/to/full_mod_dataset \
        --canonical-dir /path/to/canonical_dataset \
        --output-dir /path/to/mixed_dataset

If both input directories contain validation/, the script will also merge those
subdirectories into output-dir/validation/.

Mixing ratio and sample-count controls are intentionally defined as module-level
variables below rather than exposed as CLI flags, so they can be adjusted
directly in code for each experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# In-code experiment controls
# ---------------------------------------------------------------------------
#
# TRAIN_SOURCE_WEIGHTS controls the desired sampling ratio between sources.
# Example:
#   {"full_mod": 1.0, "canonical": 1.0} -> 1:1 balanced merge
#   {"full_mod": 2.0, "canonical": 1.0} -> 2:1 full-mod:canonical
TRAIN_SOURCE_WEIGHTS = {
    "full_mod": 1.0,
    "canonical": 1.0,
}

# If TRAIN_TARGET_TOTAL is None, the script uses the largest dataset size that
# still satisfies TRAIN_SOURCE_WEIGHTS under the source caps below.
# If set to an integer, the merged training split is downsampled to that total.
TRAIN_TARGET_TOTAL = None

# Optional hard caps applied before ratio planning.
# Set to None to keep all available samples from that source.
TRAIN_MAX_SAMPLES = {
    "full_mod": 150000,
    "canonical": 150000,
}

# Validation uses the same logic but can be controlled independently.
VALID_SOURCE_WEIGHTS = {
    "full_mod": 1.0,
    "canonical": 1.0,
}
VALID_TARGET_TOTAL = None
VALID_MAX_SAMPLES = {
    "full_mod": None,
    "canonical": None,
}

# Copy and indexing settings.
COPY_BLOCK_SIZE = 2048
WRITE_SHUFFLE_INDICES = True
MERGE_VALIDATION_IF_PRESENT = True


@dataclass
class DatasetArrays:
    directory: Path
    chunks: np.ndarray
    references: np.ndarray
    reference_lengths: np.ndarray
    mod_targets: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.reference_lengths.shape[0])

    @property
    def chunk_width(self) -> int:
        return int(self.chunks.shape[1])

    @property
    def reference_width(self) -> int:
        return int(self.references.shape[1])

    @property
    def mod_target_width(self) -> int:
        return int(self.mod_targets.shape[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-mod-dir", type=Path, required=True, help="Directory containing the full-mod dataset")
    parser.add_argument("--canonical-dir", type=Path, required=True, help="Directory containing the canonical dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the merged dataset into")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed for subset selection and shuffle indices")
    return parser.parse_args()


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_dataset(directory: Path) -> DatasetArrays:
    directory = Path(directory)
    chunks_path = directory / "chunks.npy"
    refs_path = directory / "references.npy"
    lens_path = directory / "reference_lengths.npy"
    mods_path = directory / "mod_targets.npy"

    require_file(chunks_path)
    require_file(refs_path)
    require_file(lens_path)
    require_file(mods_path)

    chunks = np.load(chunks_path, mmap_mode="r")
    references = np.load(refs_path, mmap_mode="r")
    reference_lengths = np.load(lens_path, mmap_mode="r")
    mod_targets = np.load(mods_path, mmap_mode="r")

    validate_dataset(directory, chunks, references, reference_lengths, mod_targets)
    return DatasetArrays(directory, chunks, references, reference_lengths, mod_targets)


def validate_dataset(
    directory: Path,
    chunks: np.ndarray,
    references: np.ndarray,
    reference_lengths: np.ndarray,
    mod_targets: np.ndarray,
):
    if chunks.ndim != 2:
        raise ValueError(f"{directory}: chunks.npy must be 2D, got shape {tuple(chunks.shape)}")
    if references.ndim != 2:
        raise ValueError(f"{directory}: references.npy must be 2D, got shape {tuple(references.shape)}")
    if reference_lengths.ndim != 1:
        raise ValueError(f"{directory}: reference_lengths.npy must be 1D, got shape {tuple(reference_lengths.shape)}")
    if mod_targets.ndim != 2:
        raise ValueError(f"{directory}: mod_targets.npy must be 2D, got shape {tuple(mod_targets.shape)}")

    num_samples = int(chunks.shape[0])
    if references.shape[0] != num_samples or reference_lengths.shape[0] != num_samples or mod_targets.shape[0] != num_samples:
        raise ValueError(
            f"{directory}: dataset arrays must share the same first dimension, got "
            f"chunks={chunks.shape[0]}, references={references.shape[0]}, "
            f"reference_lengths={reference_lengths.shape[0]}, mod_targets={mod_targets.shape[0]}"
        )

    if reference_lengths.size:
        max_len = int(reference_lengths.max())
        if max_len > references.shape[1]:
            raise ValueError(
                f"{directory}: max(reference_lengths.npy)={max_len} exceeds references.npy width={references.shape[1]}"
            )
        if max_len > mod_targets.shape[1]:
            raise ValueError(
                f"{directory}: max(reference_lengths.npy)={max_len} exceeds mod_targets.npy width={mod_targets.shape[1]}"
            )


def cap_count(available: int, cap: int | None) -> int:
    if cap is None:
        return int(available)
    return min(int(available), int(cap))


def plan_pair_counts(
    full_available: int,
    canonical_available: int,
    full_weight: float,
    canonical_weight: float,
    full_cap: int | None,
    canonical_cap: int | None,
    target_total: int | None,
) -> tuple[int, int]:
    full_limit = cap_count(full_available, full_cap)
    canonical_limit = cap_count(canonical_available, canonical_cap)

    if full_limit <= 0 and canonical_limit <= 0:
        return 0, 0

    if full_weight < 0 or canonical_weight < 0:
        raise ValueError("Source weights must be non-negative")

    if target_total is None:
        if full_weight == 0 and canonical_weight == 0:
            raise ValueError("At least one source weight must be positive")
        if full_weight == 0:
            return 0, canonical_limit
        if canonical_weight == 0:
            return full_limit, 0

        scale = min(full_limit / full_weight, canonical_limit / canonical_weight)
        full_count = int(math.floor(scale * full_weight))
        canonical_count = int(math.floor(scale * canonical_weight))
        return full_count, canonical_count

    target_total = int(target_total)
    if target_total < 0:
        raise ValueError("target_total must be >= 0")

    total_limit = full_limit + canonical_limit
    if target_total > total_limit:
        print(
            f"[warning] requested target_total={target_total} exceeds available capped samples={total_limit}; "
            f"using {total_limit} instead."
        )
        target_total = total_limit

    if target_total == 0:
        return 0, 0
    if full_weight == 0 and canonical_weight == 0:
        raise ValueError("At least one source weight must be positive")
    if full_weight == 0:
        return 0, min(canonical_limit, target_total)
    if canonical_weight == 0:
        return min(full_limit, target_total), 0

    weight_sum = full_weight + canonical_weight
    desired_full = target_total * full_weight / weight_sum
    desired_canonical = target_total * canonical_weight / weight_sum

    full_count = min(full_limit, int(round(desired_full)))
    canonical_count = min(canonical_limit, target_total - full_count)
    assigned = full_count + canonical_count
    remaining = target_total - assigned

    while remaining > 0:
        full_deficit = desired_full - full_count if full_count < full_limit else float("-inf")
        canonical_deficit = (
            desired_canonical - canonical_count if canonical_count < canonical_limit else float("-inf")
        )

        if full_deficit == float("-inf") and canonical_deficit == float("-inf"):
            break
        if full_deficit >= canonical_deficit and full_count < full_limit:
            full_count += 1
        elif canonical_count < canonical_limit:
            canonical_count += 1
        else:
            break
        remaining -= 1

    return full_count, canonical_count


def choose_subset_indices(num_available: int, num_selected: int, rng: np.random.Generator) -> np.ndarray:
    if num_selected < 0 or num_selected > num_available:
        raise ValueError(f"Invalid subset request: selected={num_selected}, available={num_available}")
    if num_selected == num_available:
        return np.arange(num_available, dtype=np.int64)
    indices = rng.choice(num_available, size=num_selected, replace=False)
    indices.sort()
    return indices.astype(np.int64, copy=False)


def write_1d_selected(src: np.ndarray, dst: np.memmap, src_indices: np.ndarray, dst_start: int, block_size: int):
    total = int(src_indices.shape[0])
    for offset in range(0, total, block_size):
        block_indices = src_indices[offset: offset + block_size]
        dst[dst_start + offset: dst_start + offset + len(block_indices)] = np.asarray(src[block_indices])
        del block_indices


def write_2d_selected(
    src: np.ndarray,
    dst: np.memmap,
    src_indices: np.ndarray,
    dst_start: int,
    dst_width: int,
    pad_value: int,
    block_size: int,
):
    src_width = int(src.shape[1])
    total = int(src_indices.shape[0])
    for offset in range(0, total, block_size):
        block_indices = src_indices[offset: offset + block_size]
        block = np.asarray(src[block_indices])
        if src_width == dst_width:
            dst_block = block
        else:
            dst_block = np.full((block.shape[0], dst_width), pad_value, dtype=src.dtype)
            dst_block[:, :src_width] = block
        dst[dst_start + offset: dst_start + offset + block.shape[0]] = dst_block
        del block_indices
        del block
        if src_width != dst_width:
            del dst_block


def build_shuffle_indices(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    index_dtype = np.int32 if num_samples <= np.iinfo(np.int32).max else np.int64
    indices = np.arange(num_samples, dtype=index_dtype)
    rng.shuffle(indices)
    return indices


def write_summary(path: Path, summary: dict):
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def merge_split(
    split_name: str,
    full_dir: Path,
    canonical_dir: Path,
    output_dir: Path,
    seed: int,
    source_weights: dict[str, float],
    source_caps: dict[str, int | None],
    target_total: int | None,
):
    print(f"[merge] split={split_name}")
    full = load_dataset(full_dir)
    canonical = load_dataset(canonical_dir)

    if full.chunk_width != canonical.chunk_width:
        raise ValueError(
            f"{split_name}: chunk width mismatch between sources: "
            f"full_mod={full.chunk_width}, canonical={canonical.chunk_width}"
        )

    full_count, canonical_count = plan_pair_counts(
        full_available=full.num_samples,
        canonical_available=canonical.num_samples,
        full_weight=float(source_weights["full_mod"]),
        canonical_weight=float(source_weights["canonical"]),
        full_cap=source_caps["full_mod"],
        canonical_cap=source_caps["canonical"],
        target_total=target_total,
    )

    total_count = full_count + canonical_count
    if total_count == 0:
        raise ValueError(f"{split_name}: merge plan selected zero samples")

    rng = np.random.default_rng(seed)
    full_indices = choose_subset_indices(full.num_samples, full_count, rng)
    canonical_indices = choose_subset_indices(canonical.num_samples, canonical_count, rng)

    reference_width = max(full.reference_width, canonical.reference_width)
    mod_target_width = max(full.mod_target_width, canonical.mod_target_width)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[merge] {split_name}: selected full_mod={full_count}/{full.num_samples}, "
        f"canonical={canonical_count}/{canonical.num_samples}, total={total_count}"
    )
    print(
        f"[merge] {split_name}: output shapes chunks=({total_count}, {full.chunk_width}), "
        f"references=({total_count}, {reference_width}), mod_targets=({total_count}, {mod_target_width})"
    )

    out_chunks = np.lib.format.open_memmap(
        output_dir / "chunks.npy",
        mode="w+",
        dtype=full.chunks.dtype,
        shape=(total_count, full.chunk_width),
    )
    out_refs = np.lib.format.open_memmap(
        output_dir / "references.npy",
        mode="w+",
        dtype=full.references.dtype,
        shape=(total_count, reference_width),
    )
    out_lens = np.lib.format.open_memmap(
        output_dir / "reference_lengths.npy",
        mode="w+",
        dtype=full.reference_lengths.dtype,
        shape=(total_count,),
    )
    out_mods = np.lib.format.open_memmap(
        output_dir / "mod_targets.npy",
        mode="w+",
        dtype=full.mod_targets.dtype,
        shape=(total_count, mod_target_width),
    )

    full_offset = 0
    canonical_offset = full_count

    write_2d_selected(full.chunks, out_chunks, full_indices, full_offset, full.chunk_width, 0, COPY_BLOCK_SIZE)
    write_2d_selected(
        full.references,
        out_refs,
        full_indices,
        full_offset,
        reference_width,
        0,
        COPY_BLOCK_SIZE,
    )
    write_1d_selected(full.reference_lengths, out_lens, full_indices, full_offset, COPY_BLOCK_SIZE)
    write_2d_selected(
        full.mod_targets,
        out_mods,
        full_indices,
        full_offset,
        mod_target_width,
        -100,
        COPY_BLOCK_SIZE,
    )

    write_2d_selected(
        canonical.chunks,
        out_chunks,
        canonical_indices,
        canonical_offset,
        canonical.chunk_width,
        0,
        COPY_BLOCK_SIZE,
    )
    write_2d_selected(
        canonical.references,
        out_refs,
        canonical_indices,
        canonical_offset,
        reference_width,
        0,
        COPY_BLOCK_SIZE,
    )
    write_1d_selected(canonical.reference_lengths, out_lens, canonical_indices, canonical_offset, COPY_BLOCK_SIZE)
    write_2d_selected(
        canonical.mod_targets,
        out_mods,
        canonical_indices,
        canonical_offset,
        mod_target_width,
        -100,
        COPY_BLOCK_SIZE,
    )

    out_chunks.flush()
    out_refs.flush()
    out_lens.flush()
    out_mods.flush()

    summary = {
        "split": split_name,
        "seed": int(seed),
        "full_mod_directory": str(full_dir.resolve()),
        "canonical_directory": str(canonical_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "source_weights": {
            "full_mod": float(source_weights["full_mod"]),
            "canonical": float(source_weights["canonical"]),
        },
        "source_caps": {
            "full_mod": source_caps["full_mod"],
            "canonical": source_caps["canonical"],
        },
        "target_total": target_total,
        "selected_counts": {
            "full_mod": int(full_count),
            "canonical": int(canonical_count),
            "total": int(total_count),
        },
        "output_shapes": {
            "chunks": [int(total_count), int(full.chunk_width)],
            "references": [int(total_count), int(reference_width)],
            "reference_lengths": [int(total_count)],
            "mod_targets": [int(total_count), int(mod_target_width)],
        },
    }
    write_summary(output_dir / "merge_summary.json", summary)

    if WRITE_SHUFFLE_INDICES:
        shuffle_rng = np.random.default_rng(seed + 10_000)
        indices = build_shuffle_indices(total_count, shuffle_rng)
        np.save(output_dir / "indices.npy", indices)
        print(f"[merge] {split_name}: wrote shuffled indices.npy")
        del indices

    del out_chunks
    del out_refs
    del out_lens
    del out_mods
    del full_indices
    del canonical_indices
    del full
    del canonical
    gc.collect()


def resolve_validation_dirs(full_dir: Path, canonical_dir: Path) -> tuple[Path, Path] | tuple[None, None]:
    full_valid = full_dir / "validation"
    canonical_valid = canonical_dir / "validation"
    full_exists = full_valid.exists()
    canonical_exists = canonical_valid.exists()

    if not full_exists and not canonical_exists:
        return None, None
    if full_exists != canonical_exists:
        raise FileNotFoundError(
            "validation/ must either exist in both input datasets or in neither. "
            f"full_mod={full_valid.exists()} canonical={canonical_valid.exists()}"
        )
    return full_valid, canonical_valid


def main():
    args = parse_args()

    full_dir = args.full_mod_dir.resolve()
    canonical_dir = args.canonical_dir.resolve()
    output_dir = args.output_dir.resolve()

    merge_split(
        split_name="train",
        full_dir=full_dir,
        canonical_dir=canonical_dir,
        output_dir=output_dir,
        seed=args.seed,
        source_weights=TRAIN_SOURCE_WEIGHTS,
        source_caps=TRAIN_MAX_SAMPLES,
        target_total=TRAIN_TARGET_TOTAL,
    )

    if MERGE_VALIDATION_IF_PRESENT:
        full_valid, canonical_valid = resolve_validation_dirs(full_dir, canonical_dir)
        if full_valid is not None and canonical_valid is not None:
            merge_split(
                split_name="validation",
                full_dir=full_valid,
                canonical_dir=canonical_valid,
                output_dir=output_dir / "validation",
                seed=args.seed + 1,
                source_weights=VALID_SOURCE_WEIGHTS,
                source_caps=VALID_MAX_SAMPLES,
                target_total=VALID_TARGET_TOTAL,
            )

    print(f"[done] merged dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
