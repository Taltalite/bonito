#!/usr/bin/env python3
"""
Generate Bonito-compatible chunk/reference datasets from Dorado basecaller BAM.

This script is designed to stay close to `bonito basecaller --save-ctc` while
using Dorado BAM tags to reconstruct the trimmed basecalled signal interval and
the move-table-derived base emission layout.

Key Dorado semantics followed here:
- `ns` defines the end of the basecalled signal interval, so the basecalled
  sequence and move table map to `signal[ts:ns]`.
- Split reads use `pi`/`sp` and the effective parent interval becomes
  `signal[sp + ts : sp + ns]`.
- Move table entries are stored in strided signal space and may use overflow
  encoding for values outside int8 range.

Reference:
- Dorado SAM specification
- Dorado move table documentation
- Dorado read splitting documentation
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mappy
import numpy as np
import pod5
import pysam
from tqdm import tqdm


BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
COMPLEMENT = str.maketrans("ACGTU", "TGCAA")
MAX_OPEN_POD5_PER_WORKER = 32
EPS = 1e-6
STRICT_MIN_ACCURACY = 0.99
STRICT_MIN_COVERAGE = 0.90
RELAXED_MIN_ACCURACY = 0.95
RELAXED_MIN_COVERAGE = 0.85
FLUSH_SAMPLE_COUNT = 5000
DEFAULT_TASK_BATCH_SIZE = 64
DEFAULT_MAX_PENDING_BATCHES = 4


def typical_indices(x: np.ndarray, n: float = 2.5) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.int64)
    mu = np.mean(x)
    sd = np.std(x)
    idx, = np.where((mu - n * sd < x) & (x < mu + n * sd))
    return idx


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def query_pos_to_signal_idx(q_pos: int, read_length: int, is_reverse: bool, sample_type: str) -> int:
    """
    Map BAM query indices onto physical signal-order base indices.

    This matches the repository's prior RNA/DNA handling and assumes the BAM
    record follows SAM reverse-complement rules when FLAG 0x10 is set.
    """
    if sample_type == "rna":
        return q_pos if is_reverse else read_length - 1 - q_pos
    return read_length - 1 - q_pos if is_reverse else q_pos


def chunk_windows(signal_len: int, chunk_len: int, overlap: int) -> Iterable[Tuple[int, int]]:
    if signal_len < chunk_len:
        return
    stride = chunk_len - overlap
    _, offset = divmod(signal_len - chunk_len, stride)
    for start in range(offset, signal_len - chunk_len + 1, stride):
        yield start, start + chunk_len


def encode_reference(seq: str) -> np.ndarray:
    encoded = np.zeros((len(seq),), dtype=np.uint8)
    for idx, base in enumerate(seq):
        encoded[idx] = BASE_TO_INT.get(base.upper(), 0)
    return encoded


def compute_quantile_norm(signal: np.ndarray) -> Tuple[float, float]:
    qa, qb = np.quantile(signal, [0.2, 0.9])
    shift = max(10.0, 0.51 * float(qa + qb))
    scale = max(1.0, 0.53 * float(qb - qa))
    return shift, scale


def decode_overflow_encoded_int8(values: Sequence[int]) -> List[int]:
    decoded: List[int] = []
    idx = 0
    total_values = len(values)
    while idx < total_values:
        total = 0
        while True:
            if idx >= total_values:
                raise ValueError("Malformed overflow-encoded int8 array.")
            value = int(values[idx])
            total += value
            idx += 1
            if value not in (-128, 127):
                break
        decoded.append(total)
    return decoded


def decode_move_table(mv_tag: Sequence[int]) -> Tuple[int, List[int]]:
    if mv_tag is None or len(mv_tag) == 0:
        raise ValueError("Missing mv tag.")
    decoded = decode_overflow_encoded_int8(mv_tag)
    if not decoded:
        raise ValueError("Decoded move table is empty.")
    stride = int(decoded[0])
    if stride <= 0:
        raise ValueError(f"Invalid move stride: {stride}")
    moves = [int(value) for value in decoded[1:]]
    if any(value < 0 for value in moves):
        raise ValueError("Move table contains negative move counts after decoding.")
    return stride, moves


def build_signal_order_sequence(query_sequence: str, sample_type: str, is_reverse: bool) -> str:
    if not query_sequence:
        return ""
    read_length = len(query_sequence)
    ordered = [""] * read_length
    for q_pos, base in enumerate(query_sequence.upper()):
        signal_idx = query_pos_to_signal_idx(q_pos, read_length, is_reverse, sample_type)
        if signal_idx < 0 or signal_idx >= read_length:
            raise ValueError("Signal-order index out of range.")
        ordered[signal_idx] = "T" if base == "U" else base
    if any(not base for base in ordered):
        raise ValueError("Failed to reconstruct signal-order sequence.")
    return "".join(ordered)


@dataclass
class TaskData:
    record_id: str
    pod5_read_id: str
    query_sequence: str
    sample_type: str
    is_reverse: bool
    ts: int
    ns: int
    sp: int
    mv_tag: Sequence[int]
    mean_qscore: Optional[float]
    scaling_shift: Optional[float]
    scaling_scale: Optional[float]
    chunk_len: int
    overlap: int
    min_accuracy: float
    min_coverage: float
    min_qscore: Optional[float]
    clip_value: float
    max_label_len: Optional[int]
    norm_strategy: str
    pa_mean: float
    pa_std: float


@dataclass
class Sample:
    signal: np.ndarray
    label: np.ndarray
    label_len: int


class WorkerState:
    pod5_lookup: Dict[str, Tuple[str, int, int]] = {}
    pod5_reader_cache: OrderedDict[str, pod5.Reader] = OrderedDict()
    aligner: Optional[mappy.Aligner] = None
    temp_dir: Optional[str] = None


PARENT_POD5_LOOKUP: Dict[str, Tuple[str, int, int]] = {}


def worker_init(reference_fasta: str, pod5_lookup: Optional[Dict[str, Tuple[str, int, int]]], mm2_preset: str, temp_dir: str):
    os.environ["TMPDIR"] = "/tmp"
    WorkerState.pod5_lookup = pod5_lookup if pod5_lookup is not None else PARENT_POD5_LOOKUP
    WorkerState.pod5_reader_cache = OrderedDict()
    WorkerState.aligner = mappy.Aligner(reference_fasta, preset=mm2_preset)
    WorkerState.temp_dir = temp_dir
    if WorkerState.aligner is None:
        raise RuntimeError(f"Failed to build/load minimap2 index for {reference_fasta}")


def get_pod5_reader(path: str) -> pod5.Reader:
    cache = WorkerState.pod5_reader_cache
    if path in cache:
        cache.move_to_end(path)
        return cache[path]
    if len(cache) >= MAX_OPEN_POD5_PER_WORKER:
        _, reader_to_close = cache.popitem(last=False)
        reader_to_close.close()
    reader = pod5.Reader(path)
    cache[path] = reader
    return reader


def collect_pod5_paths(pod5_dir: Path) -> List[str]:
    if not pod5_dir.is_dir():
        raise FileNotFoundError(f"POD5 directory does not exist: {pod5_dir}")
    pod5_paths = []
    fast5_paths = []
    for root, _, files in os.walk(pod5_dir):
        for name in files:
            lower = name.lower()
            full_path = os.path.join(root, name)
            if lower.endswith(".pod5"):
                pod5_paths.append(full_path)
            elif lower.endswith(".fast5"):
                fast5_paths.append(full_path)
    if fast5_paths and pod5_paths:
        raise ValueError(f"Mixed .pod5 and .fast5 inputs found under {pod5_dir}")
    if fast5_paths and not pod5_paths:
        raise ValueError(f"Only .fast5 inputs found under {pod5_dir}; this workflow expects POD5.")
    if not pod5_paths:
        raise FileNotFoundError(f"No .pod5 files found under {pod5_dir}")
    return sorted(pod5_paths)


def build_pod5_lookup(pod5_dir: Path) -> Dict[str, Tuple[str, int, int]]:
    lookup: Dict[str, Tuple[str, int, int]] = {}
    for pod5_path in tqdm(collect_pod5_paths(pod5_dir), desc="Indexing POD5", ascii=True, ncols=100):
        with pod5.Reader(pod5_path) as reader:
            for batch_idx in range(reader.batch_count):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_reads):
                    read = batch.get_read(row_idx)
                    lookup[str(read.read_id)] = (pod5_path, batch_idx, row_idx)
    return lookup


def fetch_calibrated_signal(read_id: str) -> np.ndarray:
    if read_id not in WorkerState.pod5_lookup:
        raise KeyError(f"Read {read_id} not found in POD5 lookup.")
    pod5_path, batch_idx, row_idx = WorkerState.pod5_lookup[read_id]
    reader = get_pod5_reader(pod5_path)
    batch = reader.get_batch(batch_idx)
    pod5_read = batch.get_read(row_idx)
    raw = pod5_read.signal
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"Invalid raw signal for read {read_id}")
    calibration = pod5_read.calibration
    return (raw.astype(np.float32) + calibration.offset) * calibration.scale


def normalise_interval_signal(interval_signal: np.ndarray, task: TaskData) -> np.ndarray:
    if task.norm_strategy == "from-bam":
        if task.scaling_shift is None or task.scaling_scale is None:
            raise ValueError("BAM scaling tags sm/sd are required for norm-strategy=from-bam")
        shift = float(task.scaling_shift)
        scale = float(task.scaling_scale)
    elif task.norm_strategy == "pa":
        shift = float(task.pa_mean)
        scale = float(task.pa_std)
    else:
        shift, scale = compute_quantile_norm(interval_signal)

    if abs(scale) < EPS:
        raise ValueError("Signal scale is too small.")

    normalised = (interval_signal - shift) / scale
    return np.clip(normalised, -task.clip_value, task.clip_value).astype(np.float16)


def build_task(read: pysam.AlignedSegment, args) -> Optional[TaskData]:
    if read.is_secondary or read.is_supplementary:
        return None
    if read.has_tag("dx") and int(read.get_tag("dx")) == 1:
        return None
    if not read.has_tag("mv"):
        return None
    if not read.has_tag("ns"):
        return None
    query_sequence = (read.query_sequence or "").upper().replace("U", "T")
    if not query_sequence:
        return None

    split_parent = str(read.get_tag("pi")) if read.has_tag("pi") else None
    split_start = int(read.get_tag("sp")) if read.has_tag("sp") else 0
    pod5_read_id = split_parent if split_parent else read.query_name

    return TaskData(
        record_id=str(read.query_name),
        pod5_read_id=str(pod5_read_id),
        query_sequence=query_sequence,
        sample_type=args.sample_type,
        is_reverse=bool(read.is_reverse),
        ts=int(read.get_tag("ts")) if read.has_tag("ts") else 0,
        ns=int(read.get_tag("ns")),
        sp=split_start,
        mv_tag=tuple(int(value) for value in read.get_tag("mv")),
        mean_qscore=float(read.get_tag("qs")) if read.has_tag("qs") else None,
        scaling_shift=float(read.get_tag("sm")) if read.has_tag("sm") else None,
        scaling_scale=float(read.get_tag("sd")) if read.has_tag("sd") else None,
        chunk_len=int(args.chunk_len),
        overlap=int(args.overlap),
        min_accuracy=float(args.min_accuracy),
        min_coverage=float(args.min_coverage),
        min_qscore=float(args.min_qscore) if args.min_qscore is not None else None,
        clip_value=float(args.clip_value),
        max_label_len=int(args.max_label_len) if args.max_label_len is not None else None,
        norm_strategy=str(args.norm_strategy),
        pa_mean=float(args.pa_mean),
        pa_std=float(args.pa_std),
    )


def process_task(task: TaskData) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {
        "records_seen": 1,
        "records_missing_signal": 0,
        "records_invalid_interval": 0,
        "records_invalid_move_table": 0,
        "records_move_seq_mismatch": 0,
        "records_signal_too_short": 0,
        "records_low_qscore": 0,
        "chunks_seen": 0,
        "chunks_written": 0,
        "chunks_zerolen_sequence": 0,
        "chunks_no_mapping": 0,
        "chunks_low_accuracy": 0,
        "chunks_low_coverage": 0,
        "chunks_n_in_reference": 0,
        "chunks_too_long": 0,
    }
    samples: List[Sample] = []

    try:
        raw_signal = fetch_calibrated_signal(task.pod5_read_id)
    except Exception:
        stats["records_missing_signal"] += 1
        return samples, stats

    interval_start = task.sp + task.ts
    interval_end = task.sp + task.ns
    if interval_start < 0 or interval_end <= interval_start or interval_end > raw_signal.shape[0]:
        stats["records_invalid_interval"] += 1
        return samples, stats

    interval_signal = raw_signal[interval_start:interval_end]
    if interval_signal.shape[0] < task.chunk_len:
        stats["records_signal_too_short"] += 1
        return samples, stats

    if task.min_qscore is not None and task.mean_qscore is not None and task.mean_qscore < task.min_qscore:
        stats["records_low_qscore"] += 1
        return samples, stats

    try:
        stride, moves = decode_move_table(task.mv_tag)
        signal_order_sequence = build_signal_order_sequence(task.query_sequence, task.sample_type, task.is_reverse)
    except Exception:
        stats["records_invalid_move_table"] += 1
        return samples, stats

    emitted_positions: List[int] = []
    for step_idx, move_count in enumerate(moves):
        position = (step_idx * stride) + (stride // 2)
        for _ in range(move_count):
            emitted_positions.append(position)

    if len(emitted_positions) != len(signal_order_sequence):
        stats["records_move_seq_mismatch"] += 1
        return samples, stats

    if not emitted_positions:
        stats["records_invalid_move_table"] += 1
        return samples, stats

    emitted_positions_arr = np.asarray(emitted_positions, dtype=np.int64)

    try:
        normalised_signal = normalise_interval_signal(interval_signal, task)
    except Exception:
        stats["records_invalid_interval"] += 1
        return samples, stats

    aligner = WorkerState.aligner
    if aligner is None:
        raise RuntimeError("Worker aligner is not initialised.")

    for win_start, win_end in chunk_windows(len(normalised_signal), task.chunk_len, task.overlap):
        stats["chunks_seen"] += 1
        left_idx = int(np.searchsorted(emitted_positions_arr, win_start, side="left"))
        right_idx = int(np.searchsorted(emitted_positions_arr, win_end, side="left"))
        if left_idx >= right_idx:
            stats["chunks_zerolen_sequence"] += 1
            continue

        signal_order_chunk_seq = signal_order_sequence[left_idx:right_idx]
        if not signal_order_chunk_seq:
            stats["chunks_zerolen_sequence"] += 1
            continue

        query_seq = signal_order_chunk_seq[::-1] if task.sample_type == "rna" else signal_order_chunk_seq
        mapping = next(aligner.map(query_seq, MD=True), None)
        if mapping is None:
            stats["chunks_no_mapping"] += 1
            continue

        coverage = (mapping.q_en - mapping.q_st) / max(len(query_seq), 1)
        accuracy = mapping.mlen / max(mapping.blen, 1)
        if accuracy < task.min_accuracy:
            stats["chunks_low_accuracy"] += 1
            continue
        if coverage < task.min_coverage:
            stats["chunks_low_coverage"] += 1
            continue

        ref_seq = aligner.seq(mapping.ctg, mapping.r_st, mapping.r_en)
        if ref_seq is None:
            stats["chunks_no_mapping"] += 1
            continue
        if mapping.strand == -1:
            ref_seq = mappy.revcomp(ref_seq)
        if "N" in ref_seq.upper():
            stats["chunks_n_in_reference"] += 1
            continue

        target_seq = ref_seq[::-1] if task.sample_type == "rna" else ref_seq
        target = encode_reference(target_seq)
        if np.any(target == 0):
            stats["chunks_n_in_reference"] += 1
            continue
        if task.max_label_len is not None and target.shape[0] > task.max_label_len:
            stats["chunks_too_long"] += 1
            continue

        samples.append(
            Sample(
                signal=normalised_signal[win_start:win_end],
                label=target.astype(np.uint8),
                label_len=int(target.shape[0]),
            )
        )
        stats["chunks_written"] += 1

    return samples, stats


def merge_stats(total: Dict[str, int], update: Dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def write_worker_samples(samples: List[Sample]) -> Optional[Dict[str, object]]:
    if not samples:
        return None
    temp_dir = WorkerState.temp_dir
    if temp_dir is None:
        raise RuntimeError("Worker temp_dir is not initialised.")
    signals = np.stack([sample.signal for sample in samples], axis=0)
    lengths = np.asarray([sample.label_len for sample in samples], dtype=np.uint16)
    offsets = np.zeros((len(samples) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    labels_flat = np.concatenate([sample.label for sample in samples], axis=0)

    with tempfile.NamedTemporaryFile(prefix="temp_sig_", suffix=".npy", dir=temp_dir, delete=False) as sig_tmp:
        sig_path = sig_tmp.name
    with tempfile.NamedTemporaryFile(prefix="temp_lbl_", suffix=".npy", dir=temp_dir, delete=False) as lbl_tmp:
        lbl_path = lbl_tmp.name
    with tempfile.NamedTemporaryFile(prefix="temp_off_", suffix=".npy", dir=temp_dir, delete=False) as off_tmp:
        off_path = off_tmp.name
    with tempfile.NamedTemporaryFile(prefix="temp_len_", suffix=".npy", dir=temp_dir, delete=False) as len_tmp:
        len_path = len_tmp.name

    np.save(sig_path, signals)
    np.save(lbl_path, labels_flat)
    np.save(off_path, offsets)
    np.save(len_path, lengths)
    return {
        "signals": sig_path,
        "labels": lbl_path,
        "offsets": off_path,
        "lengths": len_path,
        "num_samples": int(signals.shape[0]),
    }


def process_task_batch(tasks: Sequence[TaskData]) -> Tuple[Optional[Dict[str, object]], Dict[str, int]]:
    batch_stats: Dict[str, int] = {}
    batch_samples: List[Sample] = []
    for task in tasks:
        samples, stats = process_task(task)
        merge_stats(batch_stats, stats)
        batch_samples.extend(samples)
    manifest_entry = write_worker_samples(batch_samples)
    return manifest_entry, batch_stats


def flush_temp_chunk(chunk_id: int, chunk: List[Sample], temp_dir: Path, manifest: List[Dict[str, object]]) -> int:
    if not chunk:
        return chunk_id
    signals = np.stack([sample.signal for sample in chunk], axis=0)
    lengths = np.asarray([sample.label_len for sample in chunk], dtype=np.uint16)
    offsets = np.zeros((len(chunk) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    labels_flat = np.concatenate([sample.label for sample in chunk], axis=0)
    sig_path = temp_dir / f"temp_sig_{chunk_id}.npy"
    lbl_path = temp_dir / f"temp_lbl_{chunk_id}.npy"
    off_path = temp_dir / f"temp_off_{chunk_id}.npy"
    len_path = temp_dir / f"temp_len_{chunk_id}.npy"
    np.save(sig_path, signals)
    np.save(lbl_path, labels_flat)
    np.save(off_path, offsets)
    np.save(len_path, lengths)
    manifest.append(
        {
            "signals": str(sig_path),
            "labels": str(lbl_path),
            "offsets": str(off_path),
            "lengths": str(len_path),
            "num_samples": int(signals.shape[0]),
        }
    )
    return chunk_id + 1


def build_chunk_ranges(chunk_manifest: List[Dict[str, object]]) -> List[int]:
    starts = [0]
    total = 0
    for chunk_info in chunk_manifest:
        total += int(chunk_info["num_samples"])
        starts.append(total)
    return starts


def find_chunk_index(starts: Sequence[int], index: int) -> int:
    lo = 0
    hi = len(starts) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if starts[mid] <= index:
            lo = mid
        else:
            hi = mid
    return lo


def merge_chunks_to_final(output_dir: Path, chunk_manifest: List[Dict[str, object]], signal_len: int, max_label_len: Optional[int]) -> Dict[str, int]:
    if not chunk_manifest:
        raise RuntimeError("No valid samples remained after filtering.")

    lengths_list = []
    for chunk_info in tqdm(chunk_manifest, desc="Loading lengths", ascii=True, ncols=100):
        lengths_list.append(np.load(chunk_info["lengths"]))
    lengths = np.concatenate(lengths_list, axis=0)
    keep_indices = typical_indices(lengths)
    if keep_indices.size == 0:
        raise RuntimeError("No samples remained after typical-length filtering.")
    if max_label_len is None:
        max_label_len = int(lengths[keep_indices].max())

    keep_indices = np.random.permutation(keep_indices)
    total_samples = int(keep_indices.size)

    chunks_out = np.lib.format.open_memmap(output_dir / "chunks.npy", mode="w+", dtype=np.float16, shape=(total_samples, signal_len))
    refs_out = np.lib.format.open_memmap(output_dir / "references.npy", mode="w+", dtype=np.uint8, shape=(total_samples, max_label_len))
    lens_out = np.lib.format.open_memmap(output_dir / "reference_lengths.npy", mode="w+", dtype=np.uint16, shape=(total_samples,))

    starts = build_chunk_ranges(chunk_manifest)
    chunk_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def load_chunk_data(idx: int):
        if idx in chunk_cache:
            return chunk_cache[idx]
        info = chunk_manifest[idx]
        chunk_cache[idx] = (
            np.load(info["signals"], mmap_mode="r"),
            np.load(info["labels"], mmap_mode="r"),
            np.load(info["offsets"], mmap_mode="r"),
            np.load(info["lengths"], mmap_mode="r"),
        )
        return chunk_cache[idx]

    block_size = 2048
    for out_start in tqdm(range(0, total_samples, block_size), desc="Writing dataset", ascii=True, ncols=100):
        out_end = min(out_start + block_size, total_samples)
        block_indices = keep_indices[out_start:out_end]
        block_signals = np.empty((out_end - out_start, signal_len), dtype=np.float16)
        block_refs = np.zeros((out_end - out_start, max_label_len), dtype=np.uint8)
        block_lengths = lengths[block_indices].astype(np.uint16)
        for pos, global_index in enumerate(block_indices):
            chunk_idx = find_chunk_index(starts, int(global_index))
            local_index = int(global_index - starts[chunk_idx])
            signals, labels, offsets, lengths_chunk = load_chunk_data(chunk_idx)
            block_signals[pos] = signals[local_index]
            label_len = int(lengths_chunk[local_index])
            label_start = int(offsets[local_index])
            block_refs[pos, :label_len] = labels[label_start:label_start + label_len]
        chunks_out[out_start:out_end] = block_signals
        refs_out[out_start:out_end] = block_refs
        lens_out[out_start:out_end] = block_lengths

    del chunks_out, refs_out, lens_out

    for chunk_info in chunk_manifest:
        os.remove(chunk_info["signals"])
        os.remove(chunk_info["labels"])
        os.remove(chunk_info["offsets"])
        os.remove(chunk_info["lengths"])

    return {
        "total_pre_typical_filter": int(lengths.shape[0]),
        "total_written": total_samples,
        "max_label_len": int(max_label_len),
    }


def write_summary(output_dir: Path, args, counters: Dict[str, int], merge_stats: Dict[str, int]):
    summary = {
        "bam_file": str(Path(args.bam_file).resolve()),
        "pod5_dir": str(Path(args.pod5_dir).resolve()),
        "reference_fasta": str(Path(args.reference_fasta).resolve()),
        "output_dir": str(output_dir.resolve()),
        "sample_type": args.sample_type,
        "chunk_len": int(args.chunk_len),
        "overlap": int(args.overlap),
        "min_accuracy": float(args.min_accuracy),
        "min_coverage": float(args.min_coverage),
        "min_qscore": None if args.min_qscore is None else float(args.min_qscore),
        "norm_strategy": args.norm_strategy,
        "clip_value": float(args.clip_value),
        "counters": {key: int(value) for key, value in sorted(counters.items())},
        "merge": merge_stats,
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def resolve_thresholds(args):
    if args.filter_preset == "strict":
        default_accuracy = STRICT_MIN_ACCURACY
        default_coverage = STRICT_MIN_COVERAGE
    else:
        default_accuracy = RELAXED_MIN_ACCURACY
        default_coverage = RELAXED_MIN_COVERAGE
    if args.min_accuracy is None:
        args.min_accuracy = default_accuracy
    if args.min_coverage is None:
        args.min_coverage = default_coverage


def resolve_mp_start_method(requested: str) -> str:
    if requested != "auto":
        return requested
    return "fork" if os.name == "posix" else "spawn"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-file", required=True, help="Dorado basecaller BAM produced with --reference --emit-moves")
    parser.add_argument("--pod5-dir", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-type", choices=["dna", "rna"], default="rna")
    parser.add_argument("--chunk-len", type=int, default=12000)
    parser.add_argument("--overlap", type=int, default=600)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--filter-preset", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--min-qscore", type=float, default=None)
    parser.add_argument("--norm-strategy", choices=["from-bam", "pa", "quantile"], default="from-bam")
    parser.add_argument("--pa-mean", type=float, default=0.0)
    parser.add_argument("--pa-std", type=float, default=1.0)
    parser.add_argument("--mm2-preset", default="lr:hq", help="Minimap2 preset used for chunk-local realignment.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=DEFAULT_TASK_BATCH_SIZE)
    parser.add_argument("--max-pending-batches", type=int, default=DEFAULT_MAX_PENDING_BATCHES)
    parser.add_argument("--mp-start-method", choices=["auto", "fork", "spawn", "forkserver"], default="auto")
    return parser.parse_args()


def main():
    os.environ["TMPDIR"] = "/tmp"
    args = parse_args()
    resolve_thresholds(args)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Building POD5 index...")
    lookup = build_pod5_lookup(Path(args.pod5_dir))
    print(f"      Found {len(lookup)} reads in POD5.")
    start_method = resolve_mp_start_method(args.mp_start_method)
    print(f"      multiprocessing start method: {start_method}")

    print("[2/5] Dispatching BAM records...")
    chunk_manifest: List[Dict[str, object]] = []
    counters: Dict[str, int] = {}
    task_count = 0
    task_batch: List[TaskData] = []
    max_pending_batches = max(int(args.max_pending_batches), 1) * max(int(args.workers), 1)
    mp_context = multiprocessing.get_context(start_method)

    global PARENT_POD5_LOOKUP
    PARENT_POD5_LOOKUP = lookup
    init_lookup = None if start_method == "fork" else lookup

    with pysam.AlignmentFile(args.bam_file, "rb", check_sq=False) as bam_file:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_context,
            initializer=worker_init,
            initargs=(str(Path(args.reference_fasta).resolve()), init_lookup, args.mm2_preset, str(temp_dir.resolve())),
        ) as executor:
            futures = set()
            for read in tqdm(bam_file, desc="Dispatching", unit="record", ascii=True, ncols=100):
                if args.max_records > 0 and task_count >= args.max_records:
                    break
                task = build_task(read, args)
                if task is None:
                    continue
                task_batch.append(task)
                task_count += 1

                if len(task_batch) >= max(int(args.task_batch_size), 1):
                    futures.add(executor.submit(process_task_batch, tuple(task_batch)))
                    task_batch = []

                if len(futures) >= max_pending_batches:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    futures.difference_update(done)
                    for future in done:
                        manifest_entry, stats = future.result()
                        merge_stats(counters, stats)
                        if manifest_entry is not None:
                            chunk_manifest.append(manifest_entry)

            if task_batch:
                futures.add(executor.submit(process_task_batch, tuple(task_batch)))
                task_batch = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing", ascii=True, ncols=100):
                manifest_entry, stats = future.result()
                merge_stats(counters, stats)
                if manifest_entry is not None:
                    chunk_manifest.append(manifest_entry)

    PARENT_POD5_LOOKUP = {}

    print("[3/5] Merging passing chunks into final dataset...")
    merge_stats = merge_chunks_to_final(output_dir, chunk_manifest, int(args.chunk_len), args.max_label_len)

    print("[4/5] Writing summary...")
    write_summary(output_dir, args, counters, merge_stats)

    print("[5/5] Cleaning up temp chunks...")
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {merge_stats['total_written']} chunks written.")
    print("Reject counters:")
    for key in sorted(counters):
        print(f" - {key}: {counters[key]}")


if __name__ == "__main__":
    main()
