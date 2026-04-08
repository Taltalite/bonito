#!/usr/bin/env python
"""
Bonito Dataset Generator V8 (Correct Signal-Order Alignment)

Supports both DNA/cDNA and direct RNA training-data generation from
Dorado-style BAM + POD5 inputs.

Key rules:
1. Labels are always written in physical signal order.
2. DNA/cDNA and direct RNA use different BAM-query -> signal index mappings.
3. Only POD5 raw input is supported in this repository workflow.
"""
from __future__ import annotations

import argparse
import bisect
import json
import multiprocessing
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pod5
import pysam
from tqdm import tqdm
import shutil

# 1-based encoding: 0=Pad, 1=A, 2=C, 3=G, 4=T/U
BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
COMPLEMENT = str.maketrans("ACGTU", "TGCAA")
MAX_OPEN_POD5_PER_WORKER = 32
EPS = 1e-5

def complement_base(base: str) -> str:
    return base.upper().translate(COMPLEMENT)


def query_pos_to_signal_idx(q_pos: int, read_length: int, is_reverse: bool, sample_type: str) -> int:
    """
    Map a BAM query index to the physical signal-order base index.

    For DNA/cDNA, reverse-strand BAM records are reverse-complemented relative
    to the pore order, so only reverse-strand reads need index reversal.

    For direct RNA, Dorado/BAM query order is already reversed relative to the
    pore order on forward-strand alignments. On reverse-strand alignments, the
    SAM reverse-complement transform cancels that global reversal, so those
    reads stay in signal order.
    """
    if sample_type == "rna":
        return q_pos if is_reverse else read_length - 1 - q_pos
    return read_length - 1 - q_pos if is_reverse else q_pos

def normalize_ref_name(name: str) -> str:
    name = name.lower()
    if name.startswith("chr"): name = name[3:]
    return name

def build_reference_mapping(bam_header, fasta):
    fasta_names = list(fasta.references)
    fasta_lengths = dict(zip(fasta.references, fasta.lengths))
    normalized = {}
    for name in fasta_names:
        key = normalize_ref_name(name)
        if key not in normalized: normalized[key] = name
        else: normalized[key] = None
    mapping = {}
    for bam_name in bam_header.references:
        if bam_name in fasta_names:
            mapping[bam_name] = bam_name
            continue
        norm_key = normalize_ref_name(bam_name)
        if norm_key in normalized and normalized[norm_key] is not None:
            mapping[bam_name] = normalized[norm_key]
            continue
        bam_len = bam_header.get_reference_length(bam_name)
        length_matches = [name for name, length in fasta_lengths.items() if length == bam_len]
        if len(length_matches) == 1:
            mapping[bam_name] = length_matches[0]
    return mapping

def typical_indices(x: np.ndarray, n: float = 2.5) -> np.ndarray:
    if x.size == 0: return np.array([], dtype=np.int64)
    mu, sd = np.mean(x), np.std(x)
    idx, = np.where((mu - n*sd < x) & (x < mu + n*sd))
    return idx

def compute_norm_params(signal, strategy, pa_mean, pa_std):
    if strategy == "pa": return pa_mean, pa_std
    else:
        qa, qb = np.quantile(signal, [0.2, 0.9])
        shift = 0.51 * (qa + qb)
        scale = 0.53 * (qb - qa)
        if scale < EPS: scale = 1.0
        return shift, scale

@dataclass
class TaskData:
    read_id: str
    read_length: int
    reference_name: str
    is_reverse: bool
    sample_type: str
    ts_tag: int
    mv_tag: Sequence[int]
    nm_tag: int
    signal_len_arg: int
    stride_arg: int
    overlap_arg: int
    max_label_len: Optional[int]
    clip_value: float
    norm_strategy: str
    pa_mean: float
    pa_std: float
    aligned_pairs: List[Tuple[int, int]]

@dataclass
class Sample:
    signal: np.ndarray
    label: np.ndarray
    label_len: int

class WorkerState:
    fasta: Optional[pysam.FastaFile] = None
    pod5_lookup: Dict = {}
    pod5_reader_cache: OrderedDict = OrderedDict()
    reference_mapping: Dict = {}

def worker_init(fasta_path, lookup, reference_mapping):
    os.environ["TMPDIR"] = "/tmp"
    WorkerState.fasta = pysam.FastaFile(fasta_path)
    WorkerState.pod5_lookup = lookup
    WorkerState.reference_mapping = reference_mapping
    WorkerState.pod5_reader_cache = OrderedDict()

def get_pod5_reader(path):
    cache = WorkerState.pod5_reader_cache
    if path in cache:
        cache.move_to_end(path)
        return cache[path]
    if len(cache) >= MAX_OPEN_POD5_PER_WORKER:
        _, reader_to_close = cache.popitem(last=False)
        reader_to_close.close()
    try: new_reader = pod5.Reader(path)
    except:
        time.sleep(0.1)
        new_reader = pod5.Reader(path)
    cache[path] = new_reader
    return new_reader


def collect_pod5_paths(pod5_dir):
    if not os.path.isdir(pod5_dir):
        raise FileNotFoundError(f"POD5 directory does not exist: {pod5_dir}")

    pod5_paths = []
    fast5_paths = []
    for root, _, files in os.walk(pod5_dir):
        for name in files:
            lower_name = name.lower()
            full_path = os.path.join(root, name)
            if lower_name.endswith(".pod5"):
                pod5_paths.append(full_path)
            elif lower_name.endswith(".fast5"):
                fast5_paths.append(full_path)

    if fast5_paths and pod5_paths:
        raise ValueError(
            f"Mixed raw formats found under {pod5_dir}. "
            "This script only supports a pure .pod5 input set."
        )
    if fast5_paths and not pod5_paths:
        raise ValueError(
            f"Only .fast5 inputs were found under {pod5_dir}. "
            "This script expects .pod5 input for the current Bonito workflow."
        )
    if not pod5_paths:
        raise FileNotFoundError(f"No .pod5 files found under {pod5_dir}")

    return sorted(pod5_paths)

def fetch_calibrated_signal(read_id):
    if read_id not in WorkerState.pod5_lookup: raise KeyError("Read ID missing")
    pod5_path, batch_idx, row_idx = WorkerState.pod5_lookup[read_id]
    reader = get_pod5_reader(pod5_path)
    batch = reader.get_batch(batch_idx)
    pod5_read = batch.get_read(row_idx)
    cal = pod5_read.calibration
    raw = pod5_read.signal
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"Invalid raw signal for read {read_id}")
    pa_signal = (raw.astype(np.float32) + cal.offset) * cal.scale
    if pa_signal.ndim != 1 or pa_signal.size == 0:
        raise ValueError(f"Empty calibrated signal for read {read_id}")
    return pa_signal, pod5_read


def process_task(task: TaskData) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {
        "records_seen": 1,
        "records_missing_signal": 0,
        "records_signal_too_short": 0,
        "records_missing_reference": 0,
        "records_invalid_move_table": 0,
        "records_low_accuracy_prefilter": 0,
        "chunks_seen": 0,
        "chunks_written": 0,
        "chunks_zerolen_sequence": 0,
        "chunks_softclip_or_insertion": 0,
        "chunks_low_accuracy": 0,
        "chunks_too_short": 0,
        "chunks_too_long": 0,
    }
    samples: List[Sample] = []
    
    if task.read_length <= 0: return samples, stats

    # 1. Fetch Signal
    try: pa_signal, _ = fetch_calibrated_signal(task.read_id)
    except:
        stats["records_missing_signal"] += 1
        return samples, stats
    if pa_signal.shape[0] < task.signal_len_arg:
        stats["records_signal_too_short"] += 1
        return samples, stats

    # 2. Norm
    global_shift, global_scale = compute_norm_params(pa_signal, task.norm_strategy, task.pa_mean, task.pa_std)

    # 3. Build Labels (THE V8 FIX)
    # -----------------------------------------------------------------------
    # We need an array that maps: Signal_Index (0..N) -> Reference Base
    # The 'full_labels' array must represent the PHYSICAL order of bases as seen by the pore.
    full_labels = np.zeros(task.read_length, dtype=np.uint8)
    
    reference_name = WorkerState.reference_mapping.get(task.reference_name)
    if not reference_name:
        stats["records_missing_reference"] += 1
        return samples, stats
    fasta = WorkerState.fasta
    
    for q_pos, r_pos in task.aligned_pairs:
        if q_pos is None or r_pos is None: continue
        
        try:
            # Map BAM query order back into physical signal order. This differs
            # between DNA/cDNA and direct RNA, so keep it explicit.
            signal_idx = query_pos_to_signal_idx(
                q_pos=q_pos,
                read_length=task.read_length,
                is_reverse=task.is_reverse,
                sample_type=task.sample_type,
            )

            # Safety check
            if signal_idx < 0 or signal_idx >= task.read_length: continue

            # Fetch Reference Base (Forward Strand)
            base = fasta.fetch(reference_name, r_pos, r_pos + 1)
            if not base: continue
            
            # If Reverse:
            # Signal was 'T'. Ref (Forward) is 'A'.
            # We want the Label to be 'T' (to match signal).
            # So we take Complement('A') -> 'T'.
            if task.is_reverse:
                base = complement_base(base)
            
            full_labels[signal_idx] = BASE_TO_INT.get(base.upper(), 0)
            
        except Exception:
            continue
    # -----------------------------------------------------------------------

    # 4. Filter by Accuracy
    if task.nm_tag is not None:
        approx_identity = 1.0 - (task.nm_tag / task.read_length)
        if approx_identity < 0.90: 
            stats['records_low_accuracy_prefilter'] += 1

    # 5. Map Signal to Bases
    mv = np.asarray(task.mv_tag, dtype=np.int64)
    if mv.size <= 1:
        stats["records_invalid_move_table"] += 1
        return samples, stats
    stride = int(mv[0])
    moves = mv[1:]
    ts_offset = int(task.ts_tag) if task.ts_tag is not None else 0
    base_pos_arr = ts_offset + (np.arange(len(moves)) * stride) + (stride // 2)
    base_pos_arr = base_pos_arr[moves == 1]

    # 6. Chunking (Left Drop)
    chunk_len = task.signal_len_arg
    overlap = task.overlap_arg
    chunk_stride = task.stride_arg
    signal_len = pa_signal.shape[0]
    
    if signal_len < chunk_len: return samples, stats
    _, offset_start = divmod(signal_len - chunk_len, chunk_stride)
    
    for win_start in range(offset_start, signal_len - chunk_len + 1, chunk_stride):
        stats["chunks_seen"] += 1
        win_end = win_start + chunk_len
        left_idx = np.searchsorted(base_pos_arr, win_start, side="left")
        right_idx = np.searchsorted(base_pos_arr, win_end, side="left")
        
        if left_idx >= right_idx:
            stats["chunks_zerolen_sequence"] += 1
            continue
        if right_idx > len(full_labels): right_idx = len(full_labels)
        
        label_seq = full_labels[left_idx:right_idx]

        # 7. Strict Filter
        if np.any(label_seq == 0): # Rejects SoftClips & Insertions
            stats['chunks_softclip_or_insertion'] += 1
            continue
        
        # Local Acc Check
        if task.nm_tag is not None:
             approx_identity = 1.0 - (task.nm_tag / task.read_length)
             if approx_identity < 0.85:
                 stats['chunks_low_accuracy'] += 1
                 continue

        if label_seq.size < 5:
            stats["chunks_too_short"] += 1
            continue
        if task.max_label_len is not None and label_seq.size > task.max_label_len:
            stats["chunks_too_long"] += 1
            continue

        signal_window = pa_signal[win_start:win_end]
        normalized_signal = (signal_window - global_shift) / global_scale
        normalized_signal = np.clip(normalized_signal, -task.clip_value, task.clip_value)
        
        samples.append(Sample(
            signal=normalized_signal.astype(np.float16),
            label=label_seq.astype(np.uint8),
            label_len=int(label_seq.size)
        ))
        stats["chunks_written"] += 1

    return samples, stats

# [Boilerplate continues same as before...]
def build_pod5_lookup(pod5_dir):
    lookup = {}
    pod5_paths = collect_pod5_paths(pod5_dir)
    for pod5_path in tqdm(pod5_paths, desc="Indexing POD5"):
        with pod5.Reader(pod5_path) as reader:
            for batch_idx in range(reader.batch_count):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_reads):
                    read = batch.get_read(row_idx)
                    lookup[str(read.read_id)] = (pod5_path, batch_idx, row_idx)
    return lookup

def flush_temp_chunk(chunk_id, chunk, temp_dir, manifest_list):
    if not chunk: return chunk_id
    signals = np.stack([s.signal for s in chunk], axis=0)
    lengths = np.asarray([s.label_len for s in chunk], dtype=np.uint16)
    offsets = np.zeros(len(chunk) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    labels_flat = np.concatenate([s.label for s in chunk], axis=0)
    sig_path = os.path.join(temp_dir, f"temp_sig_{chunk_id}.npy")
    lbl_path = os.path.join(temp_dir, f"temp_lbl_{chunk_id}.npy")
    off_path = os.path.join(temp_dir, f"temp_off_{chunk_id}.npy")
    len_path = os.path.join(temp_dir, f"temp_len_{chunk_id}.npy")
    np.save(sig_path, signals)
    np.save(lbl_path, labels_flat)
    np.save(off_path, offsets)
    np.save(len_path, lengths)
    manifest_list.append({"signals": sig_path, "labels": lbl_path, "offsets": off_path, "lengths": len_path, "num_samples": int(signals.shape[0])})
    return chunk_id + 1

def build_chunk_ranges(chunk_manifest):
    starts = [0]
    total = 0
    for chunk_info in chunk_manifest:
        total += int(chunk_info["num_samples"])
        starts.append(total)
    return starts

def find_chunk_index(starts, index):
    return bisect.bisect_right(starts, index) - 1

def merge_counters(total, update):
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)

def merge_chunks_to_final(output_dir, chunk_manifest, signal_len, max_label_len):
    if not chunk_manifest:
        return None
    lengths_list = []
    for chunk_info in tqdm(chunk_manifest, desc="Loading lengths"):
        lengths_list.append(np.load(chunk_info["lengths"]))
    lengths = np.concatenate(lengths_list, axis=0)
    indices = typical_indices(lengths)
    if indices.size == 0:
        print("No samples remained.")
        return None
    if max_label_len is None:
        max_label_len = int(lengths[indices].max())
    indices = np.random.permutation(indices)
    total_samples = int(indices.size)
    final_sig_path = os.path.join(output_dir, "chunks.npy")
    final_lbl_path = os.path.join(output_dir, "references.npy")
    final_len_path = os.path.join(output_dir, "reference_lengths.npy")
    fp_sig = np.lib.format.open_memmap(final_sig_path, mode="w+", dtype=np.float16, shape=(total_samples, signal_len))
    fp_lbl = np.lib.format.open_memmap(final_lbl_path, mode="w+", dtype=np.uint8, shape=(total_samples, max_label_len))
    fp_len = np.lib.format.open_memmap(final_len_path, mode="w+", dtype=np.uint16, shape=(total_samples,))
    starts = build_chunk_ranges(chunk_manifest)
    chunk_cache = {}
    def load_chunk_data(idx):
        if idx in chunk_cache: return chunk_cache[idx]
        info = chunk_manifest[idx]
        chunk_cache[idx] = (np.load(info["signals"], mmap_mode="r"), np.load(info["labels"], mmap_mode="r"), np.load(info["offsets"], mmap_mode="r"), np.load(info["lengths"], mmap_mode="r"))
        return chunk_cache[idx]
    block_size = 2048
    for out_start in tqdm(range(0, total_samples, block_size), desc="Writing dataset"):
        out_end = min(out_start + block_size, total_samples)
        block_indices = indices[out_start:out_end]
        block_signals = np.empty((out_end - out_start, signal_len), dtype=np.float16)
        block_labels = np.zeros((out_end - out_start, max_label_len), dtype=np.uint8)
        block_lengths = lengths[block_indices].astype(np.uint16)
        for pos, global_index in enumerate(block_indices):
            chunk_idx = find_chunk_index(starts, int(global_index))
            local_index = int(global_index - starts[chunk_idx])
            signals, labels, offsets, lengths_chunk = load_chunk_data(chunk_idx)
            block_signals[pos] = signals[local_index]
            l_len = int(lengths_chunk[local_index])
            l_start = int(offsets[local_index])
            block_labels[pos, :l_len] = labels[l_start:l_start + l_len]
        fp_sig[out_start:out_end] = block_signals
        fp_lbl[out_start:out_end] = block_labels
        fp_len[out_start:out_end] = block_lengths
    del fp_sig, fp_lbl, fp_len
    for chunk_info in chunk_manifest:
        os.remove(chunk_info["signals"])
        os.remove(chunk_info["labels"])
        os.remove(chunk_info["offsets"])
        os.remove(chunk_info["lengths"])
    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {total_samples} chunks.")
    return {
        "total_pre_typical_filter": int(lengths.shape[0]),
        "total_post_typical_filter": int(indices.size),
        "total_written": int(total_samples),
        "max_label_len": int(max_label_len),
    }

def write_summary(output_dir, args, counters, merge_stats):
    chunk_accept_rate = None
    if counters.get("chunks_seen", 0) > 0:
        chunk_accept_rate = counters.get("chunks_written", 0) / counters["chunks_seen"]
    summary = {
        "bam_file": os.path.abspath(args.bam_file),
        "pod5_dir": os.path.abspath(args.pod5_dir),
        "reference_fasta": os.path.abspath(args.reference_fasta),
        "output_dir": os.path.abspath(args.output_dir),
        "sample_type": args.sample_type,
        "chunk_len": int(args.chunk_len),
        "overlap": int(args.overlap),
        "stride": int(args.stride) if args.stride is not None else int(args.chunk_len - args.overlap),
        "max_chunks": None if int(args.max_chunks) <= 0 else int(args.max_chunks),
        "norm_strategy": args.norm_strategy,
        "clip_value": float(args.clip_value),
        "accepted_rate_step3": chunk_accept_rate,
        "counters": {key: int(value) for key, value in sorted(counters.items())},
        "merge": merge_stats,
    }
    with open(os.path.join(output_dir, "dataset_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

def get_total_reads_from_index(bam_path):
    try:
        lines = pysam.idxstats(bam_path).splitlines()
        return sum(int(x.split("\t")[2]) for x in lines)
    except: return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-file", required=True)
    parser.add_argument("--pod5-dir", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-type", choices=["dna", "rna"], default="dna")
    parser.add_argument("--chunk-len", type=int, default=12000)
    parser.add_argument("--overlap", type=int, default=600)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--max-chunks", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--norm-strategy", choices=["quantile", "pa"], default="quantile")
    parser.add_argument("--pa-mean", type=float, default=0.0)
    parser.add_argument("--pa-std", type=float, default=1.0)
    return parser.parse_args()

def main():
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["TMPDIR"] = "/tmp"
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    stride = args.stride if args.stride else args.chunk_len - args.overlap
    print(f"[config] sample_type={args.sample_type}")
    print("[1/5] Building POD5 index...")
    lookup = build_pod5_lookup(args.pod5_dir)
    print(f"      Found {len(lookup)} reads.")
    print("[2/5] Preparing reference mapping...")
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")
    fasta = pysam.FastaFile(args.reference_fasta)
    reference_mapping = build_reference_mapping(bam_file.header, fasta)
    fasta.close()
    total_est = get_total_reads_from_index(args.bam_file)
    chunk_manifest = []
    chunk_buffer = []
    chunk_id = 0
    total_written_chunks = 0
    counters = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(args.reference_fasta, lookup, reference_mapping)) as executor:
        futures = set()
        for read in tqdm(bam_file, total=total_est, unit="read", desc="Dispatching"):
            if args.max_chunks > 0 and total_written_chunks >= args.max_chunks: break
            if read.is_unmapped or not read.has_tag("mv") or read.query_name not in lookup: continue
            
            # aligned_pairs
            aligned_pairs = read.get_aligned_pairs(matches_only=False)
            
            task = TaskData(
                read_id=read.query_name, read_length=read.query_length,
                reference_name=read.reference_name, is_reverse=read.is_reverse,
                sample_type=args.sample_type,
                ts_tag=read.get_tag("ts") if read.has_tag("ts") else 0,
                mv_tag=read.get_tag("mv"), nm_tag=read.get_tag("NM") if read.has_tag("NM") else None,
                signal_len_arg=args.chunk_len, stride_arg=stride, overlap_arg=args.overlap,
                max_label_len=args.max_label_len, clip_value=args.clip_value,
                norm_strategy=args.norm_strategy, pa_mean=args.pa_mean, pa_std=args.pa_std,
                aligned_pairs=aligned_pairs
            )
            futures.add(executor.submit(process_task, task))
            if len(futures) >= args.workers * 10:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                futures.difference_update(done)
                for f in done:
                    s, stats = f.result()
                    merge_counters(counters, stats)
                    chunk_buffer.extend(s)
                    total_written_chunks += stats["chunks_written"]
                if len(chunk_buffer) >= 5000:
                    chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
                    chunk_buffer = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Finishing"):
            s, stats = f.result()
            merge_counters(counters, stats)
            chunk_buffer.extend(s)
            total_written_chunks += stats["chunks_written"]
            if len(chunk_buffer) >= 5000:
                chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
                chunk_buffer = []
    if chunk_buffer:
        chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
    print("[4/5] Merging...")
    merge_stats = merge_chunks_to_final(args.output_dir, chunk_manifest, args.chunk_len, args.max_label_len)
    if merge_stats is not None:
        write_summary(args.output_dir, args, counters, merge_stats)
    print("Reject counters:")
    for key in sorted(counters):
        print(f" - {key}: {counters[key]}")
    if counters.get("chunks_seen", 0) > 0:
        accepted_rate = counters.get("chunks_written", 0) / counters["chunks_seen"]
        print(f"accepted_rate_step3: {accepted_rate:.4%}")
    try: shutil.rmtree(temp_dir)
    except: pass

if __name__ == "__main__":
    main()
