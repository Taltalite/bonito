#!/usr/bin/env python3
"""
Diagnose a single read through official basecaller vs standalone mod-head basecaller.

Layers compared:
1. Input signal after Reader normalization/trimming
2. Chunk-level base_scores and chunk-level decode
3. Read-level stitched sequence
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from edlib import align as edlib_align
from koi.decode import to_str

from bonito.crf.basecall import basecall as official_basecall
from bonito.crf.basecall import decode_scores
from bonito.reader import Reader
from bonito.transformer.multihead_basecall import basecall as standalone_basecall
from bonito.util import chunk, init, load_model


CIGAR_RE = re.compile(r"(\d+)([=XID])")


def parse_cigar_counts(cigar: str) -> Dict[str, int]:
    counts = {"=": 0, "X": 0, "I": 0, "D": 0}
    for count, op in CIGAR_RE.findall(cigar or ""):
        counts[op] += int(count)
    return counts


def compare_pair(query: str, target: str) -> Dict[str, object]:
    result = edlib_align(query, target, mode="NW", task="path")
    counts = parse_cigar_counts(result.get("cigar", ""))
    aligned = counts["="] + counts["X"] + counts["I"] + counts["D"]
    return {
        "identity": float(counts["="] / aligned) if aligned else 0.0,
        "edit_distance": int(result.get("editDistance", -1)),
        "matches": int(counts["="]),
        "mismatches": int(counts["X"]),
        "insertions": int(counts["I"]),
        "deletions": int(counts["D"]),
        "cigar": str(result.get("cigar", "")),
    }


def norm_params_from_model(model) -> Dict[str, object] | None:
    scaling = model.config.get("scaling")
    if scaling and scaling.get("strategy") == "pa":
        return model.config.get("standardisation")
    return model.config.get("normalisation")


def load_single_read(reader: Reader, reads_directory: str, model, read_id: str, recursive: bool, do_trim: bool):
    reads = reader.get_reads(
        reads_directory,
        n_proc=1,
        recursive=recursive,
        read_ids={read_id},
        skip=False,
        do_trim=do_trim,
        scaling_strategy=model.config.get("scaling"),
        norm_params=norm_params_from_model(model),
        cancel=None,
    )
    try:
        return next(reads)
    except StopIteration as exc:
        raise RuntimeError(f"Read not found in pod5 input: {read_id}") from exc


def summarize_read_signal(read) -> Dict[str, object]:
    signal = np.asarray(read.signal, dtype=np.float32)
    return {
        "read_id": str(read.read_id),
        "num_samples": int(read.num_samples),
        "trimmed_samples": int(read.trimmed_samples),
        "signal_length": int(signal.shape[0]),
        "shift": float(read.shift),
        "scale": float(read.scale),
        "signal_mean": float(signal.mean()) if signal.size else 0.0,
        "signal_std": float(signal.std()) if signal.size else 0.0,
        "scaling_strategy": str(read.scaling_strategy),
    }


def compare_signals(signal_a: np.ndarray, signal_b: np.ndarray) -> Dict[str, object]:
    signal_a = np.asarray(signal_a, dtype=np.float32)
    signal_b = np.asarray(signal_b, dtype=np.float32)
    same_length = signal_a.shape == signal_b.shape
    common = min(signal_a.shape[0], signal_b.shape[0])
    if common == 0:
        return {
            "same_length": bool(same_length),
            "common_length": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "allclose_atol_1e-5_rtol_1e-4": False,
        }
    left = signal_a[:common]
    right = signal_b[:common]
    diff = np.abs(left - right)
    return {
        "same_length": bool(same_length),
        "common_length": int(common),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_atol_1e-5_rtol_1e-4": bool(np.allclose(left, right, atol=1e-5, rtol=1e-4)),
    }


def chunk_tensor_from_read(read, chunksize: int, overlap: int) -> torch.Tensor:
    return chunk(torch.from_numpy(read.signal), chunksize, overlap)


def decode_chunk_scores(model, scores: torch.Tensor, rna: bool) -> Dict[str, object]:
    attrs = decode_scores(scores, model.seqdist, reverse=False)
    sequence = to_str(attrs["sequence"])
    qstring = to_str(attrs["qstring"])
    if rna:
        sequence = sequence[::-1]
        qstring = qstring[::-1]
    moves = attrs["moves"]
    if isinstance(moves, torch.Tensor):
        moves = moves.detach().cpu().numpy()
    return {
        "sequence": sequence,
        "qstring": qstring,
        "moves_len": int(len(moves)),
    }


def compare_chunk_layers(
    official_model,
    standalone_model,
    official_chunks: torch.Tensor,
    standalone_chunks: torch.Tensor,
    *,
    rna: bool,
    max_chunks: int,
) -> Dict[str, object]:
    device = next(official_model.parameters()).device
    official_dtype = next(official_model.parameters()).dtype
    standalone_dtype = next(standalone_model.parameters()).dtype

    chunk_count = min(int(official_chunks.shape[0]), int(standalone_chunks.shape[0]), int(max_chunks))
    score_rows: List[Dict[str, object]] = []
    decode_rows: List[Dict[str, object]] = []

    with torch.inference_mode():
        for chunk_index in range(chunk_count):
            official_batch = official_chunks[chunk_index:chunk_index + 1].to(device=device, dtype=official_dtype)
            standalone_batch = standalone_chunks[chunk_index:chunk_index + 1].to(device=device, dtype=standalone_dtype)

            official_scores = official_model(official_batch)
            standalone_outputs = standalone_model(standalone_batch)
            standalone_scores = standalone_outputs["base_scores"]

            row = {
                "chunk_index": int(chunk_index),
                "official_shape": list(map(int, official_scores.shape)),
                "standalone_shape": list(map(int, standalone_scores.shape)),
                "same_shape": bool(tuple(official_scores.shape) == tuple(standalone_scores.shape)),
            }
            if row["same_shape"]:
                diff = (official_scores.to(torch.float32) - standalone_scores.to(torch.float32)).abs()
                row["max_abs_diff"] = float(diff.max().item())
                row["mean_abs_diff"] = float(diff.mean().item())
                row["allclose_atol_1e-5_rtol_1e-4"] = bool(
                    torch.allclose(
                        official_scores.to(torch.float32),
                        standalone_scores.to(torch.float32),
                        atol=1e-5,
                        rtol=1e-4,
                    )
                )
            else:
                row["max_abs_diff"] = None
                row["mean_abs_diff"] = None
                row["allclose_atol_1e-5_rtol_1e-4"] = False
            score_rows.append(row)

            official_decoded = decode_chunk_scores(official_model, official_scores, rna=rna)
            standalone_decoded = decode_chunk_scores(standalone_model, standalone_scores, rna=rna)
            compare = compare_pair(standalone_decoded["sequence"], official_decoded["sequence"])
            decode_rows.append({
                "chunk_index": int(chunk_index),
                "official_sequence_length": int(len(official_decoded["sequence"])),
                "standalone_sequence_length": int(len(standalone_decoded["sequence"])),
                "exact_match": bool(official_decoded["sequence"] == standalone_decoded["sequence"]),
                "identity": float(compare["identity"]),
                "edit_distance": int(compare["edit_distance"]),
                "official_prefix": official_decoded["sequence"][:120],
                "standalone_prefix": standalone_decoded["sequence"][:120],
            })

    return {
        "num_official_chunks": int(official_chunks.shape[0]),
        "num_standalone_chunks": int(standalone_chunks.shape[0]),
        "num_chunks_compared": int(chunk_count),
        "score_rows": score_rows,
        "decode_rows": decode_rows,
    }


def summarize_chunk_comparison(layer: Dict[str, object]) -> Dict[str, object]:
    score_rows = layer["score_rows"]
    decode_rows = layer["decode_rows"]
    score_diffs = [row["max_abs_diff"] for row in score_rows if row["max_abs_diff"] is not None]
    decode_identities = [row["identity"] for row in decode_rows]
    return {
        "num_official_chunks": int(layer["num_official_chunks"]),
        "num_standalone_chunks": int(layer["num_standalone_chunks"]),
        "num_chunks_compared": int(layer["num_chunks_compared"]),
        "score_allclose_fraction": (
            float(sum(1 for row in score_rows if row["allclose_atol_1e-5_rtol_1e-4"]) / len(score_rows))
            if score_rows else 0.0
        ),
        "score_max_abs_diff_max": float(max(score_diffs)) if score_diffs else None,
        "score_max_abs_diff_mean": float(sum(score_diffs) / len(score_diffs)) if score_diffs else None,
        "decode_exact_match_fraction": (
            float(sum(1 for row in decode_rows if row["exact_match"]) / len(decode_rows))
            if decode_rows else 0.0
        ),
        "decode_mean_identity": float(sum(decode_identities) / len(decode_identities)) if decode_identities else 0.0,
    }


def run_read_level_basecall(official_model, standalone_model, official_read, standalone_read, *, rna: bool) -> Dict[str, object]:
    official_result = list(
        official_basecall(
            official_model,
            [official_read],
            batchsize=official_model.config["basecaller"]["batchsize"],
            chunksize=official_model.config["basecaller"]["chunksize"],
            overlap=official_model.config["basecaller"]["overlap"],
            reverse=False,
            rna=rna,
        )
    )[0][1]

    standalone_result = list(
        standalone_basecall(
            standalone_model,
            [standalone_read],
            batchsize=standalone_model.config["basecaller"]["batchsize"],
            chunksize=standalone_model.config["basecaller"]["chunksize"],
            overlap=standalone_model.config["basecaller"]["overlap"],
            reverse=False,
            rna=rna,
            mod_threshold=0.5,
        )
    )[0][1]

    forward_compare = compare_pair(standalone_result["sequence"], official_result["sequence"])
    reversed_compare = compare_pair(standalone_result["sequence"][::-1], official_result["sequence"])
    return {
        "official_length": int(len(official_result["sequence"])),
        "standalone_length": int(len(standalone_result["sequence"])),
        "same_length": bool(len(official_result["sequence"]) == len(standalone_result["sequence"])),
        "exact_match": bool(official_result["sequence"] == standalone_result["sequence"]),
        "forward_compare": forward_compare,
        "reversed_compare": reversed_compare,
        "official_prefix": official_result["sequence"][:200],
        "standalone_prefix": standalone_result["sequence"][:200],
        "official_suffix": official_result["sequence"][-200:],
        "standalone_suffix": standalone_result["sequence"][-200:],
    }


def build_text_report(report: Dict[str, object]) -> str:
    lines = [
        "[inputs]",
        f"read_id: {report['inputs']['read_id']}",
        f"official_model: {report['inputs']['official_model']}",
        f"standalone_model: {report['inputs']['standalone_model']}",
        f"reads_directory: {report['inputs']['reads_directory']}",
        "",
        "[signal]",
        f"official_signal_length: {report['signal']['official']['signal_length']}",
        f"standalone_signal_length: {report['signal']['standalone']['signal_length']}",
        f"same_length: {int(report['signal']['compare']['same_length'])}",
        f"common_length: {report['signal']['compare']['common_length']}",
        f"max_abs_diff: {report['signal']['compare']['max_abs_diff']}",
        f"mean_abs_diff: {report['signal']['compare']['mean_abs_diff']}",
        "",
        "[chunk_scores]",
        f"num_chunks_compared: {report['chunk_summary']['num_chunks_compared']}",
        f"score_allclose_fraction: {report['chunk_summary']['score_allclose_fraction']:.6f}",
        f"score_max_abs_diff_max: {report['chunk_summary']['score_max_abs_diff_max']}",
        f"score_max_abs_diff_mean: {report['chunk_summary']['score_max_abs_diff_mean']}",
        "",
        "[chunk_decode]",
        f"decode_exact_match_fraction: {report['chunk_summary']['decode_exact_match_fraction']:.6f}",
        f"decode_mean_identity: {report['chunk_summary']['decode_mean_identity']:.6f}",
        "",
        "[stitched_sequence]",
        f"same_length: {int(report['read_level']['same_length'])}",
        f"exact_match: {int(report['read_level']['exact_match'])}",
        f"forward_identity: {report['read_level']['forward_compare']['identity']:.6f}",
        f"forward_edit_distance: {report['read_level']['forward_compare']['edit_distance']}",
        f"reversed_identity: {report['read_level']['reversed_compare']['identity']:.6f}",
        f"reversed_edit_distance: {report['read_level']['reversed_compare']['edit_distance']}",
    ]
    return "\n".join(lines) + "\n"


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init(args.seed, args.device)
    reader = Reader(args.reads_directory, args.recursive)

    official_model = load_model(
        args.official_model,
        args.device,
        half=not args.no_half,
        use_koi=False,
        compile=False,
    )
    standalone_model = load_model(
        args.standalone_model,
        args.device,
        half=not args.no_half,
        use_koi=False,
        compile=False,
    )

    do_trim = not args.no_trim
    official_read = load_single_read(reader, args.reads_directory, official_model, args.read_id, args.recursive, do_trim)
    standalone_read = load_single_read(reader, args.reads_directory, standalone_model, args.read_id, args.recursive, do_trim)

    official_chunks = chunk_tensor_from_read(
        official_read,
        official_model.config["basecaller"]["chunksize"],
        official_model.config["basecaller"]["overlap"],
    )
    standalone_chunks = chunk_tensor_from_read(
        standalone_read,
        standalone_model.config["basecaller"]["chunksize"],
        standalone_model.config["basecaller"]["overlap"],
    )

    chunk_layer = compare_chunk_layers(
        official_model,
        standalone_model,
        official_chunks,
        standalone_chunks,
        rna=args.rna,
        max_chunks=args.max_chunks,
    )
    report = {
        "inputs": {
            "read_id": str(args.read_id),
            "official_model": str(Path(args.official_model).resolve()),
            "standalone_model": str(Path(args.standalone_model).resolve()),
            "reads_directory": str(Path(args.reads_directory).resolve()),
            "device": str(args.device),
            "rna": bool(args.rna),
            "recursive": bool(args.recursive),
            "do_trim": bool(do_trim),
        },
        "signal": {
            "official": summarize_read_signal(official_read),
            "standalone": summarize_read_signal(standalone_read),
            "compare": compare_signals(official_read.signal, standalone_read.signal),
        },
        "chunk_summary": summarize_chunk_comparison(chunk_layer),
        "chunk_details": chunk_layer,
        "read_level": run_read_level_basecall(
            official_model,
            standalone_model,
            official_read,
            standalone_read,
            rna=args.rna,
        ),
    }

    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "diagnostic_report.txt").write_text(build_text_report(report), encoding="utf-8")

    print(f"read_id: {args.read_id}")
    print(f"output_dir: {output_dir}")
    print(f"forward_identity: {report['read_level']['forward_compare']['identity']:.6f}")
    print(f"decode_mean_identity: {report['chunk_summary']['decode_mean_identity']:.6f}")
    print(f"score_allclose_fraction: {report['chunk_summary']['score_allclose_fraction']:.6f}")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("official_model", help="Official basecaller model directory.")
    parser.add_argument("standalone_model", help="Standalone mod-head model directory.")
    parser.add_argument("reads_directory", help="pod5 input directory.")
    parser.add_argument("--read-id", required=True, help="Single read id to diagnose.")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic_report.json/txt.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--max-chunks", default=8, type=int, help="Maximum number of chunks to compare at chunk level.")
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--no-half", action="store_true", default=False, help="Load models in fp32 for diagnosis.")
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
