#!/usr/bin/env python3
"""
Compare multi-head model basecalls against an official model BAM.

Inputs:
- basecalls.tsv from validate/predict_mods_from_pod5.py
- official basecalling BAM/CRAM/SAM

Outputs:
- per_read_comparison.tsv
- worst_reads.tsv
- summary.json
- summary.txt
"""

from __future__ import annotations

import csv
import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import pysam
from edlib import align as edlib_align


CIGAR_RE = re.compile(r"(\d+)([=XID])")


def normalize_seq(seq: str, convert_u_to_t: bool = True) -> str:
    text = str(seq or "").upper()
    return text.replace("U", "T") if convert_u_to_t else text


def parse_cigar_counts(cigar: str) -> Dict[str, int]:
    counts = {"=": 0, "X": 0, "I": 0, "D": 0}
    for count, op in CIGAR_RE.findall(cigar or ""):
        counts[op] += int(count)
    return counts


def compare_pair(pred_seq: str, official_seq: str) -> Dict[str, object]:
    result = edlib_align(pred_seq, official_seq, mode="NW", task="path")
    counts = parse_cigar_counts(result.get("cigar", ""))
    aligned = counts["="] + counts["X"] + counts["I"] + counts["D"]
    identity = (counts["="] / aligned) if aligned else 0.0
    return {
        "edit_distance": int(result.get("editDistance", -1)),
        "identity": float(identity),
        "matches": int(counts["="]),
        "mismatches": int(counts["X"]),
        "insertions": int(counts["I"]),
        "deletions": int(counts["D"]),
    }


def load_predicted_basecalls(path: Path, convert_u_to_t: bool) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            read_id = str(row["read_id"])
            rows[read_id] = {
                "sequence": normalize_seq(row.get("sequence", ""), convert_u_to_t=convert_u_to_t),
                "filename": str(row.get("filename", "")),
                "run_id": str(row.get("run_id", "")),
            }
    return rows


def load_official_bam_sequences(path: Path, convert_u_to_t: bool, primary_only: bool) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        for read in bam:
            if primary_only and (read.is_secondary or read.is_supplementary):
                continue
            if read.query_sequence is None:
                continue
            read_id = str(read.query_name)
            if read_id in sequences:
                continue
            sequences[read_id] = normalize_seq(read.query_sequence, convert_u_to_t=convert_u_to_t)
    return sequences


def build_text_summary(summary: Dict[str, object]) -> str:
    lines = [
        "[inputs]",
        f"predicted_basecalls_tsv: {summary['predicted_basecalls_tsv']}",
        f"official_bam: {summary['official_bam']}",
        "",
        "[counts]",
        f"num_predicted_reads: {summary['counts']['num_predicted_reads']}",
        f"num_official_reads_indexed: {summary['counts']['num_official_reads_indexed']}",
        f"num_reads_compared: {summary['counts']['num_reads_compared']}",
        f"num_reads_missing_in_official_bam: {summary['counts']['num_reads_missing_in_official_bam']}",
        "",
        "[metrics]",
        f"mean_identity: {summary['metrics']['mean_identity']:.6f}",
        f"median_identity: {summary['metrics']['median_identity']:.6f}",
        f"mean_edit_distance: {summary['metrics']['mean_edit_distance']:.3f}",
        f"same_length_fraction: {summary['metrics']['same_length_fraction']:.6f}",
    ]
    return "\n".join(lines) + "\n"


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predicted = load_predicted_basecalls(Path(args.predicted_basecalls_tsv), convert_u_to_t=not args.keep_u)
    official = load_official_bam_sequences(Path(args.official_bam), convert_u_to_t=not args.keep_u, primary_only=not args.include_non_primary)

    rows: List[Dict[str, object]] = []
    missing = 0
    identities: List[float] = []
    edit_distances: List[int] = []
    same_length_count = 0

    for read_id, pred_record in predicted.items():
        official_seq = official.get(read_id)
        if official_seq is None:
            missing += 1
            continue

        pred_seq = pred_record["sequence"]
        best_compare = compare_pair(pred_seq, official_seq)
        orientation = "forward"
        if args.allow_reverse:
            reversed_compare = compare_pair(pred_seq[::-1], official_seq)
            if reversed_compare["identity"] > best_compare["identity"]:
                best_compare = reversed_compare
                orientation = "reversed"

        pred_len = len(pred_seq)
        official_len = len(official_seq)
        same_length = int(pred_len == official_len)
        same_length_count += same_length
        identities.append(float(best_compare["identity"]))
        edit_distances.append(int(best_compare["edit_distance"]))

        rows.append({
            "read_id": read_id,
            "filename": pred_record["filename"],
            "run_id": pred_record["run_id"],
            "orientation_used": orientation,
            "pred_length": pred_len,
            "official_length": official_len,
            "length_delta": pred_len - official_len,
            "same_length": same_length,
            "identity": float(best_compare["identity"]),
            "edit_distance": int(best_compare["edit_distance"]),
            "matches": int(best_compare["matches"]),
            "mismatches": int(best_compare["mismatches"]),
            "insertions": int(best_compare["insertions"]),
            "deletions": int(best_compare["deletions"]),
            "pred_sequence_prefix": pred_seq[:120],
            "official_sequence_prefix": official_seq[:120],
        })

    rows.sort(key=lambda row: (row["identity"], row["edit_distance"], row["read_id"]))

    per_read_tsv = out_dir / "per_read_comparison.tsv"
    worst_tsv = out_dir / "worst_reads.tsv"

    fieldnames = [
        "read_id",
        "filename",
        "run_id",
        "orientation_used",
        "pred_length",
        "official_length",
        "length_delta",
        "same_length",
        "identity",
        "edit_distance",
        "matches",
        "mismatches",
        "insertions",
        "deletions",
        "pred_sequence_prefix",
        "official_sequence_prefix",
    ]
    with per_read_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with worst_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows[: args.num_examples])

    num_compared = len(rows)
    summary = {
        "predicted_basecalls_tsv": str(Path(args.predicted_basecalls_tsv).resolve()),
        "official_bam": str(Path(args.official_bam).resolve()),
        "artifacts": {
            "per_read_comparison_tsv": str(per_read_tsv.resolve()),
            "worst_reads_tsv": str(worst_tsv.resolve()),
        },
        "counts": {
            "num_predicted_reads": int(len(predicted)),
            "num_official_reads_indexed": int(len(official)),
            "num_reads_compared": int(num_compared),
            "num_reads_missing_in_official_bam": int(missing),
        },
        "metrics": {
            "mean_identity": float(sum(identities) / num_compared) if num_compared else 0.0,
            "median_identity": float(median(identities)) if identities else 0.0,
            "mean_edit_distance": float(sum(edit_distances) / num_compared) if num_compared else 0.0,
            "same_length_fraction": float(same_length_count / num_compared) if num_compared else 0.0,
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    (out_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")
    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {out_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("--predicted-basecalls-tsv", type=Path, required=True)
    parser.add_argument("--official-bam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--allow-reverse", action="store_true", default=True)
    parser.add_argument("--no-allow-reverse", dest="allow_reverse", action="store_false")
    parser.add_argument("--include-non-primary", action="store_true", default=False)
    parser.add_argument("--keep-u", action="store_true", default=False, help="Do not normalize U->T before comparison")
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
