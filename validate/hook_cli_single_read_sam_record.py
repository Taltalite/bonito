#!/usr/bin/env python3
"""
Capture the real sequence/qstring/tags passed into bonito.io.sam_record for a
single read from:
1. bonito.cli.basecaller
2. bonito.cli.basecaller_mod
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List
from unittest import mock

from edlib import align as edlib_align

import bonito.cli.basecaller as cli_basecaller
import bonito.cli.basecaller_mod as cli_basecaller_mod
import bonito.io as bonito_io


CIGAR_RE = re.compile(r"(\d+)([=XID])")


def parse_cigar_counts(cigar: str) -> Dict[str, int]:
    counts = {"=": 0, "X": 0, "I": 0, "D": 0}
    for count, op in CIGAR_RE.findall(cigar or ""):
        counts[op] += int(count)
    return counts


def compare_pair(left: str, right: str) -> Dict[str, object]:
    result = edlib_align(left, right, mode="NW", task="path")
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


class DummyCSVLogger:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def append(self, row):
        return None


class DummyAlignmentFile:
    def __init__(self, *args, **kwargs):
        self.header = kwargs.get("header")

    def write(self, *args, **kwargs):
        return None


def sanitize_capture(capture: Dict[str, object]) -> Dict[str, object]:
    sequence = str(capture.get("sequence", ""))
    qstring = str(capture.get("qstring", ""))
    tags = list(capture.get("tags", []))
    return {
        "read_id": str(capture.get("read_id", "")),
        "sequence": sequence,
        "qstring": qstring,
        "tags": tags,
        "mapping_present": bool(capture.get("mapping_present", False)),
        "sequence_length": int(len(sequence)),
        "qstring_length": int(len(qstring)),
        "tags_count": int(len(tags)),
        "sequence_prefix": sequence[:200],
        "qstring_prefix": qstring[:200],
        "tags_prefix": tags[:20],
    }


@contextmanager
def capture_sam_record(target_read_id: str):
    original_sam_record = bonito_io.sam_record
    captures: List[Dict[str, object]] = []

    def wrapped_sam_record(read_id, sequence, qstring, mapping, tags=None, sep="\t"):
        if str(read_id) == target_read_id:
            captures.append({
                "read_id": str(read_id),
                "sequence": str(sequence),
                "qstring": str(qstring),
                "tags": list(tags or []),
                "mapping_present": bool(mapping),
            })
        return original_sam_record(read_id, sequence, qstring, mapping, tags=tags, sep=sep)

    with mock.patch.object(bonito_io, "sam_record", wrapped_sam_record), \
         mock.patch.object(bonito_io, "AlignmentFile", DummyAlignmentFile), \
         mock.patch.object(bonito_io, "CSVLogger", DummyCSVLogger):
        yield captures


def run_cli_capture(cli_module, argv: List[str], target_read_id: str) -> Dict[str, object]:
    parsed = cli_module.argparser().parse_args(argv)
    with capture_sam_record(target_read_id) as captures, \
         mock.patch.object(cli_module, "tqdm", lambda iterable, **kwargs: iterable), \
         mock.patch.object(cli_module, "biofmt", lambda aligned=False: bonito_io.Format("unaligned", "bam", "wb")):
        cli_module.main(parsed)
    if not captures:
        raise RuntimeError(f"sam_record was not called for target read_id: {target_read_id}")
    return sanitize_capture(captures[0])


def build_summary(official_capture: Dict[str, object], standalone_capture: Dict[str, object]) -> Dict[str, object]:
    return {
        "official_capture": official_capture,
        "standalone_capture": standalone_capture,
        "compare": {
            "sequence_exact_match": bool(official_capture["sequence"] == standalone_capture["sequence"]),
            "qstring_exact_match": bool(official_capture["qstring"] == standalone_capture["qstring"]),
            "tags_exact_match": bool(official_capture["tags"] == standalone_capture["tags"]),
            "sequence_compare": compare_pair(standalone_capture["sequence"], official_capture["sequence"]),
            "qstring_compare": compare_pair(standalone_capture["qstring"], official_capture["qstring"]),
        },
    }


def build_text_report(summary: Dict[str, object]) -> str:
    lines = [
        "[official_capture]",
        f"sequence_length: {summary['official_capture']['sequence_length']}",
        f"qstring_length: {summary['official_capture']['qstring_length']}",
        f"tags_count: {summary['official_capture']['tags_count']}",
        f"sequence_prefix: {summary['official_capture']['sequence_prefix']}",
        "",
        "[standalone_capture]",
        f"sequence_length: {summary['standalone_capture']['sequence_length']}",
        f"qstring_length: {summary['standalone_capture']['qstring_length']}",
        f"tags_count: {summary['standalone_capture']['tags_count']}",
        f"sequence_prefix: {summary['standalone_capture']['sequence_prefix']}",
        "",
        "[compare]",
        f"sequence_exact_match: {int(summary['compare']['sequence_exact_match'])}",
        f"qstring_exact_match: {int(summary['compare']['qstring_exact_match'])}",
        f"tags_exact_match: {int(summary['compare']['tags_exact_match'])}",
        f"sequence_identity: {summary['compare']['sequence_compare']['identity']:.6f}",
        f"sequence_edit_distance: {summary['compare']['sequence_compare']['edit_distance']}",
        f"qstring_identity: {summary['compare']['qstring_compare']['identity']:.6f}",
        f"qstring_edit_distance: {summary['compare']['qstring_compare']['edit_distance']}",
    ]
    return "\n".join(lines) + "\n"


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as fh:
        fh.write(args.read_id + "\n")
        read_ids_path = fh.name

    try:
        official_argv = [
            args.official_model,
            args.reads_directory,
            "--device", args.device,
            "--read-ids", read_ids_path,
            "--no-use-koi",
        ]
        standalone_argv = [
            args.standalone_model,
            args.reads_directory,
            "--device", args.device,
            "--read-ids", read_ids_path,
            "--no-use-koi",
            "--mod-threshold", str(args.mod_threshold),
        ]

        if args.rna:
            official_argv.append("--rna")
            standalone_argv.append("--rna")
        if args.recursive:
            official_argv.append("--recursive")
            standalone_argv.append("--recursive")
        if args.no_trim:
            official_argv.append("--no-trim")
            standalone_argv.append("--no-trim")
        if args.verbose:
            official_argv.extend(["-v"] * int(args.verbose))
            standalone_argv.extend(["-v"] * int(args.verbose))

        official_capture = run_cli_capture(cli_basecaller, official_argv, args.read_id)
        standalone_capture = run_cli_capture(cli_basecaller_mod, standalone_argv, args.read_id)
    finally:
        Path(read_ids_path).unlink(missing_ok=True)

    summary = build_summary(official_capture, standalone_capture)

    (output_dir / "official_sam_record_capture.json").write_text(
        json.dumps(official_capture, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "standalone_sam_record_capture.json").write_text(
        json.dumps(standalone_capture, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "sam_record_capture_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "sam_record_capture_summary.txt").write_text(
        build_text_report(summary),
        encoding="utf-8",
    )

    print(f"read_id: {args.read_id}")
    print(f"output_dir: {output_dir}")
    print(f"sequence_exact_match: {int(summary['compare']['sequence_exact_match'])}")
    print(f"sequence_identity: {summary['compare']['sequence_compare']['identity']:.6f}")
    print(f"official_length: {summary['official_capture']['sequence_length']}")
    print(f"standalone_length: {summary['standalone_capture']['sequence_length']}")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("official_model", help="Official basecaller model directory.")
    parser.add_argument("standalone_model", help="Standalone mod-head model directory.")
    parser.add_argument("reads_directory", help="pod5 input directory.")
    parser.add_argument("--read-id", required=True, help="Single read id to capture.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/TXT outputs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--mod-threshold", default=0.5, type=float)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
