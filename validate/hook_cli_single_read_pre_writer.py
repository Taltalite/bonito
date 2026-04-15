#!/usr/bin/env python3
"""
Capture the real CLI pre-Writer `res` for a single read from:
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
from typing import Dict
from unittest import mock

import numpy as np
from edlib import align as edlib_align

import bonito.cli.basecaller as cli_basecaller
import bonito.cli.basecaller_mod as cli_basecaller_mod


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


def sanitize_res(res: Dict[str, object]) -> Dict[str, object]:
    moves = res.get("moves")
    if isinstance(moves, np.ndarray):
        moves_list = moves.tolist()
    elif moves is None:
        moves_list = None
    else:
        moves_list = list(moves)

    sequence = str(res.get("sequence", ""))
    qstring = str(res.get("qstring", ""))
    mods = list(res.get("mods", []))
    mapping = res.get("mapping")

    return {
        "stride": int(res.get("stride", 0)) if res.get("stride") is not None else None,
        "sequence": sequence,
        "qstring": qstring,
        "moves": moves_list,
        "mods": mods,
        "mapping_present": bool(mapping),
        "sequence_length": int(len(sequence)),
        "qstring_length": int(len(qstring)),
        "moves_length": int(len(moves_list)) if moves_list is not None else None,
        "mods_count": int(len(mods)),
    }


class CaptureWriter:
    captured = None

    def __init__(self, mode, iterator, *args, **kwargs):
        self.iterator = iterator
        self.log = []

    def start(self):
        for read, res in self.iterator:
            self.log.append((read.read_id, len(read.signal)))
            CaptureWriter.captured = {
                "read_id": str(read.read_id),
                "read_signal_length": int(len(read.signal)),
                "res": sanitize_res(res),
            }
            break

    def join(self):
        return None


@contextmanager
def capture_cli_writer(cli_module):
    CaptureWriter.captured = None
    with mock.patch.object(cli_module, "Writer", CaptureWriter), \
         mock.patch.object(cli_module, "tqdm", lambda iterable, **kwargs: iterable):
        yield


def run_basecaller_capture(args) -> Dict[str, object]:
    argv = [
        args.official_model,
        args.reads_directory,
        "--device", args.device,
        "--read-ids", args.read_ids_file,
        "--no-use-koi",
    ]
    if args.rna:
        argv.append("--rna")
    if args.recursive:
        argv.append("--recursive")
    if args.no_trim:
        argv.append("--no-trim")
    if args.verbose:
        argv.extend(["-v"] * int(args.verbose))

    parsed = cli_basecaller.argparser().parse_args(argv)
    with capture_cli_writer(cli_basecaller):
        cli_basecaller.main(parsed)
    if CaptureWriter.captured is None:
        raise RuntimeError("Failed to capture official basecaller Writer input.")
    return CaptureWriter.captured


def run_basecaller_mod_capture(args) -> Dict[str, object]:
    argv = [
        args.standalone_model,
        args.reads_directory,
        "--device", args.device,
        "--read-ids", args.read_ids_file,
        "--no-use-koi",
        "--mod-threshold", str(args.mod_threshold),
    ]
    if args.rna:
        argv.append("--rna")
    if args.recursive:
        argv.append("--recursive")
    if args.no_trim:
        argv.append("--no-trim")
    if args.verbose:
        argv.extend(["-v"] * int(args.verbose))

    parsed = cli_basecaller_mod.argparser().parse_args(argv)
    with capture_cli_writer(cli_basecaller_mod):
        cli_basecaller_mod.main(parsed)
    if CaptureWriter.captured is None:
        raise RuntimeError("Failed to capture basecaller_mod Writer input.")
    return CaptureWriter.captured


def build_summary(official_capture: Dict[str, object], standalone_capture: Dict[str, object]) -> Dict[str, object]:
    official_res = official_capture["res"]
    standalone_res = standalone_capture["res"]
    return {
        "official_capture": {
            "read_id": official_capture["read_id"],
            "read_signal_length": int(official_capture["read_signal_length"]),
            "sequence_length": int(official_res["sequence_length"]),
            "qstring_length": int(official_res["qstring_length"]),
            "moves_length": official_res["moves_length"],
            "mods_count": int(official_res["mods_count"]),
            "sequence_prefix": official_res["sequence"][:200],
            "qstring_prefix": official_res["qstring"][:200],
            "moves_prefix": official_res["moves"][:80] if official_res["moves"] is not None else None,
        },
        "standalone_capture": {
            "read_id": standalone_capture["read_id"],
            "read_signal_length": int(standalone_capture["read_signal_length"]),
            "sequence_length": int(standalone_res["sequence_length"]),
            "qstring_length": int(standalone_res["qstring_length"]),
            "moves_length": standalone_res["moves_length"],
            "mods_count": int(standalone_res["mods_count"]),
            "sequence_prefix": standalone_res["sequence"][:200],
            "qstring_prefix": standalone_res["qstring"][:200],
            "moves_prefix": standalone_res["moves"][:80] if standalone_res["moves"] is not None else None,
            "mods_prefix": standalone_res["mods"][:10],
        },
        "compare": {
            "sequence_exact_match": bool(official_res["sequence"] == standalone_res["sequence"]),
            "qstring_exact_match": bool(official_res["qstring"] == standalone_res["qstring"]),
            "moves_exact_match": bool(official_res["moves"] == standalone_res["moves"]),
            "sequence_compare": compare_pair(standalone_res["sequence"], official_res["sequence"]),
            "qstring_compare": compare_pair(standalone_res["qstring"], official_res["qstring"]),
        },
    }


def build_text_report(summary: Dict[str, object]) -> str:
    lines = [
        "[official_capture]",
        f"sequence_length: {summary['official_capture']['sequence_length']}",
        f"qstring_length: {summary['official_capture']['qstring_length']}",
        f"moves_length: {summary['official_capture']['moves_length']}",
        f"sequence_prefix: {summary['official_capture']['sequence_prefix']}",
        "",
        "[standalone_capture]",
        f"sequence_length: {summary['standalone_capture']['sequence_length']}",
        f"qstring_length: {summary['standalone_capture']['qstring_length']}",
        f"moves_length: {summary['standalone_capture']['moves_length']}",
        f"mods_count: {summary['standalone_capture']['mods_count']}",
        f"sequence_prefix: {summary['standalone_capture']['sequence_prefix']}",
        "",
        "[compare]",
        f"sequence_exact_match: {int(summary['compare']['sequence_exact_match'])}",
        f"qstring_exact_match: {int(summary['compare']['qstring_exact_match'])}",
        f"moves_exact_match: {int(summary['compare']['moves_exact_match'])}",
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
        args.read_ids_file = read_ids_path
        official_capture = run_basecaller_capture(args)
        standalone_capture = run_basecaller_mod_capture(args)
    finally:
        Path(read_ids_path).unlink(missing_ok=True)

    summary = build_summary(official_capture, standalone_capture)

    (output_dir / "official_cli_pre_writer_res.json").write_text(
        json.dumps(official_capture, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "standalone_cli_pre_writer_res.json").write_text(
        json.dumps(standalone_capture, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "cli_pre_writer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "cli_pre_writer_summary.txt").write_text(
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
