#!/usr/bin/env python3
"""
Dump the single-read result dict immediately before Writer for:
1. official basecaller path
2. standalone basecaller_mod path
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
from edlib import align as edlib_align

from bonito.crf.basecall import basecall as official_crf_basecall
from bonito.reader import Reader
from bonito.transformer.multihead_basecall import basecall as standalone_basecall
from bonito.util import init, load_model, load_symbol


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


def norm_params_from_model(model):
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


def summarize_read(read) -> Dict[str, object]:
    return {
        "read_id": str(read.read_id),
        "num_samples": int(read.num_samples),
        "trimmed_samples": int(read.trimmed_samples),
        "signal_length": int(len(read.signal)),
        "shift": float(read.shift),
        "scale": float(read.scale),
        "scaling_strategy": str(read.scaling_strategy),
    }


def get_official_basecall_fn(model_dir: str):
    try:
        return load_symbol(model_dir, "basecall")
    except AttributeError:
        return official_crf_basecall


def build_summary(official_read, standalone_read, official_res: Dict[str, object], standalone_res: Dict[str, object]) -> Dict[str, object]:
    seq_compare = compare_pair(standalone_res["sequence"], official_res["sequence"])
    qstring_compare = compare_pair(standalone_res["qstring"], official_res["qstring"])
    return {
        "official_read": summarize_read(official_read),
        "standalone_read": summarize_read(standalone_read),
        "official_result": {
            "sequence_length": int(official_res["sequence_length"]),
            "qstring_length": int(official_res["qstring_length"]),
            "moves_length": official_res["moves_length"],
            "mods_count": int(official_res["mods_count"]),
            "sequence_prefix": official_res["sequence"][:200],
            "qstring_prefix": official_res["qstring"][:200],
            "moves_prefix": official_res["moves"][:80] if official_res["moves"] is not None else None,
        },
        "standalone_result": {
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
            "sequence_compare": seq_compare,
            "qstring_compare": qstring_compare,
        },
    }


def build_text_report(summary: Dict[str, object]) -> str:
    lines = [
        "[official_read]",
        f"signal_length: {summary['official_read']['signal_length']}",
        f"trimmed_samples: {summary['official_read']['trimmed_samples']}",
        "",
        "[standalone_read]",
        f"signal_length: {summary['standalone_read']['signal_length']}",
        f"trimmed_samples: {summary['standalone_read']['trimmed_samples']}",
        "",
        "[official_result]",
        f"sequence_length: {summary['official_result']['sequence_length']}",
        f"qstring_length: {summary['official_result']['qstring_length']}",
        f"moves_length: {summary['official_result']['moves_length']}",
        f"sequence_prefix: {summary['official_result']['sequence_prefix']}",
        "",
        "[standalone_result]",
        f"sequence_length: {summary['standalone_result']['sequence_length']}",
        f"qstring_length: {summary['standalone_result']['qstring_length']}",
        f"moves_length: {summary['standalone_result']['moves_length']}",
        f"mods_count: {summary['standalone_result']['mods_count']}",
        f"sequence_prefix: {summary['standalone_result']['sequence_prefix']}",
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

    official_basecall_fn = get_official_basecall_fn(args.official_model)
    official_res = sanitize_res(
        list(
            official_basecall_fn(
                official_model,
                [official_read],
                reverse=False,
                rna=args.rna,
                batchsize=official_model.config["basecaller"]["batchsize"],
                chunksize=official_model.config["basecaller"]["chunksize"],
                overlap=official_model.config["basecaller"]["overlap"],
            )
        )[0][1]
    )
    standalone_res = sanitize_res(
        list(
            standalone_basecall(
                standalone_model,
                [standalone_read],
                reverse=False,
                rna=args.rna,
                batchsize=standalone_model.config["basecaller"]["batchsize"],
                chunksize=standalone_model.config["basecaller"]["chunksize"],
                overlap=standalone_model.config["basecaller"]["overlap"],
                mod_threshold=args.mod_threshold,
            )
        )[0][1]
    )

    summary = build_summary(official_read, standalone_read, official_res, standalone_res)

    (output_dir / "official_pre_writer_res.json").write_text(
        json.dumps(official_res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "standalone_pre_writer_res.json").write_text(
        json.dumps(standalone_res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "pre_writer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "pre_writer_summary.txt").write_text(
        build_text_report(summary),
        encoding="utf-8",
    )

    print(f"read_id: {args.read_id}")
    print(f"output_dir: {output_dir}")
    print(f"sequence_exact_match: {int(summary['compare']['sequence_exact_match'])}")
    print(f"sequence_identity: {summary['compare']['sequence_compare']['identity']:.6f}")
    print(f"official_length: {summary['official_result']['sequence_length']}")
    print(f"standalone_length: {summary['standalone_result']['sequence_length']}")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("official_model", help="Official basecaller model directory.")
    parser.add_argument("standalone_model", help="Standalone mod-head model directory.")
    parser.add_argument("reads_directory", help="pod5 input directory.")
    parser.add_argument("--read-id", required=True, help="Single read id to dump.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/TXT outputs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--no-half", action="store_true", default=False, help="Load models in fp32.")
    parser.add_argument("--mod-threshold", default=0.5, type=float)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
