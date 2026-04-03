#!/usr/bin/env python3
"""
Visualize basecalling + modification predictions on selected POD5 reads.

This script is intended to be used after validate/predict_mods_from_pod5.py.
It selects a small set of reads from mod_site_predictions.tsv, reruns inference
on those reads, and renders:
- raw/normalized current trace
- emitted base positions on the signal axis
- per-base predicted mod score
- basecall / mod label annotations
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from predict_mods_from_pod5 import (
    build_single_read_outputs,
    get_reads,
    init,
    iter_stitched_outputs,
    load_model,
    orient_sites_for_output,
    resolve_output_orientation,
)


def load_requested_read_ids(path: str | None) -> List[str] | None:
    if not path:
        return None
    read_ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            read_ids.append(text.split()[0])
    return read_ids


def select_read_ids(
    mod_sites_tsv: Path,
    max_reads: int,
    requested_read_ids: List[str] | None = None,
) -> List[str]:
    if requested_read_ids is not None:
        return requested_read_ids[:max_reads]

    per_read: Dict[str, Dict[str, float]] = {}
    with mod_sites_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            read_id = str(row["read_id"])
            pred_is_modified = int(float(row.get("pred_is_modified", 0)))
            score = float(row.get("score", 0.0))
            stats = per_read.setdefault(
                read_id,
                {
                    "num_sites": 0.0,
                    "num_modified": 0.0,
                    "max_score": 0.0,
                },
            )
            stats["num_sites"] += 1
            stats["num_modified"] += pred_is_modified
            stats["max_score"] = max(stats["max_score"], score)

    ranked = sorted(
        per_read.items(),
        key=lambda item: (
            -item[1]["num_modified"],
            -item[1]["max_score"],
            -item[1]["num_sites"],
            item[0],
        ),
    )
    return [read_id for read_id, _ in ranked[:max_reads]]


def downsample_trace(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points, dtype=np.int64)
    return x[idx], y[idx]


def build_site_summary(sites: List[Dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for site in sites:
        counts[str(site["global_pred_label"])] += 1
    return counts


def save_read_plot(
    read,
    sequence_output: str,
    sequence_signal_order: str,
    ordered_sites: List[Dict[str, object]],
    stride: int,
    output_path: Path,
    reverse_output: bool,
    max_annotations: int,
) -> None:
    signal = np.asarray(read.signal, dtype=np.float32)
    if signal.size == 0:
        return

    x = np.arange(signal.size, dtype=np.int64)
    plot_x, plot_signal = downsample_trace(x, signal, max_points=4000)
    signal_min = float(np.min(signal))
    signal_max = float(np.max(signal))
    signal_span = max(signal_max - signal_min, 1e-6)

    emit_positions = np.asarray([int(site["emit_position"]) for site in ordered_sites], dtype=np.int64)
    emit_signal_positions = np.clip(emit_positions * int(stride), 0, signal.size - 1) if emit_positions.size else emit_positions
    site_scores = np.asarray([float(site["score"]) for site in ordered_sites], dtype=np.float32)
    pred_labels = [str(site["global_pred_label"]) for site in ordered_sites]
    pred_bases = [str(site["base_label"]) for site in ordered_sites]

    fig, ax_signal = plt.subplots(figsize=(18, 8))
    ax_mod = ax_signal.twinx()

    ax_signal.plot(plot_x, plot_signal, linewidth=0.9, color="#4c566a", label="signal")
    ax_signal.set_xlabel("Signal sample index")
    ax_signal.set_ylabel("Normalized current")
    ax_mod.set_ylabel("Predicted class confidence")
    ax_mod.set_ylim(-0.08, 1.22)

    title_lines = [
        f"read_id={read.read_id}",
        f"signal_len={len(read.signal)} called_bases={len(sequence_signal_order)} output_is_rna={reverse_output}",
    ]
    label_counts = build_site_summary(ordered_sites)
    if label_counts:
        title_lines.append("pred_labels=" + ", ".join(f"{k}:{v}" for k, v in sorted(label_counts.items())))
    ax_signal.set_title("\n".join(title_lines))

    if emit_signal_positions.size:
        line_alpha = min(0.18, max(0.03, 18.0 / max(len(emit_signal_positions), 1)))
        for raw_x in emit_signal_positions:
            ax_signal.axvline(int(raw_x), color="#8fbcbb", alpha=line_alpha, linewidth=0.6, zorder=0)

        ax_mod.plot(
            emit_signal_positions,
            site_scores,
            color="#2ca02c",
            linewidth=1.1,
            alpha=0.9,
            label="pred site score",
        )
        scatter_colors = ["#d62728" if "canonical" not in label.lower() else "#1f77b4" for label in pred_labels]
        ax_mod.scatter(
            emit_signal_positions,
            site_scores,
            c=scatter_colors,
            s=24,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.25,
            label="predicted sites",
            zorder=4,
        )

        label_limit = min(len(ordered_sites), max_annotations)
        annotate_stride = max(label_limit // 20, 1) if label_limit else 1
        base_y = signal_max + (0.05 * signal_span)
        for idx in range(0, label_limit, annotate_stride):
            x_pos = int(emit_signal_positions[idx])
            base_text = pred_bases[idx]
            mod_text = pred_labels[idx]
            color = "#bf616a" if "canonical" not in mod_text.lower() else "#5e81ac"
            ax_signal.text(
                x_pos,
                base_y,
                base_text,
                fontsize=9,
                ha="center",
                va="bottom",
                rotation=90,
                color="#2e3440",
            )
            ax_mod.text(
                x_pos,
                min(float(site_scores[idx]) + 0.05, 1.04),
                mod_text,
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=90,
                color=color,
            )

    ax_signal.text(
        0.01,
        0.98,
        f"output_sequence_prefix={sequence_output[:80]}",
        transform=ax_signal.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        color="#2e3440",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#d8dee9"},
    )

    signal_handles, signal_labels = ax_signal.get_legend_handles_labels()
    mod_handles, mod_labels = ax_mod.get_legend_handles_labels()
    if signal_handles or mod_handles:
        ax_signal.legend(signal_handles + mod_handles, signal_labels + mod_labels, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_text_summary(summary: Dict[str, object]) -> str:
    lines = [
        "[inputs]",
        f"model_directory: {summary['model_directory']}",
        f"reads_directory: {summary['reads_directory']}",
        f"mod_sites_tsv: {summary['mod_sites_tsv']}",
        "",
        "[settings]",
        f"num_reads_requested: {summary['settings']['num_reads_requested']}",
        f"max_annotations: {summary['settings']['max_annotations']}",
        f"output_rna_orientation: {summary['settings']['output_rna_orientation']}",
        "",
        "[counts]",
        f"num_plots_written: {summary['counts']['num_plots_written']}",
        "",
        "[selected_reads]",
    ]
    lines.extend(summary["selected_reads"])
    return "\n".join(lines) + "\n"


def main(args):
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_read_ids = load_requested_read_ids(args.read_ids)
    selected_read_ids = select_read_ids(
        Path(args.mod_sites_tsv),
        max_reads=args.num_reads,
        requested_read_ids=requested_read_ids,
    )
    if not selected_read_ids:
        raise ValueError("No reads were selected for visualization.")

    init(args.seed, args.device, deterministic=not args.nondeterministic)
    use_half = str(args.device).startswith("cuda") and not args.no_half
    model = load_model(
        args.model_directory,
        args.device,
        weights=args.weights,
        half=use_half,
        chunksize=args.chunksize,
        batchsize=args.batchsize,
        overlap=args.overlap,
        use_koi=args.use_koi,
        compile=not args.no_compile,
    )
    reverse_output = resolve_output_orientation(args, model)

    class ReaderArgs:
        pass

    reader_args = ReaderArgs()
    reader_args.reads_directory = args.reads_directory
    reader_args.model_directory = args.model_directory
    reader_args.recursive = args.recursive
    reader_args.read_ids = str(out_dir / "selected_read_ids.txt")
    reader_args.skip = False
    reader_args.no_trim = args.no_trim
    reader_args.max_reads = 0

    with open(reader_args.read_ids, "w", encoding="utf-8") as fh:
        for read_id in selected_read_ids:
            fh.write(read_id + "\n")

    reads, _ = get_reads(reader_args, model)
    device = str(next(model.parameters()).device)
    written = []

    for read, stitched_outputs in tqdm(
        iter_stitched_outputs(
            model,
            reads,
            batchsize=model.config["basecaller"]["batchsize"],
            chunksize=model.config["basecaller"]["chunksize"],
            overlap=model.config["basecaller"]["overlap"],
        ),
        total=len(selected_read_ids),
        desc="visualizing",
        ascii=True,
        ncols=100,
    ):
        single_outputs = build_single_read_outputs(stitched_outputs, device=device)
        sequence_signal_order = model.decode_batch(single_outputs)[0]
        site_prediction_record = model.predict_mods(single_outputs, mod_threshold=args.mod_threshold)[0]
        sequence_output = sequence_signal_order[::-1] if reverse_output else sequence_signal_order
        ordered_sites = orient_sites_for_output(
            site_prediction_record.get("sites", []),
            sequence_length=len(sequence_signal_order),
            reverse_output=reverse_output,
        )

        png_path = out_dir / f"{read.read_id}.png"
        save_read_plot(
            read=read,
            sequence_output=sequence_output,
            sequence_signal_order=sequence_signal_order,
            ordered_sites=ordered_sites,
            stride=int(model.stride),
            output_path=png_path,
            reverse_output=reverse_output,
            max_annotations=args.max_annotations,
        )
        written.append(png_path.name)

    summary = {
        "model_directory": str(Path(args.model_directory).resolve()),
        "reads_directory": str(Path(args.reads_directory).resolve()),
        "mod_sites_tsv": str(Path(args.mod_sites_tsv).resolve()),
        "settings": {
            "num_reads_requested": int(args.num_reads),
            "max_annotations": int(args.max_annotations),
            "output_rna_orientation": bool(reverse_output),
        },
        "counts": {
            "num_plots_written": int(len(written)),
        },
        "selected_reads": selected_read_ids,
        "artifacts": written,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    (out_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")
    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {out_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("reads_directory", type=Path)
    parser.add_argument("--mod-sites-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--read-ids", help="Optional file listing read IDs to visualize, one per line")
    parser.add_argument("--num-reads", default=10, type=int)
    parser.add_argument("--max-annotations", default=40, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=None, type=int)
    parser.add_argument("--mod-threshold", default=0.5, type=float)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--use-koi", action="store_true", default=False)

    orientation_group = parser.add_mutually_exclusive_group()
    orientation_group.add_argument("--rna", dest="rna", action="store_true", default=None)
    orientation_group.add_argument("--no-rna", dest="rna", action="store_false")
    parser.set_defaults(rna=None)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
