#!/usr/bin/env python3
"""
Sample and visualize modified-base calls from a basecaller_mod BAM.

Outputs:
- selected_reads.tsv
- one .txt visualization per selected read
- one .png visualization per selected read when matplotlib is available
"""

from __future__ import annotations

import csv
import random
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pysam

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


MM_SEGMENT_RE = re.compile(r"^([ACGTUN])([+-])([A-Za-z0-9]+)([.?]),(.*)$")


@dataclass
class ModSite:
    query_index: int
    canonical_base: str
    mod_code: str
    strand: str
    implicit_skips: str
    prob_255: Optional[int]

    @property
    def prob(self) -> Optional[float]:
        if self.prob_255 is None:
            return None
        return float(self.prob_255) / 255.0

    @property
    def label(self) -> str:
        return f"{self.canonical_base}+{self.mod_code}"


@dataclass
class ReadRecord:
    read_id: str
    sequence: str
    is_reverse: bool
    reference_name: str
    mods: List[ModSite]


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned or "read"


def load_requested_read_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    read_ids: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                read_ids.append(text.split()[0])
    return read_ids


def get_tag_or_none(read: pysam.AlignedSegment, primary: str, secondary: str):
    if read.has_tag(primary):
        return read.get_tag(primary)
    if read.has_tag(secondary):
        return read.get_tag(secondary)
    return None


def parse_mm_ml_tags(sequence: str, mm_tag: Optional[str], ml_tag) -> List[ModSite]:
    if not sequence or not mm_tag:
        return []

    sequence_upper = sequence.upper()
    ml_values = list(ml_tag) if ml_tag is not None else []
    ml_index = 0
    mods: List[ModSite] = []

    for raw_segment in mm_tag.split(";"):
        segment = raw_segment.strip()
        if not segment:
            continue

        match = MM_SEGMENT_RE.match(segment)
        if match is None:
            continue

        canonical_base, strand, mod_code, implicit_skips, deltas_blob = match.groups()
        deltas = [item.strip() for item in deltas_blob.split(",") if item.strip() != ""]
        if not deltas:
            continue

        base_positions = [
            idx for idx, base in enumerate(sequence_upper)
            if base == canonical_base.upper()
        ]
        if not base_positions:
            continue

        occurrence = -1
        for delta_text in deltas:
            delta = int(delta_text)
            occurrence = delta if occurrence < 0 else occurrence + delta + 1
            if occurrence < 0 or occurrence >= len(base_positions):
                break
            prob_255 = None
            if ml_index < len(ml_values):
                prob_255 = int(ml_values[ml_index])
                ml_index += 1
            mods.append(
                ModSite(
                    query_index=int(base_positions[occurrence]),
                    canonical_base=canonical_base.upper(),
                    mod_code=mod_code,
                    strand=strand,
                    implicit_skips=implicit_skips,
                    prob_255=prob_255,
                )
            )

    return sorted(mods, key=lambda item: item.query_index)


def iter_read_records(
    bam_path: Path,
    *,
    requested_ids: Optional[Sequence[str]] = None,
    include_unmodified: bool = False,
    min_mod_prob: float = 0.0,
) -> Iterable[ReadRecord]:
    wanted = set(requested_ids) if requested_ids is not None else None

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam:
            if read.is_secondary or read.is_supplementary:
                continue
            if read.query_sequence is None:
                continue
            read_id = str(read.query_name)
            if wanted is not None and read_id not in wanted:
                continue

            mm_tag = get_tag_or_none(read, "MM", "Mm")
            ml_tag = get_tag_or_none(read, "ML", "Ml")
            mods = parse_mm_ml_tags(read.query_sequence, mm_tag, ml_tag)
            if min_mod_prob > 0.0:
                mods = [
                    site for site in mods
                    if site.prob is None or site.prob >= min_mod_prob
                ]
            if not include_unmodified and not mods:
                continue

            yield ReadRecord(
                read_id=read_id,
                sequence=str(read.query_sequence),
                is_reverse=bool(read.is_reverse),
                reference_name=str(read.reference_name) if read.reference_name else "*",
                mods=mods,
            )


def select_records(
    records: Sequence[ReadRecord],
    *,
    max_reads: int,
    strategy: str,
    seed: int,
) -> List[ReadRecord]:
    if max_reads <= 0:
        return []
    if strategy == "first":
        return list(records[:max_reads])
    if strategy == "top_mods":
        ranked = sorted(
            records,
            key=lambda item: (
                -len(item.mods),
                -(max((site.prob or 0.0) for site in item.mods) if item.mods else 0.0),
                -len(item.sequence),
                item.read_id,
            ),
        )
        return ranked[:max_reads]
    if strategy == "random":
        rng = random.Random(seed)
        pool = list(records)
        rng.shuffle(pool)
        return pool[:max_reads]
    raise ValueError(f"Unsupported strategy: {strategy}")


def build_window_lines(
    sequence: str,
    mods: Sequence[ModSite],
    *,
    start: int,
    end: int,
) -> List[str]:
    window_len = end - start
    seq_chars = list(sequence[start:end])
    marker_chars = [" "] * window_len
    label_chars = [" "] * window_len
    mods_in_window: List[str] = []

    for site in mods:
        if site.query_index < start or site.query_index >= end:
            continue
        offset = site.query_index - start
        marker_chars[offset] = "^"
        label_char = site.mod_code[0] if site.mod_code else "*"
        label_chars[offset] = label_char
        prob_text = "NA" if site.prob is None else f"{site.prob:.2f}"
        mods_in_window.append(f"{site.query_index}:{site.label}({prob_text})")

    lines = [
        f"[{start:>6}-{end - 1:>6}]",
        "".join(seq_chars),
        "".join(marker_chars),
        "".join(label_chars),
    ]
    if mods_in_window:
        lines.append("mods: " + ", ".join(mods_in_window))
    else:
        lines.append("mods: none")
    return lines


def render_text_visualization(
    record: ReadRecord,
    *,
    width: int,
    max_bases: int,
) -> str:
    sequence = record.sequence[:max_bases] if max_bases > 0 else record.sequence
    visible_end = len(sequence)
    visible_mods = [site for site in record.mods if site.query_index < visible_end]
    hidden_mods = max(len(record.mods) - len(visible_mods), 0)

    header = [
        f"read_id: {record.read_id}",
        f"reference_name: {record.reference_name}",
        f"is_reverse: {int(record.is_reverse)}",
        f"sequence_length: {len(record.sequence)}",
        f"shown_bases: {visible_end}",
        f"num_mod_sites: {len(record.mods)}",
    ]
    if hidden_mods:
        header.append(f"hidden_mod_sites_after_truncation: {hidden_mods}")

    blocks: List[str] = []
    for start in range(0, visible_end, width):
        end = min(start + width, visible_end)
        blocks.extend(build_window_lines(sequence, visible_mods, start=start, end=end))
        blocks.append("")

    return "\n".join(header + [""] + blocks).rstrip() + "\n"


def render_png_visualization(
    record: ReadRecord,
    output_path: Path,
    *,
    width: int,
    max_bases: int,
) -> bool:
    if plt is None:
        return False

    sequence = record.sequence[:max_bases] if max_bases > 0 else record.sequence
    visible_end = len(sequence)
    visible_mods = [site for site in record.mods if site.query_index < visible_end]
    num_windows = max((visible_end + width - 1) // width, 1)

    fig_height = max(2.8 * num_windows, 3.2)
    fig, axes = plt.subplots(num_windows, 1, figsize=(18, fig_height), squeeze=False)
    axes_flat = axes.flatten().tolist()

    color_map: Dict[str, str] = {
        "a": "#d55e00",
        "m": "#0072b2",
        "h": "#009e73",
    }

    for window_index, ax in enumerate(axes_flat):
        start = window_index * width
        end = min(start + width, visible_end)
        ax.set_xlim(-1, max(end - start, 1))
        ax.set_ylim(-0.9, 1.7)
        ax.axis("off")

        if start >= end:
            continue

        window_seq = sequence[start:end]
        window_mods = [site for site in visible_mods if start <= site.query_index < end]
        mod_lookup = {site.query_index - start: site for site in window_mods}

        for offset, base in enumerate(window_seq):
            site = mod_lookup.get(offset)
            bbox = None
            if site is not None:
                bbox = {
                    "boxstyle": "round,pad=0.18",
                    "facecolor": color_map.get(site.mod_code[:1].lower(), "#f0e442"),
                    "alpha": 0.35,
                    "edgecolor": "none",
                }
            ax.text(offset, 0.0, base, ha="center", va="center", family="monospace", fontsize=12, bbox=bbox)

        ax.text(-0.9, 1.25, f"{start}-{end - 1}", ha="left", va="center", family="monospace", fontsize=10)

        for site in window_mods:
            x = site.query_index - start
            color = color_map.get(site.mod_code[:1].lower(), "#cc79a7")
            ax.scatter([x], [0.85], s=42, color=color, zorder=3)
            prob_text = "NA" if site.prob is None else f"{site.prob:.2f}"
            ax.text(
                x,
                1.12,
                f"{site.label}\n{prob_text}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    fig.suptitle(
        f"{record.read_id} | len={len(record.sequence)} | shown={visible_end} | mods={len(record.mods)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_selected_summary(records: Sequence[ReadRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "read_id",
                "sequence_length",
                "num_mod_sites",
                "max_mod_prob",
                "reference_name",
                "is_reverse",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "read_id": record.read_id,
                "sequence_length": len(record.sequence),
                "num_mod_sites": len(record.mods),
                "max_mod_prob": max((site.prob or 0.0) for site in record.mods) if record.mods else 0.0,
                "reference_name": record.reference_name,
                "is_reverse": int(record.is_reverse),
            })


def main(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = load_requested_read_ids(args.read_ids)
    records = list(
        iter_read_records(
            args.bam,
            requested_ids=requested_ids,
            include_unmodified=args.include_unmodified,
            min_mod_prob=args.min_mod_prob,
        )
    )
    if not records:
        raise RuntimeError("No reads matched the selection criteria.")

    selected = select_records(
        records,
        max_reads=args.max_reads,
        strategy=args.strategy,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No reads were selected for visualization.")

    write_selected_summary(selected, output_dir / "selected_reads.tsv")

    png_written = 0
    for record in selected:
        stem = safe_filename(record.read_id)
        text = render_text_visualization(
            record,
            width=args.width,
            max_bases=args.max_bases,
        )
        (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        if render_png_visualization(
            record,
            output_dir / f"{stem}.png",
            width=args.width,
            max_bases=args.max_bases,
        ):
            png_written += 1

    print(f"selected_reads: {len(selected)}")
    print(f"text_outputs: {len(selected)}")
    print(f"png_outputs: {png_written}")
    print(f"output_dir: {output_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Sample reads from a basecaller_mod BAM and visualize sequence + modified sites.",
    )
    parser.add_argument("bam", type=Path, help="BAM produced by bonito basecaller_mod.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-reads", type=int, default=8)
    parser.add_argument("--strategy", choices=("top_mods", "random", "first"), default="top_mods")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--width", type=int, default=80, help="Bases per visualization row.")
    parser.add_argument("--max-bases", type=int, default=400, help="Maximum bases shown per read; 0 means show all.")
    parser.add_argument("--read-ids", type=Path, help="Optional file with read ids to visualize.")
    parser.add_argument("--include-unmodified", action="store_true", default=False)
    parser.add_argument("--min-mod-prob", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
