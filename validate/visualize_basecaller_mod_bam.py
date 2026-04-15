#!/usr/bin/env python3
"""
Sample and visualize modified-base calls from a basecaller_mod BAM.

Optional dorado-BAM comparison outputs:
- comparison_summary.json
- comparison_summary.txt
- per_read_comparison.tsv
- comparison_identity_hist.png (if matplotlib is available)

Visualization outputs:
- selected_reads.tsv
- one .txt visualization per selected read
- one .png visualization per selected read when matplotlib is available
"""

from __future__ import annotations

import csv
import json
import random
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pysam
from edlib import align as edlib_align

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


MM_SEGMENT_RE = re.compile(r"^([ACGTUN])([+-])([A-Za-z0-9]+)([.?]),(.*)$")
CIGAR_RE = re.compile(r"(\d+)([=XID])")


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


def normalize_seq(seq: str, convert_u_to_t: bool = True) -> str:
    text = str(seq or "").upper()
    return text.replace("U", "T") if convert_u_to_t else text


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
    include_non_primary: bool = False,
    keep_u: bool = False,
) -> Iterable[ReadRecord]:
    wanted = set(requested_ids) if requested_ids is not None else None

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam:
            if not include_non_primary and (read.is_secondary or read.is_supplementary):
                continue
            if read.query_sequence is None:
                continue
            read_id = str(read.query_name)
            if wanted is not None and read_id not in wanted:
                continue

            sequence = normalize_seq(read.query_sequence, convert_u_to_t=not keep_u)
            mm_tag = get_tag_or_none(read, "MM", "Mm")
            ml_tag = get_tag_or_none(read, "ML", "Ml")
            mods = parse_mm_ml_tags(sequence, mm_tag, ml_tag)
            if min_mod_prob > 0.0:
                mods = [
                    site for site in mods
                    if site.prob is None or site.prob >= min_mod_prob
                ]
            if not include_unmodified and not mods:
                continue

            yield ReadRecord(
                read_id=read_id,
                sequence=sequence,
                is_reverse=bool(read.is_reverse),
                reference_name=str(read.reference_name) if read.reference_name else "*",
                mods=mods,
            )


def load_sequence_map(
    bam_path: Path,
    *,
    requested_ids: Optional[Sequence[str]] = None,
    include_non_primary: bool = False,
    keep_u: bool = False,
) -> Dict[str, str]:
    wanted = set(requested_ids) if requested_ids is not None else None
    sequences: Dict[str, str] = {}
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam:
            if not include_non_primary and (read.is_secondary or read.is_supplementary):
                continue
            if read.query_sequence is None:
                continue
            read_id = str(read.query_name)
            if wanted is not None and read_id not in wanted:
                continue
            if read_id in sequences:
                continue
            sequences[read_id] = normalize_seq(read.query_sequence, convert_u_to_t=not keep_u)
    return sequences


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


def parse_cigar_counts(cigar: str) -> Dict[str, int]:
    counts = {"=": 0, "X": 0, "I": 0, "D": 0}
    for count, op in CIGAR_RE.findall(cigar or ""):
        counts[op] += int(count)
    return counts


def compare_pair(pred_seq: str, ref_seq: str) -> Dict[str, object]:
    result = edlib_align(pred_seq, ref_seq, mode="NW", task="path")
    counts = parse_cigar_counts(result.get("cigar", ""))
    aligned = counts["="] + counts["X"] + counts["I"] + counts["D"]
    return {
        "edit_distance": int(result.get("editDistance", -1)),
        "identity": float(counts["="] / aligned) if aligned else 0.0,
        "matches": int(counts["="]),
        "mismatches": int(counts["X"]),
        "insertions": int(counts["I"]),
        "deletions": int(counts["D"]),
        "cigar": str(result.get("cigar", "")),
    }


def build_comparison_rows(
    basecaller_records: Sequence[ReadRecord],
    dorado_sequences: Dict[str, str],
    *,
    allow_reverse: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    identities: List[float] = []
    edit_distances: List[int] = []
    same_length_count = 0
    missing = 0

    for record in basecaller_records:
        dorado_seq = dorado_sequences.get(record.read_id)
        if dorado_seq is None:
            missing += 1
            continue

        best = compare_pair(record.sequence, dorado_seq)
        orientation = "forward"
        if allow_reverse:
            reversed_compare = compare_pair(record.sequence[::-1], dorado_seq)
            if reversed_compare["identity"] > best["identity"]:
                best = reversed_compare
                orientation = "reversed"

        pred_len = len(record.sequence)
        dorado_len = len(dorado_seq)
        same_length = int(pred_len == dorado_len)
        identities.append(float(best["identity"]))
        edit_distances.append(int(best["edit_distance"]))
        same_length_count += same_length
        rows.append({
            "read_id": record.read_id,
            "orientation_used": orientation,
            "basecaller_mod_length": pred_len,
            "dorado_length": dorado_len,
            "length_delta": pred_len - dorado_len,
            "same_length": same_length,
            "identity": float(best["identity"]),
            "edit_distance": int(best["edit_distance"]),
            "matches": int(best["matches"]),
            "mismatches": int(best["mismatches"]),
            "insertions": int(best["insertions"]),
            "deletions": int(best["deletions"]),
            "basecaller_mod_prefix": record.sequence[:120],
            "dorado_prefix": dorado_seq[:120],
        })

    rows.sort(key=lambda row: (row["identity"], row["edit_distance"], row["read_id"]))
    summary = {
        "counts": {
            "num_basecaller_mod_reads": int(len(basecaller_records)),
            "num_dorado_reads_indexed": int(len(dorado_sequences)),
            "num_reads_compared": int(len(rows)),
            "num_reads_missing_in_dorado_bam": int(missing),
        },
        "metrics": {
            "mean_identity": float(sum(identities) / len(rows)) if rows else 0.0,
            "median_identity": float(median(identities)) if identities else 0.0,
            "mean_edit_distance": float(sum(edit_distances) / len(rows)) if rows else 0.0,
            "same_length_fraction": float(same_length_count / len(rows)) if rows else 0.0,
        },
    }
    return rows, summary


def save_comparison_plot(rows: Sequence[Dict[str, object]], output_path: Path) -> bool:
    if plt is None or not rows:
        return False

    identities = [float(row["identity"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(identities, bins=30, color="#4c78a8", edgecolor="black")
    ax.set_title("basecaller_mod vs dorado identity")
    ax.set_xlabel("Identity")
    ax.set_ylabel("Reads")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def build_comparison_summary_text(summary: Dict[str, object], basecaller_bam: Path, dorado_bam: Path) -> str:
    lines = [
        "[inputs]",
        f"basecaller_mod_bam: {basecaller_bam}",
        f"dorado_bam: {dorado_bam}",
        "",
        "[counts]",
        f"num_basecaller_mod_reads: {summary['counts']['num_basecaller_mod_reads']}",
        f"num_dorado_reads_indexed: {summary['counts']['num_dorado_reads_indexed']}",
        f"num_reads_compared: {summary['counts']['num_reads_compared']}",
        f"num_reads_missing_in_dorado_bam: {summary['counts']['num_reads_missing_in_dorado_bam']}",
        "",
        "[metrics]",
        f"mean_identity: {summary['metrics']['mean_identity']:.6f}",
        f"median_identity: {summary['metrics']['median_identity']:.6f}",
        f"mean_edit_distance: {summary['metrics']['mean_edit_distance']:.6f}",
        f"same_length_fraction: {summary['metrics']['same_length_fraction']:.6f}",
    ]
    return "\n".join(lines) + "\n"


def build_aligned_strings(query_seq: str, target_seq: str) -> Tuple[str, str, str]:
    result = edlib_align(query_seq, target_seq, mode="NW", task="path")
    cigar = result.get("cigar", "")
    query_index = 0
    target_index = 0
    aligned_query: List[str] = []
    aligned_target: List[str] = []
    aligned_match: List[str] = []

    for count_text, op in CIGAR_RE.findall(cigar):
        count = int(count_text)
        if op in ("=", "X"):
            for _ in range(count):
                q = query_seq[query_index]
                t = target_seq[target_index]
                aligned_query.append(q)
                aligned_target.append(t)
                aligned_match.append("|" if q == t else "x")
                query_index += 1
                target_index += 1
        elif op == "I":
            for _ in range(count):
                q = query_seq[query_index]
                aligned_query.append(q)
                aligned_target.append("-")
                aligned_match.append(" ")
                query_index += 1
        elif op == "D":
            for _ in range(count):
                t = target_seq[target_index]
                aligned_query.append("-")
                aligned_target.append(t)
                aligned_match.append(" ")
                target_index += 1

    return "".join(aligned_query), "".join(aligned_match), "".join(aligned_target)


def project_mods_to_aligned_query(aligned_query: str, mods: Sequence[ModSite]) -> Tuple[List[str], List[str]]:
    marker_chars = [" "] * len(aligned_query)
    label_chars = [" "] * len(aligned_query)
    query_index = -1
    mod_lookup = {site.query_index: site for site in mods}

    for aligned_index, char in enumerate(aligned_query):
        if char != "-":
            query_index += 1
        if char == "-":
            continue
        site = mod_lookup.get(query_index)
        if site is None:
            continue
        marker_chars[aligned_index] = "^"
        label_chars[aligned_index] = site.mod_code[:1] if site.mod_code else "*"
    return marker_chars, label_chars


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
        label_chars[offset] = site.mod_code[:1] if site.mod_code else "*"
        prob_text = "NA" if site.prob is None else f"{site.prob:.2f}"
        mods_in_window.append(f"{site.query_index}:{site.label}({prob_text})")

    lines = [
        f"[{start:>6}-{end - 1:>6}]",
        "".join(seq_chars),
        "".join(marker_chars),
        "".join(label_chars),
    ]
    lines.append("mods: " + ", ".join(mods_in_window) if mods_in_window else "mods: none")
    return lines


def render_text_visualization(
    record: ReadRecord,
    *,
    width: int,
    max_bases: int,
    dorado_sequence: Optional[str] = None,
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
    if dorado_sequence is None:
        for start in range(0, visible_end, width):
            end = min(start + width, visible_end)
            blocks.extend(build_window_lines(sequence, visible_mods, start=start, end=end))
            blocks.append("")
        return "\n".join(header + [""] + blocks).rstrip() + "\n"

    dorado_visible = dorado_sequence[:max_bases] if max_bases > 0 else dorado_sequence
    aligned_query, aligned_match, aligned_dorado = build_aligned_strings(sequence, dorado_visible)
    marker_chars, label_chars = project_mods_to_aligned_query(aligned_query, visible_mods)
    compare = compare_pair(sequence, dorado_visible)
    header.extend([
        f"dorado_sequence_length: {len(dorado_sequence)}",
        f"dorado_shown_bases: {len(dorado_visible)}",
        f"dorado_identity: {compare['identity']:.4f}",
        f"dorado_edit_distance: {compare['edit_distance']}",
    ])

    for start in range(0, len(aligned_query), width):
        end = min(start + width, len(aligned_query))
        blocks.append(f"[{start:>6}-{end - 1:>6}]")
        blocks.append("bonito : " + aligned_query[start:end])
        blocks.append("mods   : " + "".join(marker_chars[start:end]))
        blocks.append("labels : " + "".join(label_chars[start:end]))
        blocks.append("match  : " + aligned_match[start:end])
        blocks.append("dorado : " + aligned_dorado[start:end])
        window_mods = []
        query_index = -1
        for aligned_index, char in enumerate(aligned_query[start:end], start=start):
            if char != "-":
                query_index += 1
            if aligned_index >= len(marker_chars) or marker_chars[aligned_index] != "^":
                continue
            site = next((mod for mod in visible_mods if mod.query_index == query_index), None)
            if site is None:
                continue
            prob_text = "NA" if site.prob is None else f"{site.prob:.2f}"
            window_mods.append(f"{site.query_index}:{site.label}({prob_text})")
        blocks.append("mods: " + ", ".join(window_mods) if window_mods else "mods: none")
        blocks.append("")

    return "\n".join(header + [""] + blocks).rstrip() + "\n"


def render_png_visualization(
    record: ReadRecord,
    output_path: Path,
    *,
    width: int,
    max_bases: int,
    dorado_sequence: Optional[str] = None,
) -> bool:
    if plt is None:
        return False

    sequence = record.sequence[:max_bases] if max_bases > 0 else record.sequence
    visible_end = len(sequence)
    visible_mods = [site for site in record.mods if site.query_index < visible_end]

    color_map: Dict[str, str] = {
        "a": "#d55e00",
        "m": "#0072b2",
        "h": "#009e73",
    }

    if dorado_sequence is None:
        num_windows = max((visible_end + width - 1) // width, 1)
        fig_height = max(2.8 * num_windows, 3.2)
        fig, axes = plt.subplots(num_windows, 1, figsize=(18, fig_height), squeeze=False)
        axes_flat = axes.flatten().tolist()

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
                ax.text(x, 1.12, f"{site.label}\n{prob_text}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(
            f"{record.read_id} | len={len(record.sequence)} | shown={visible_end} | mods={len(record.mods)}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return True

    dorado_visible = dorado_sequence[:max_bases] if max_bases > 0 else dorado_sequence
    aligned_query, aligned_match, aligned_dorado = build_aligned_strings(sequence, dorado_visible)
    marker_chars, _ = project_mods_to_aligned_query(aligned_query, visible_mods)
    num_windows = max((len(aligned_query) + width - 1) // width, 1)
    fig_height = max(3.2 * num_windows, 3.8)
    fig, axes = plt.subplots(num_windows, 1, figsize=(18, fig_height), squeeze=False)
    axes_flat = axes.flatten().tolist()

    for window_index, ax in enumerate(axes_flat):
        start = window_index * width
        end = min(start + width, len(aligned_query))
        ax.set_xlim(-1, max(end - start, 1))
        ax.set_ylim(-1.2, 1.8)
        ax.axis("off")

        if start >= end:
            continue

        bonito_window = aligned_query[start:end]
        match_window = aligned_match[start:end]
        dorado_window = aligned_dorado[start:end]
        marker_window = marker_chars[start:end]

        for offset, base in enumerate(bonito_window):
            bbox = None
            if marker_window[offset] == "^":
                query_index = sum(1 for char in aligned_query[:start + offset + 1] if char != "-") - 1
                site = next((mod for mod in visible_mods if mod.query_index == query_index), None)
                if site is not None:
                    bbox = {
                        "boxstyle": "round,pad=0.18",
                        "facecolor": color_map.get(site.mod_code[:1].lower(), "#f0e442"),
                        "alpha": 0.35,
                        "edgecolor": "none",
                    }
            ax.text(offset, 0.55, base, ha="center", va="center", family="monospace", fontsize=11, bbox=bbox)
            ax.text(offset, -0.45, dorado_window[offset], ha="center", va="center", family="monospace", fontsize=11)
            ax.text(offset, 0.05, match_window[offset], ha="center", va="center", family="monospace", fontsize=10)

        ax.text(-0.9, 1.2, f"{start}-{end - 1}", ha="left", va="center", family="monospace", fontsize=10)
        ax.text(-0.9, 0.55, "bonito", ha="left", va="center", fontsize=9)
        ax.text(-0.9, 0.05, "match", ha="left", va="center", fontsize=9)
        ax.text(-0.9, -0.45, "dorado", ha="left", va="center", fontsize=9)

    compare = compare_pair(sequence, dorado_visible)
    fig.suptitle(
        f"{record.read_id} | bonito={len(sequence)} | dorado={len(dorado_visible)} | identity={compare['identity']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_selected_summary(
    records: Sequence[ReadRecord],
    output_path: Path,
    *,
    dorado_sequences: Optional[Dict[str, str]] = None,
) -> None:
    fieldnames = [
        "read_id",
        "sequence_length",
        "num_mod_sites",
        "max_mod_prob",
        "reference_name",
        "is_reverse",
    ]
    if dorado_sequences is not None:
        fieldnames.extend([
            "has_dorado_match",
            "dorado_length",
            "dorado_identity",
            "dorado_edit_distance",
        ])

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            row = {
                "read_id": record.read_id,
                "sequence_length": len(record.sequence),
                "num_mod_sites": len(record.mods),
                "max_mod_prob": max((site.prob or 0.0) for site in record.mods) if record.mods else 0.0,
                "reference_name": record.reference_name,
                "is_reverse": int(record.is_reverse),
            }
            if dorado_sequences is not None:
                dorado_seq = dorado_sequences.get(record.read_id)
                row["has_dorado_match"] = int(dorado_seq is not None)
                if dorado_seq is not None:
                    compare = compare_pair(record.sequence, dorado_seq)
                    row["dorado_length"] = len(dorado_seq)
                    row["dorado_identity"] = float(compare["identity"])
                    row["dorado_edit_distance"] = int(compare["edit_distance"])
                else:
                    row["dorado_length"] = 0
                    row["dorado_identity"] = 0.0
                    row["dorado_edit_distance"] = -1
            writer.writerow(row)


def main(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = load_requested_read_ids(args.read_ids)
    all_basecaller_records = list(
        iter_read_records(
            args.bam,
            requested_ids=None,
            include_unmodified=True,
            min_mod_prob=0.0,
            include_non_primary=args.include_non_primary,
            keep_u=args.keep_u,
        )
    )

    dorado_sequences: Optional[Dict[str, str]] = None
    if args.dorado_bam is not None:
        dorado_sequences = load_sequence_map(
            args.dorado_bam,
            requested_ids=None,
            include_non_primary=args.include_non_primary,
            keep_u=args.keep_u,
        )
        comparison_rows, comparison_summary = build_comparison_rows(
            all_basecaller_records,
            dorado_sequences,
            allow_reverse=args.allow_reverse,
        )
        with (output_dir / "per_read_comparison.tsv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "read_id",
                    "orientation_used",
                    "basecaller_mod_length",
                    "dorado_length",
                    "length_delta",
                    "same_length",
                    "identity",
                    "edit_distance",
                    "matches",
                    "mismatches",
                    "insertions",
                    "deletions",
                    "basecaller_mod_prefix",
                    "dorado_prefix",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(comparison_rows)

        comparison_text = build_comparison_summary_text(comparison_summary, args.bam.resolve(), args.dorado_bam.resolve())
        (output_dir / "comparison_summary.txt").write_text(comparison_text, encoding="utf-8")
        with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "basecaller_mod_bam": str(args.bam.resolve()),
                    "dorado_bam": str(args.dorado_bam.resolve()),
                    **comparison_summary,
                },
                fh,
                indent=2,
            )
        save_comparison_plot(comparison_rows, output_dir / "comparison_identity_hist.png")

    records = list(
        iter_read_records(
            args.bam,
            requested_ids=requested_ids,
            include_unmodified=args.include_unmodified,
            min_mod_prob=args.min_mod_prob,
            include_non_primary=args.include_non_primary,
            keep_u=args.keep_u,
        )
    )
    if not records:
        raise RuntimeError("No reads matched the selection criteria.")

    if dorado_sequences is not None:
        records = [record for record in records if record.read_id in dorado_sequences]
        if not records:
            raise RuntimeError("No selected reads overlap with the dorado BAM.")

    selected = select_records(
        records,
        max_reads=args.max_reads,
        strategy=args.strategy,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No reads were selected for visualization.")

    write_selected_summary(selected, output_dir / "selected_reads.tsv", dorado_sequences=dorado_sequences)

    png_written = 0
    for record in selected:
        stem = safe_filename(record.read_id)
        dorado_seq = dorado_sequences.get(record.read_id) if dorado_sequences is not None else None
        text = render_text_visualization(
            record,
            width=args.width,
            max_bases=args.max_bases,
            dorado_sequence=dorado_seq,
        )
        (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        if render_png_visualization(
            record,
            output_dir / f"{stem}.png",
            width=args.width,
            max_bases=args.max_bases,
            dorado_sequence=dorado_seq,
        ):
            png_written += 1

    print(f"selected_reads: {len(selected)}")
    print(f"text_outputs: {len(selected)}")
    print(f"png_outputs: {png_written}")
    if args.dorado_bam is not None:
        print(f"dorado_bam: {args.dorado_bam}")
        print("comparison_outputs: comparison_summary.json, comparison_summary.txt, per_read_comparison.tsv")
    print(f"output_dir: {output_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Sample reads from a basecaller_mod BAM and visualize sequence + modified sites.",
    )
    parser.add_argument("bam", type=Path, help="BAM produced by bonito basecaller_mod.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, help="Optional dorado BAM for overall sequence comparison and per-read visualization.")
    parser.add_argument("--max-reads", type=int, default=8)
    parser.add_argument("--strategy", choices=("top_mods", "random", "first"), default="top_mods")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--width", type=int, default=80, help="Bases per visualization row.")
    parser.add_argument("--max-bases", type=int, default=400, help="Maximum bases shown per read; 0 means show all.")
    parser.add_argument("--read-ids", type=Path, help="Optional file with read ids to visualize.")
    parser.add_argument("--include-unmodified", action="store_true", default=False)
    parser.add_argument("--min-mod-prob", type=float, default=0.0)
    parser.add_argument("--include-non-primary", action="store_true", default=False)
    parser.add_argument("--keep-u", action="store_true", default=False, help="Do not normalize U->T before comparison")
    parser.add_argument("--allow-reverse", action="store_true", default=True)
    parser.add_argument("--no-allow-reverse", dest="allow_reverse", action="store_false")
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
