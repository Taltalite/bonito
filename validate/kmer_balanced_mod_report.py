#!/usr/bin/env python3
"""
Build a k-mer balanced modification report from evaluate_train_mod outputs.

Inputs:
- mod_site_examples.tsv (from validate/evaluate_train_mod.py)
- references.npy
- reference_lengths.npy

Outputs:
- balanced_sites.tsv
- kmer_stats.tsv
- summary.json
- summary.txt
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def binary_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    y_true = y_true.astype(np.int64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return None, None

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    tps = np.cumsum(sorted_true == 1)
    fps = np.cumsum(sorted_true == 0)
    threshold_idx = np.where(np.diff(sorted_scores))[0]
    threshold_idx = np.r_[threshold_idx, len(sorted_scores) - 1]

    tps = tps[threshold_idx]
    fps = fps[threshold_idx]

    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    roc_auc = float(np.trapz(tpr, fpr))

    precision = np.r_[1.0, tps / np.maximum(tps + fps, 1)]
    recall = np.r_[0.0, tps / positives]
    pr_auc = float(np.trapz(precision, recall))
    return roc_auc, pr_auc


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    roc_auc, pr_auc = binary_auc_metrics(y_true, y_prob)

    return {
        "num_sites": int(y_true.size),
        "num_positive": int(np.sum(y_true == 1)),
        "num_negative": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": safe_div(tp + tn, y_true.size),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, (2 * tp) + fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def parse_binary_label(label: str, canonical_re: re.Pattern, positive_re: re.Pattern) -> Optional[int]:
    text = str(label)
    if canonical_re.search(text):
        return 0
    if positive_re.search(text):
        return 1
    return None


def decode_window(ref_row: np.ndarray, center: int, kmer_size: int, alphabet: str) -> Optional[str]:
    half = kmer_size // 2
    start = center - half
    end = center + half
    if start < 0 or end >= ref_row.shape[0]:
        return None

    chars = []
    for idx in range(start, end + 1):
        token = int(ref_row[idx])
        if token <= 0 or token >= len(alphabet):
            return None
        chars.append(alphabet[token])
    return "".join(chars)


def infer_positive_probability(df: pd.DataFrame, positive_re: re.Pattern) -> np.ndarray:
    if "pred_mod_label" not in df.columns:
        raise ValueError("mod_site_examples.tsv is missing 'pred_mod_label'; cannot infer positive-class probability.")
    if "score" not in df.columns:
        raise ValueError("mod_site_examples.tsv is missing 'score'.")

    pred_is_positive = df["pred_mod_label"].astype(str).str.contains(positive_re, na=False)
    conf = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # score is confidence of predicted class.
    # convert into P(positive):
    # pred=positive -> p_pos = score
    # pred=canonical -> p_pos = 1-score
    p_pos = np.where(pred_is_positive.to_numpy(), conf.to_numpy(), 1.0 - conf.to_numpy())
    return p_pos.astype(np.float32)


def sample_balanced_per_kmer(df: pd.DataFrame, max_per_class_per_kmer: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    sampled_indices: List[int] = []
    rows = []
    for kmer, group in df.groupby("kmer"):
        g0 = group[group["y_true"] == 0]
        g1 = group[group["y_true"] == 1]
        if g0.empty or g1.empty:
            rows.append({
                "kmer": kmer,
                "num_canonical": int(len(g0)),
                "num_modified": int(len(g1)),
                "num_sampled_per_class": 0,
                "num_sampled_total": 0,
            })
            continue

        n = min(len(g0), len(g1))
        if max_per_class_per_kmer > 0:
            n = min(n, max_per_class_per_kmer)

        idx0 = rng.choice(g0.index.to_numpy(), size=n, replace=False)
        idx1 = rng.choice(g1.index.to_numpy(), size=n, replace=False)
        sampled_indices.extend(idx0.tolist())
        sampled_indices.extend(idx1.tolist())

        rows.append({
            "kmer": kmer,
            "num_canonical": int(len(g0)),
            "num_modified": int(len(g1)),
            "num_sampled_per_class": int(n),
            "num_sampled_total": int(2 * n),
        })

    sampled = df.loc[sampled_indices].copy() if sampled_indices else df.iloc[0:0].copy()
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True) if not sampled.empty else sampled
    kmer_stats = pd.DataFrame(rows).sort_values("kmer").reset_index(drop=True) if rows else pd.DataFrame()
    return sampled, kmer_stats


def build_text_summary(summary: Dict[str, object]) -> str:
    metrics = summary["balanced_metrics"]
    lines = [
        "[inputs]",
        f"mod_sites_tsv: {summary['mod_sites_tsv']}",
        f"references_npy: {summary['references_npy']}",
        f"reference_lengths_npy: {summary['reference_lengths_npy']}",
        "",
        "[settings]",
        f"head_name: {summary['settings']['head_name']}",
        f"kmer_size: {summary['settings']['kmer_size']}",
        f"max_per_class_per_kmer: {summary['settings']['max_per_class_per_kmer']}",
        f"threshold: {summary['settings']['threshold']}",
        "",
        "[counts]",
        f"num_input_sites: {summary['counts']['num_input_sites']}",
        f"num_sites_with_kmer: {summary['counts']['num_sites_with_kmer']}",
        f"num_kmers_with_both_classes: {summary['counts']['num_kmers_with_both_classes']}",
        f"num_balanced_sites: {summary['counts']['num_balanced_sites']}",
        "",
        "[balanced_metrics]",
        f"accuracy: {metrics['accuracy']:.6f}",
        f"precision: {metrics['precision']:.6f}",
        f"recall: {metrics['recall']:.6f}",
        f"f1: {metrics['f1']:.6f}",
        f"roc_auc: {metrics['roc_auc']}",
        f"pr_auc: {metrics['pr_auc']}",
    ]
    return "\n".join(lines) + "\n"


def main(args):
    if args.kmer_size % 2 == 0 or args.kmer_size < 3:
        raise ValueError("--kmer-size must be an odd integer >= 3")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod_df = pd.read_csv(args.mod_sites_tsv, sep="\t")
    if "head_name" not in mod_df.columns:
        raise ValueError("mod_site_examples.tsv is missing 'head_name' column")
    if "target_pos" not in mod_df.columns or "chunk_index" not in mod_df.columns:
        raise ValueError("mod_site_examples.tsv must include 'chunk_index' and 'target_pos' columns")
    if "true_mod_label" not in mod_df.columns:
        raise ValueError("mod_site_examples.tsv is missing 'true_mod_label'")

    head_df = mod_df[mod_df["head_name"].astype(str) == str(args.head_name)].copy()
    if head_df.empty:
        raise ValueError(f"No rows found for --head-name={args.head_name!r}")

    canonical_re = re.compile(args.canonical_label_regex)
    positive_re = re.compile(args.positive_label_regex)

    y_true = head_df["true_mod_label"].apply(lambda x: parse_binary_label(str(x), canonical_re, positive_re))
    head_df["y_true"] = y_true
    head_df = head_df[head_df["y_true"].notna()].copy()
    head_df["y_true"] = head_df["y_true"].astype(np.int64)

    if head_df.empty:
        raise ValueError("No binary labels left after regex mapping. Check label regex arguments.")

    head_df["p_pos"] = infer_positive_probability(head_df, positive_re)

    refs = np.load(args.references_npy, mmap_mode="r")
    lengths = np.load(args.reference_lengths_npy, mmap_mode="r")

    if refs.ndim != 2 or lengths.ndim != 1:
        raise ValueError("references.npy must be 2D and reference_lengths.npy must be 1D")

    alphabet = str(args.alphabet)
    kmers: List[Optional[str]] = []
    keep_mask: List[bool] = []

    for _, row in head_df.iterrows():
        chunk_index = int(row["chunk_index"])
        target_pos = int(row["target_pos"])

        if chunk_index < 0 or chunk_index >= refs.shape[0]:
            kmers.append(None)
            keep_mask.append(False)
            continue

        max_len = int(lengths[chunk_index])
        if target_pos < 0 or target_pos >= max_len:
            kmers.append(None)
            keep_mask.append(False)
            continue

        kmer = decode_window(refs[chunk_index], target_pos, args.kmer_size, alphabet=alphabet)
        if kmer is None:
            kmers.append(None)
            keep_mask.append(False)
            continue

        center = kmer[args.kmer_size // 2]
        if center.upper() != args.center_base.upper():
            kmers.append(None)
            keep_mask.append(False)
            continue

        kmers.append(kmer)
        keep_mask.append(True)

    head_df["kmer"] = kmers
    work_df = head_df[np.asarray(keep_mask, dtype=bool)].copy()

    if work_df.empty:
        raise ValueError("No sites left after k-mer extraction/filtering.")

    sampled_df, kmer_stats = sample_balanced_per_kmer(
        work_df,
        max_per_class_per_kmer=args.max_per_class_per_kmer,
        seed=args.seed,
    )
    if sampled_df.empty:
        raise ValueError("No balanced samples could be drawn (no k-mer has both classes).")

    metrics = compute_binary_metrics(
        sampled_df["y_true"].to_numpy(np.int64),
        sampled_df["p_pos"].to_numpy(np.float32),
        threshold=args.threshold,
    )

    usable_kmers = kmer_stats[kmer_stats["num_sampled_per_class"] > 0] if not kmer_stats.empty else pd.DataFrame()
    summary = {
        "mod_sites_tsv": str(Path(args.mod_sites_tsv).resolve()),
        "references_npy": str(Path(args.references_npy).resolve()),
        "reference_lengths_npy": str(Path(args.reference_lengths_npy).resolve()),
        "settings": {
            "head_name": args.head_name,
            "kmer_size": int(args.kmer_size),
            "center_base": args.center_base,
            "max_per_class_per_kmer": int(args.max_per_class_per_kmer),
            "threshold": float(args.threshold),
            "seed": int(args.seed),
            "alphabet": args.alphabet,
            "canonical_label_regex": args.canonical_label_regex,
            "positive_label_regex": args.positive_label_regex,
        },
        "counts": {
            "num_input_sites": int(len(mod_df)),
            "num_head_sites": int(len(head_df)),
            "num_sites_with_kmer": int(len(work_df)),
            "num_kmers_total": int(work_df["kmer"].nunique()),
            "num_kmers_with_both_classes": int(len(usable_kmers)),
            "num_balanced_sites": int(len(sampled_df)),
        },
        "balanced_metrics": metrics,
    }

    sampled_df.to_csv(out_dir / "balanced_sites.tsv", sep="\t", index=False)
    kmer_stats.to_csv(out_dir / "kmer_stats.tsv", sep="\t", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (out_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")

    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {out_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("--mod-sites-tsv", type=Path, required=True, help="Path to mod_site_examples.tsv")
    parser.add_argument("--references-npy", type=Path, required=True)
    parser.add_argument("--reference-lengths-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--head-name", type=str, default="A", help="Head name to evaluate, e.g. A")
    parser.add_argument("--kmer-size", type=int, default=5)
    parser.add_argument("--center-base", type=str, default="A")
    parser.add_argument("--max-per-class-per-kmer", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9)

    parser.add_argument("--alphabet", type=str, default="NACGT", help="Token-id to base lookup string")
    parser.add_argument("--canonical-label-regex", type=str, default=r"canonical", help="Regex mapping true label -> class 0")
    parser.add_argument("--positive-label-regex", type=str, default=r"m6a|modified", help="Regex mapping true label -> class 1")
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
