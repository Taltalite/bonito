#!/usr/bin/env python3
"""
Evaluate a train_mod checkpoint on basecalling and modification targets.

Outputs:
- summary.json
- summary.txt
- base_alignments.tsv
- sequence_examples.tsv
- mod_site_examples.tsv
- mod_alignment_summary.tsv
- predicted_base_examples.tsv
- PNG plots when matplotlib is available
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import parasail
import torch
import torch.amp as amp
from tqdm import tqdm

from bonito.data import ComputeSettings, DataSettings, ModelSetup, load_data, load_mod_data
from bonito.util import accuracy, decode_ref, init, load_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class AlignResult:
    accuracy_pct: float = 0.0
    num_correct: int = 0
    num_mismatches: int = 0
    num_insertions: int = 0
    num_deletions: int = 0
    ref_len: int = 0
    seq_len: int = 0
    align_ref_start: int = 0
    align_ref_end: int = 0
    align_seq_start: int = 0
    align_seq_end: int = 0


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def align(ref: str, seq: str) -> AlignResult:
    if not ref:
        return AlignResult(ref_len=0, seq_len=len(seq), num_insertions=len(seq))
    if not seq:
        return AlignResult(ref_len=len(ref), seq_len=0)

    try:
        res = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    except Exception:
        return AlignResult(ref_len=len(ref), seq_len=len(seq))
    cigar = res.cigar.decode.decode()
    counts = defaultdict(int)
    for count, op in re.findall(r"(\d+)([A-Z\W])", cigar):
        counts[op] += int(count)

    del_start = int(match[0]) if (match := re.findall(r"^(\d+)D", cigar)) else 0
    counts["D"] -= del_start

    total = sum(counts.values())
    ref_start = res.end_ref - counts["="] - counts["X"] - counts["D"] + 1
    seq_start = res.end_query - counts["="] - counts["X"] - counts["I"] + 1
    return AlignResult(
        accuracy_pct=100.0 * safe_div(counts["="], total),
        num_correct=counts["="],
        num_mismatches=counts["X"],
        num_insertions=counts["I"],
        num_deletions=counts["D"],
        ref_len=len(ref),
        seq_len=len(seq),
        align_ref_start=ref_start,
        align_ref_end=res.end_ref,
        align_seq_start=seq_start,
        align_seq_end=res.end_query,
    )


def safe_accuracy(ref: str, seq: str) -> float:
    if not ref or not seq:
        return 0.0
    try:
        return float(accuracy(ref, seq, min_coverage=0.5))
    except Exception:
        return 0.0


def confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(true_labels, pred_labels):
        matrix[int(truth), int(pred)] += 1
    return matrix


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


def compute_binary_mod_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    roc_auc, pr_auc = binary_auc_metrics(y_true, y_prob)

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    return {
        "num_sites": int(y_true.size),
        "num_positive": positives,
        "num_negative": negatives,
        "positive_rate": safe_div(positives, y_true.size),
        "predicted_positive_rate": float(np.mean(y_pred)) if y_pred.size else 0.0,
        "accuracy": safe_div(tp + tn, y_true.size),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, (2 * tp) + fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_positive_prob": float(np.mean(y_prob[y_true == 1])) if positives else None,
        "mean_negative_prob": float(np.mean(y_prob[y_true == 0])) if negatives else None,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def compute_multiclass_mod_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray, num_classes: int) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    supports = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    per_class = []
    f1s = []
    for cls in range(num_classes):
        tp = int(cm[cls, cls])
        support = int(supports[cls])
        pred_count = int(predicted[cls])
        precision = safe_div(tp, pred_count)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        f1s.append(f1)
        per_class.append({
            "class_id": cls,
            "support": support,
            "predicted": pred_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return {
        "num_sites": int(y_true.size),
        "accuracy": safe_div(np.sum(y_true == y_pred), y_true.size),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences.size else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def get_global_mod_labels(model) -> List[str]:
    labels = list(getattr(model, "mod_global_labels", []) or [])
    if labels:
        return labels
    num_classes = int(getattr(model, "num_mod_classes", 1))
    if num_classes <= 1:
        return ["canonical", "modified"]
    return [f"class_{idx}" for idx in range(num_classes)]


def get_head_display_name(model, head_name: str) -> str:
    return str(getattr(model, "head_display_names", {}).get(head_name, head_name))


def compute_head_predictions(logits: torch.Tensor, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = logits.detach().to(torch.float32)
    num_classes = int(logits.shape[-1])

    if num_classes == 1:
        probs = torch.ones_like(logits, dtype=torch.float32)
        preds = torch.zeros((logits.shape[0],), device=logits.device, dtype=torch.int64)
        conf = torch.ones((logits.shape[0],), device=logits.device, dtype=torch.float32)
        return (
            preds.cpu().numpy().astype(np.int64),
            conf.cpu().numpy().astype(np.float32),
            probs.cpu().numpy().astype(np.float32),
        )

    probs = torch.softmax(logits, dim=-1)
    if num_classes == 2:
        modified_prob = probs[:, 1]
        preds = torch.where(
            modified_prob >= threshold,
            torch.ones_like(modified_prob, dtype=torch.int64),
            torch.zeros_like(modified_prob, dtype=torch.int64),
        )
    else:
        preds = probs.argmax(dim=-1)
    conf = probs.gather(1, preds.unsqueeze(-1)).squeeze(-1)
    return (
        preds.cpu().numpy().astype(np.int64),
        conf.cpu().numpy().astype(np.float32),
        probs.cpu().numpy().astype(np.float32),
    )


def aggregate_modification_metrics(model, projection: Dict[str, object], threshold: float) -> Dict[str, object]:
    global_labels = get_global_mod_labels(model)
    canonical_global_ids = {
        int(getattr(model, "global_label_to_id", {}).get(f"canonical_{base}"))
        for base in getattr(model, "mod_bases", [])
        if f"canonical_{base}" in getattr(model, "global_label_to_id", {})
    }

    global_true_parts: List[np.ndarray] = []
    global_pred_parts: List[np.ndarray] = []
    global_conf_parts: List[np.ndarray] = []
    binary_true_parts: List[np.ndarray] = []
    binary_prob_parts: List[np.ndarray] = []
    per_head_summary: Dict[str, object] = {}

    for head_name, head_projection in projection.get("per_head", {}).items():
        flat_logits = head_projection["flat_logits"].detach().to(torch.float32)
        flat_targets = head_projection["flat_targets"].detach().cpu().numpy().astype(np.int64)
        flat_global_targets = head_projection["flat_global_targets"].detach().cpu().numpy().astype(np.int64)
        class_labels = list(getattr(model, "mod_head_defs", {}).get(head_name, []))
        head_display_name = get_head_display_name(model, head_name)
        num_classes = int(flat_logits.shape[-1]) if flat_logits.ndim == 2 else 0

        if flat_global_targets.size == 0:
            per_head_summary[head_name] = {
                "display_name": head_display_name,
                "task_type": "empty",
                "num_sites": 0,
                "class_labels": class_labels,
                "global_label_ids": list(getattr(model, "head_global_ids", {}).get(head_name, [])),
            }
            continue

        local_pred, local_conf, local_probs = compute_head_predictions(flat_logits, threshold)
        head_global_ids = list(getattr(model, "head_global_ids", {}).get(head_name, []))
        global_pred = np.asarray([head_global_ids[idx] for idx in local_pred], dtype=np.int64)

        global_true_parts.append(flat_global_targets)
        global_pred_parts.append(global_pred)
        global_conf_parts.append(local_conf)

        is_modified_true = (~np.isin(flat_global_targets, list(canonical_global_ids))).astype(np.int64)
        if local_probs.shape[1] <= 1:
            modified_prob = np.zeros(flat_global_targets.shape[0], dtype=np.float32)
        else:
            modified_prob = 1.0 - local_probs[:, 0]
        binary_true_parts.append(is_modified_true)
        binary_prob_parts.append(modified_prob.astype(np.float32))

        if num_classes <= 1:
            per_head_metrics = {
                "task_type": "single_class",
                "num_sites": int(flat_targets.size),
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "mean_confidence": 1.0,
                "class_labels": class_labels,
            }
        elif num_classes == 2:
            per_head_metrics = {
                "task_type": "binary",
                "threshold": threshold,
                "class_labels": class_labels,
                **compute_binary_mod_metrics(flat_targets, local_probs[:, 1], threshold),
            }
        else:
            per_head_metrics = {
                "task_type": "multiclass",
                "class_labels": class_labels,
                **compute_multiclass_mod_metrics(flat_targets, local_pred, local_conf, num_classes),
            }

        per_head_summary[head_name] = {
            "display_name": head_display_name,
            "global_label_ids": head_global_ids,
            **per_head_metrics,
        }

    if global_true_parts:
        global_true = np.concatenate(global_true_parts)
        global_pred = np.concatenate(global_pred_parts)
        global_conf = np.concatenate(global_conf_parts)
        overall = {
            "task_type": "global_multiclass",
            "global_labels": global_labels,
            **compute_multiclass_mod_metrics(global_true, global_pred, global_conf, len(global_labels)),
        }
    else:
        overall = {
            "task_type": "global_multiclass",
            "global_labels": global_labels,
            "num_sites": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mean_confidence": 0.0,
            "confusion_matrix": [],
            "per_class": [],
        }

    if binary_true_parts:
        binary_true = np.concatenate(binary_true_parts)
        binary_prob = np.concatenate(binary_prob_parts)
        modified_vs_canonical = {
            "task_type": "binary_modified_vs_canonical",
            "threshold": threshold,
            **compute_binary_mod_metrics(binary_true, binary_prob, threshold),
        }
    else:
        modified_vs_canonical = {
            "task_type": "binary_modified_vs_canonical",
            "threshold": threshold,
            "num_sites": 0,
            "num_positive": 0,
            "num_negative": 0,
            "positive_rate": 0.0,
            "predicted_positive_rate": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "mean_positive_prob": None,
            "mean_negative_prob": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    return {
        "overall": overall,
        "modified_vs_canonical": modified_vs_canonical,
        "per_head": per_head_summary,
    }


def maybe_trim_refs(refs: List[str], model) -> List[str]:
    n_pre = getattr(model, "n_pre_context_bases", 0)
    n_post = getattr(model, "n_post_context_bases", 0)
    if n_pre <= 0 and n_post <= 0:
        return refs

    trimmed = []
    for ref in refs:
        start = n_pre
        end = len(ref) - n_post if n_post else len(ref)
        trimmed.append(ref[start:end])
    return trimmed


def resolve_output_dir(args) -> Path:
    weights_label = "last" if args.weights is None else str(args.weights)
    if args.output_dir:
        return args.output_dir
    return Path(args.model_directory) / f"validate_{args.dataset}_weights_{weights_label}"


def resolve_data_settings(args) -> DataSettings:
    output_dir = resolve_output_dir(args)
    has_validation_dir = (args.directory / "validation").exists()
    if has_validation_dir:
        num_train_chunks = args.chunks if args.chunks is not None else args.valid_chunks
        num_valid_chunks = args.valid_chunks if args.valid_chunks is not None else args.chunks
        if num_train_chunks is None:
            num_train_chunks = 512
        if num_valid_chunks is None:
            num_valid_chunks = 512
        return DataSettings(args.directory, num_train_chunks, num_valid_chunks, output_dir)

    if args.chunks is None or args.valid_chunks is None:
        raise ValueError(
            "This dataset has no validation/ directory. Pass both --chunks and --valid-chunks so the evaluator can "
            "reproduce the same train/valid split that train_mod used."
        )
    return DataSettings(args.directory, args.chunks, args.valid_chunks, output_dir)


def save_base_plots(base_df: pd.DataFrame, output_dir: Path) -> List[str]:
    written = []
    if plt is None or base_df.empty:
        return written

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(base_df["accuracy_pct"], bins=20, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Base Accuracy Distribution")
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_ylabel("Chunks")

    axes[1].scatter(base_df["ref_len"], base_df["seq_len"], s=14, alpha=0.7, color="#ff7f0e")
    axes[1].set_title("Predicted Length vs Reference Length")
    axes[1].set_xlabel("Reference Length")
    axes[1].set_ylabel("Predicted Length")

    fig.tight_layout()
    path = output_dir / "base_eval.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    error_rates = {
        "sub": safe_div(base_df["num_mismatches"].sum(), max(base_df["num_correct"].sum(), 1)),
        "ins": safe_div(base_df["num_insertions"].sum(), max(base_df["num_correct"].sum(), 1)),
        "del": safe_div(base_df["num_deletions"].sum(), max(base_df["num_correct"].sum(), 1)),
    }
    ax.bar(list(error_rates.keys()), list(error_rates.values()), color=["#2ca02c", "#d62728", "#9467bd"])
    ax.set_title("Alignment Error Rates")
    ax.set_ylabel("Rate")
    path = output_dir / "base_error_rates.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def save_binary_mod_plots(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, output_dir: Path) -> List[str]:
    written = []
    if plt is None or y_true.size == 0:
        return written

    fig, ax = plt.subplots(figsize=(7, 4.5))
    positives = y_prob[y_true == 1]
    negatives = y_prob[y_true == 0]
    if negatives.size:
        ax.hist(negatives, bins=30, alpha=0.6, label="true=0", color="#1f77b4")
    if positives.size:
        ax.hist(positives, bins=30, alpha=0.6, label="true=1", color="#d62728")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"threshold={threshold:.2f}")
    ax.set_title("Modification Probability Distribution")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Sites")
    ax.legend()
    path = output_dir / "mod_probability_hist.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.85)
    ax.set_title("Binary Mod Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    path = output_dir / "mod_confusion_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def save_multiclass_mod_plot(confusion: np.ndarray, output_dir: Path, class_labels: Optional[List[str]] = None) -> List[str]:
    written = []
    if plt is None or confusion.size == 0:
        return written

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.85)
    ax.set_title("Modification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(confusion.shape[1]))
    ax.set_yticks(range(confusion.shape[0]))
    if class_labels and len(class_labels) == confusion.shape[0]:
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(int(confusion[i, j])), ha="center", va="center", color="black")
    path = output_dir / "mod_confusion_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def save_training_curves(model_directory: Path, output_dir: Path) -> List[str]:
    written = []
    if plt is None:
        return written

    training_csv = model_directory / "training.csv"
    if not training_csv.exists():
        return written

    history = pd.read_csv(training_csv)
    if history.empty:
        return written

    for column in ["epoch", "train_loss", "train_mod_loss", "train_total_loss", "val_loss", "val_mod_loss", "val_total_loss", "val_mean", "val_median"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_cols = [
        ("train_loss", "train base loss"),
        ("val_loss", "val base loss"),
        ("train_mod_loss", "train mod loss"),
        ("val_mod_loss", "val mod loss"),
    ]
    for column, label in loss_cols:
        if column in history.columns and history[column].notna().any():
            axes[0].plot(history["epoch"], history[column], label=label)
    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "val_mean" in history.columns and history["val_mean"].notna().any():
        axes[1].plot(history["epoch"], history["val_mean"], label="val mean acc")
    if "val_median" in history.columns and history["val_median"].notna().any():
        axes[1].plot(history["epoch"], history["val_median"], label="val median acc")
    axes[1].set_title("Validation Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def save_alignment_projection_plots(alignment_df: pd.DataFrame, base_df: pd.DataFrame, output_dir: Path) -> List[str]:
    written = []
    if plt is None or alignment_df.empty:
        return written

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(alignment_df["target_coverage"], bins=25, alpha=0.7, label="target coverage", color="#1f77b4")
    axes[0].hist(alignment_df["valid_mod_coverage"], bins=25, alpha=0.7, label="valid mod coverage", color="#d62728")
    axes[0].set_title("Per-base Projection Coverage")
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Chunks")
    axes[0].legend()

    if not base_df.empty and "accuracy_pct" in base_df.columns:
        axes[1].scatter(base_df["accuracy_pct"], alignment_df["valid_mod_coverage"], s=14, alpha=0.7, color="#2ca02c")
        axes[1].set_xlabel("Base accuracy (%)")
    else:
        axes[1].scatter(np.arange(len(alignment_df)), alignment_df["valid_mod_coverage"], s=14, alpha=0.7, color="#2ca02c")
        axes[1].set_xlabel("Chunk index")
    axes[1].set_title("Valid Mod Coverage vs Base Accuracy")
    axes[1].set_ylabel("Valid mod coverage")

    fig.tight_layout()
    path = output_dir / "mod_alignment_coverage.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def _downsample_trace(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points, dtype=np.int64)
    return x[idx], y[idx]


def save_signal_alignment_examples(signal_examples: List[Dict[str, object]], output_dir: Path, stride: int) -> List[str]:
    written = []
    if plt is None or not signal_examples:
        return written

    for example in signal_examples:
        signal = np.asarray(example["signal"], dtype=np.float32)
        if signal.size == 0:
            continue

        x = np.arange(signal.size, dtype=np.int64)
        plot_x, plot_signal = _downsample_trace(x, signal, max_points=4000)
        signal_min = float(np.min(signal))
        signal_max = float(np.max(signal))
        signal_span = max(signal_max - signal_min, 1e-6)

        emit_positions = np.asarray(example["emit_positions"], dtype=np.int64)
        emit_signal_positions = np.clip(emit_positions * int(stride), 0, signal.size - 1)
        site_scores = np.asarray(example["site_scores"], dtype=np.float32)
        base_labels = list(example["base_labels"])
        site_records = list(example["site_records"])

        fig, ax_signal = plt.subplots(figsize=(16, 7))
        ax_mod = ax_signal.twinx()

        ax_signal.plot(plot_x, plot_signal, linewidth=0.9, color="#4c566a", label="signal")
        ax_signal.set_title(
            f"Chunk {example['chunk_index']} Aligned Signal / Base / Mod View\n"
            f"pred_len={example['predicted_base_len']} target_len={example['target_len']} "
            f"target_cov={example['target_coverage']:.3f} valid_mod_cov={example['valid_mod_coverage']:.3f}"
        )
        ax_signal.set_xlabel("Signal sample index")
        ax_signal.set_ylabel("Current")
        ax_mod.set_ylabel("Predicted site score")
        ax_mod.set_ylim(-0.10, 1.25)

        if emit_signal_positions.size:
            line_alpha = min(0.18, max(0.03, 18.0 / max(len(emit_signal_positions), 1)))
            for raw_x in emit_signal_positions:
                ax_signal.axvline(int(raw_x), color="#8fbcbb", alpha=line_alpha, linewidth=0.6, zorder=0)

            ax_mod.plot(emit_signal_positions, site_scores, color="#2ca02c", linewidth=1.2, alpha=0.9, label="pred site score")
            scatter = ax_mod.scatter(
                emit_signal_positions,
                site_scores,
                c=site_scores if site_scores.size else np.zeros_like(emit_signal_positions, dtype=np.float32),
                cmap="coolwarm",
                s=22,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.2,
                vmin=0.0,
                vmax=1.0,
                label="pred bases",
            )
            # cbar = fig.colorbar(scatter, ax=ax_signal, pad=0.01)
            # cbar.set_label("Predicted mod probability")

            label_limit = min(len(base_labels), 40)
            label_stride = max(label_limit // 20, 1)
            # label_y = signal_max + 0.04 * signal_span
            # label_y = signal_min - 0.15 
            # for idx in range(0, label_limit, label_stride):
            #     ax_signal.text(
            #         int(emit_signal_positions[idx]),
            #         label_y,
            #         str(base_labels[idx]),
            #         fontsize=10,
            #         ha="center",
            #         va="bottom",
            #         rotation=90,
            #         color="#2e3440",
            #     )

        if site_records:
            aligned_x = np.asarray([
                max(0, min(int(record["time_step"]) * int(stride), signal.size - 1))
                for record in site_records
            ], dtype=np.int64)
            pred_indices = [int(record["predicted_base_index"]) for record in site_records]
            pred_scores = np.asarray([
                float(site_scores[idx]) if idx < len(site_scores) else float(record.get("score", 0.0))
                for idx in pred_indices
            ], dtype=np.float32)
            pred_colors = ["#d62728" if int(record["pred_mod"]) != int(record["true_mod"]) else "#2ca02c" for record in site_records]
            true_colors = ["#5e81ac" if "canonical" in str(record.get("true_mod_label", "")).lower() else "#bf616a" for record in site_records]

            ax_mod.scatter(
                aligned_x,
                pred_scores,
                marker="D",
                c=pred_colors,
                s=34,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.35,
                label="aligned predicted mod",
                zorder=4,
            )
            ax_mod.scatter(
                aligned_x,
                np.full(len(aligned_x), 1.10, dtype=np.float32),
                marker="v",
                c=true_colors,
                s=42,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.35,
                label="aligned target mod",
                zorder=5,
            )

            annotate_limit = min(len(site_records), 30)
            annotate_stride = max(annotate_limit // 15, 1)
            for idx in range(0, annotate_limit, annotate_stride):
                record = site_records[idx]
                x_pos = int(aligned_x[idx])
                ref_text = f"{record['ref_base']}:{record.get('true_mod_label', int(record['true_mod']))}"
                pred_text = f"p:{record.get('pred_mod_label', int(record['pred_mod']))}"
                ax_mod.text(x_pos, 1.16, ref_text, fontsize=10, ha="center", va="bottom", rotation=90, color="#5e81ac")
                ax_mod.text(x_pos, min(float(pred_scores[idx]) + 0.06, 1.02), pred_text, fontsize=10, ha="center", va="bottom", rotation=90, color="#bf616a")

        signal_handles, signal_labels = ax_signal.get_legend_handles_labels()
        mod_handles, mod_labels = ax_mod.get_legend_handles_labels()
        if signal_handles or mod_handles:
            ax_signal.legend(signal_handles + mod_handles, signal_labels + mod_labels, loc="upper left", fontsize=8)

        fig.tight_layout()
        path = output_dir / f"signal_mod_alignment_chunk_{int(example['chunk_index'])}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)

    return written
def build_text_summary(summary: Dict[str, object]) -> str:
    base = summary["base"]
    alignment = summary["alignment"]
    mod = summary["modification"]
    overall = mod["overall"]
    binary = mod["modified_vs_canonical"]
    per_head = mod["per_head"]
    lines = [
        f"model_directory: {summary['model_directory']}",
        f"dataset_directory: {summary['dataset_directory']}",
        f"dataset_split: {summary['dataset_split']}",
        f"evaluated_chunks: {summary['num_chunks']}",
        "",
        "[basecalling]",
        f"mean_loss: {base['mean_loss']:.4f}",
        f"mean_mod_loss: {base['mean_mod_loss']:.4f}",
        f"mean_total_loss: {base['mean_total_loss']:.4f}",
        f"mean_acc_pct: {base['mean_acc_pct']:.3f}",
        f"median_acc_pct: {base['median_acc_pct']:.3f}",
        f"sub_rate: {base['sub_rate']:.4f}",
        f"ins_rate: {base['ins_rate']:.4f}",
        f"del_rate: {base['del_rate']:.4f}",
        "",
        "[alignment]",
        f"mean_target_coverage: {alignment['mean_target_coverage']:.4f}",
        f"mean_predicted_base_coverage: {alignment['mean_predicted_base_coverage']:.4f}",
        f"mean_valid_mod_coverage: {alignment['mean_valid_mod_coverage']:.4f}",
        f"mean_predicted_base_len: {alignment['mean_predicted_base_len']:.2f}",
        f"mean_target_len: {alignment['mean_target_len']:.2f}",
        "",
        "[modification]",
        f"task_type: {overall['task_type']}",
        f"num_sites: {overall['num_sites']}",
        f"accuracy: {overall['accuracy']:.4f}",
        f"macro_f1: {overall['macro_f1']:.4f}",
        f"mean_confidence: {overall['mean_confidence']:.4f}",
        "",
        "[modified_vs_canonical]",
        f"threshold: {binary['threshold']:.3f}",
        f"num_sites: {binary['num_sites']}",
        f"positive_rate: {binary['positive_rate']:.4f}",
        f"predicted_positive_rate: {binary['predicted_positive_rate']:.4f}",
        f"accuracy: {binary['accuracy']:.4f}",
        f"precision: {binary['precision']:.4f}",
        f"recall: {binary['recall']:.4f}",
        f"f1: {binary['f1']:.4f}",
        f"roc_auc: {binary['roc_auc']}",
        f"pr_auc: {binary['pr_auc']}",
    ]
    if per_head:
        lines.extend(["", "[per_head]"])
        for head_name, head_summary in per_head.items():
            lines.append(
                f"- {head_summary['display_name']} ({head_name}): task_type={head_summary['task_type']}, "
                f"num_sites={head_summary['num_sites']}"
            )
            if head_summary["task_type"] == "binary":
                lines.append(
                    f"  accuracy={head_summary['accuracy']:.4f}, f1={head_summary['f1']:.4f}, "
                    f"roc_auc={head_summary['roc_auc']}"
                )
            elif head_summary["task_type"] == "multiclass":
                lines.append(
                    f"  accuracy={head_summary['accuracy']:.4f}, macro_f1={head_summary['macro_f1']:.4f}, "
                    f"mean_confidence={head_summary['mean_confidence']:.4f}"
                )
            elif head_summary["task_type"] == "single_class":
                lines.append("  single-class head (canonical-only in this run)")
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "[warnings]"])
        lines.extend(f"- {item}" for item in warnings)
    return "\n".join(lines) + "\n"


def main(args):
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    init(args.seed, args.device, deterministic=not args.nondeterministic)
    use_half = str(args.device).startswith("cuda") and not args.no_half
    model = load_model(
        args.model_directory,
        args.device,
        weights=args.weights,
        half=use_half,
        compile=not args.no_compile,
    )
    supports_mod_eval = hasattr(model, "align_predictions_to_targets") and hasattr(model, "predict_mods")

    standardisation = model.config.get("standardisation", {}) if args.standardise else {}
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=standardisation,
    )
    compute_settings = ComputeSettings(batch_size=args.batchsize, num_workers=args.num_workers, seed=args.seed)
    data_settings = resolve_data_settings(args)
    if supports_mod_eval:
        train_loader, valid_loader = load_mod_data(data_settings, model_setup, compute_settings)
    else:
        train_loader, valid_loader = load_data(data_settings, model_setup, compute_settings)
    dataloader = valid_loader if args.dataset == "valid" else train_loader

    model_dtype = next(model.parameters()).dtype
    use_amp = str(args.device).startswith("cuda") and not args.no_amp
    warnings: List[str] = []
    if not supports_mod_eval:
        warnings.append(
            "Loaded model does not provide modification-head helpers (predict_mods/align_predictions_to_targets). "
            "This run will report basecalling-only metrics, and modification metrics will be marked as unavailable."
        )
    if getattr(model, "mod_loss_weight", 1.0) == 0:
        warnings.append("model.mod_loss_weight is 0. The modification branch is not contributing to total_loss.")
    if plt is None:
        warnings.append("matplotlib is not installed. Summary files will be written, but PNG plots will be skipped.")

    base_records: List[Dict[str, object]] = []
    sequence_examples: List[Dict[str, object]] = []
    mod_site_examples: List[Dict[str, object]] = []
    mod_alignment_records: List[Dict[str, object]] = []
    predicted_base_examples: List[Dict[str, object]] = []
    signal_examples: List[Dict[str, object]] = []
    projection_batches: List[Dict[str, object]] = []

    loss_sums = {"loss": 0.0, "mod_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    global_chunk_index = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), ascii=True, ncols=100, desc="evaluating"):
            if supports_mod_eval:
                data, targets, lengths, mod_targets, *extra = batch
            else:
                data, targets, lengths, *extra = batch
                mod_targets = None
            data_cpu = data.detach().cpu()
            data = data.to(args.device, dtype=model_dtype, non_blocking=True)
            targets_device = targets.to(args.device, non_blocking=True)
            lengths_device = lengths.to(args.device, non_blocking=True)
            mod_targets_device = mod_targets.to(args.device, non_blocking=True) if mod_targets is not None else None
            extra_device = [x.to(args.device, non_blocking=True) for x in extra]

            with amp.autocast("cuda", enabled=use_amp):
                outputs = model(data, *extra_device)
                if supports_mod_eval:
                    loss_outputs = model.loss(outputs, targets_device, lengths_device, mod_targets_device)
                else:
                    loss_outputs = model.loss(outputs, targets_device, lengths_device)

            if isinstance(loss_outputs, dict):
                losses = {
                    "loss": float(loss_outputs.get("loss", loss_outputs.get("total_loss", 0.0))),
                    "mod_loss": float(loss_outputs.get("mod_loss", 0.0)),
                    "total_loss": float(loss_outputs.get("total_loss", loss_outputs.get("loss", 0.0))),
                }
            else:
                scalar_loss = float(loss_outputs.item())
                losses = {"loss": scalar_loss, "mod_loss": 0.0, "total_loss": scalar_loss}

            num_batches += 1
            for key in loss_sums:
                loss_sums[key] += float(losses[key])

            if hasattr(model, "decode_batch"):
                seqs = model.decode_batch(outputs)
            else:
                raise RuntimeError("The loaded model does not provide decode_batch().")

            refs = [decode_ref(target, model.alphabet) for target in targets]
            refs = maybe_trim_refs(refs, model)
            accs = [safe_accuracy(ref, seq) for ref, seq in zip(refs, seqs)]

            for local_idx, (ref, seq, acc_pct) in enumerate(zip(refs, seqs, accs)):
                aln = align(ref, seq)
                record = {
                    "chunk_index": global_chunk_index + local_idx,
                    "accuracy_pct": acc_pct,
                    **asdict(aln),
                }
                base_records.append(record)
                if len(sequence_examples) < args.max_examples:
                    sequence_examples.append({
                        "chunk_index": global_chunk_index + local_idx,
                        "reference": ref,
                        "prediction": seq,
                        "accuracy_pct": acc_pct,
                    })

            if supports_mod_eval:
                per_base_predictions = model.predict_mods(outputs, mod_threshold=args.mod_threshold)
                remaining_site_slots = max(args.site_report_limit - len(mod_site_examples), 0)
                projection = model.align_predictions_to_targets(
                    outputs,
                    targets_device,
                    lengths_device,
                    mod_targets_device,
                    mod_threshold=args.mod_threshold,
                    include_site_records=remaining_site_slots > 0,
                    site_record_limit=remaining_site_slots,
                )
                projection_batches.append(projection)

                for local_idx, sample_record in enumerate(projection["sample_records"]):
                    mod_alignment_records.append({
                        "chunk_index": global_chunk_index + local_idx,
                        **sample_record,
                    })

                for site_record in projection["site_records"]:
                    mod_site_examples.append({
                        "chunk_index": global_chunk_index + int(site_record["sample_index_in_batch"]),
                        **site_record,
                    })

                for local_idx, prediction in enumerate(per_base_predictions):
                    site_predictions = list(prediction.get("sites", []))
                    if len(predicted_base_examples) < args.max_examples:
                        score_preview = [float(site["score"]) for site in site_predictions[:40]]
                        pred_preview = [site["global_pred_label"] for site in site_predictions[:40]]
                        head_preview = [site["head_name"] for site in site_predictions[:40]]
                        predicted_base_examples.append({
                            "chunk_index": global_chunk_index + local_idx,
                            "predicted_base_len": len(prediction["sequence"]),
                            "sequence_prefix": prediction["sequence"][:80],
                            "first_20_time_steps": json.dumps(prediction["emit_positions"][:20]),
                            "first_40_mod_scores": json.dumps(score_preview),
                            "first_40_mod_preds": json.dumps(pred_preview),
                            "first_40_heads": json.dumps(head_preview),
                        })

                    if len(signal_examples) < args.signal_example_limit:
                        sample_sites = [
                            dict(site_record)
                            for site_record in projection["site_records"]
                            if int(site_record["sample_index_in_batch"]) == local_idx
                        ]
                        if sample_sites:
                            signal_examples.append({
                                "chunk_index": global_chunk_index + local_idx,
                                "signal": data_cpu[local_idx].reshape(-1).numpy().tolist(),
                                "emit_positions": list(prediction["emit_positions"]),
                                "base_labels": list(prediction["base_labels"]),
                                "site_scores": [float(site["score"]) for site in site_predictions],
                                "site_records": sample_sites,
                                "predicted_base_len": len(prediction["sequence"]),
                                "target_len": int(projection["sample_records"][local_idx]["target_len"]),
                                "target_coverage": float(projection["sample_records"][local_idx]["target_coverage"]),
                                "valid_mod_coverage": float(projection["sample_records"][local_idx]["valid_mod_coverage"]),
                            })

            global_chunk_index += len(seqs)

    base_df = pd.DataFrame(base_records)
    alignment_df = pd.DataFrame(mod_alignment_records)
    if base_df.empty:
        raise RuntimeError("No evaluation records were produced.")

    mean_losses = {key: safe_div(value, num_batches) for key, value in loss_sums.items()}
    base_summary = {
        "mean_loss": mean_losses["loss"],
        "mean_mod_loss": mean_losses["mod_loss"],
        "mean_total_loss": mean_losses["total_loss"],
        "mean_acc_pct": float(base_df["accuracy_pct"].mean()),
        "median_acc_pct": float(base_df["accuracy_pct"].median()),
        "mean_ref_len": float(base_df["ref_len"].mean()),
        "mean_seq_len": float(base_df["seq_len"].mean()),
        "sub_rate": safe_div(base_df["num_mismatches"].sum(), max(base_df["num_correct"].sum(), 1)),
        "ins_rate": safe_div(base_df["num_insertions"].sum(), max(base_df["num_correct"].sum(), 1)),
        "del_rate": safe_div(base_df["num_deletions"].sum(), max(base_df["num_correct"].sum(), 1)),
    }

    if alignment_df.empty:
        if supports_mod_eval:
            warnings.append("No target-axis projection records were produced for the modification branch.")
        alignment_summary = {
            "mean_target_coverage": 0.0,
            "mean_predicted_base_coverage": 0.0,
            "mean_valid_mod_coverage": 0.0,
            "mean_predicted_base_len": 0.0,
            "mean_target_len": 0.0,
        }
    else:
        alignment_summary = {
            "mean_target_coverage": float(alignment_df["target_coverage"].mean()),
            "mean_predicted_base_coverage": float(alignment_df["predicted_base_coverage"].mean()),
            "mean_valid_mod_coverage": float(alignment_df["valid_mod_coverage"].mean()),
            "mean_predicted_base_len": float(alignment_df["predicted_base_len"].mean()),
            "mean_target_len": float(alignment_df["target_len"].mean()),
        }
        if alignment_summary["mean_valid_mod_coverage"] < 0.5:
            warnings.append(
                "Per-base target-axis projection is covering fewer than half of valid modification sites on average. "
                "This usually means base predictions and references are still too far apart for stable mod supervision."
            )

    if supports_mod_eval:
        merged_projection = {
            "per_head": {},
            "sample_records": [],
            "site_records": [],
        }
        for projection in projection_batches:
            merged_projection["sample_records"].extend(projection["sample_records"])
            merged_projection["site_records"].extend(projection["site_records"])
            for head_name, head_projection in projection["per_head"].items():
                merged_projection["per_head"].setdefault(head_name, {
                    "flat_logits": [],
                    "flat_targets": [],
                    "flat_global_targets": [],
                })
                for key in ("flat_logits", "flat_targets", "flat_global_targets"):
                    merged_projection["per_head"][head_name][key].append(head_projection[key].detach().cpu())

        for head_name, head_projection in merged_projection["per_head"].items():
            flat_logits_parts = head_projection["flat_logits"]
            flat_targets_parts = head_projection["flat_targets"]
            flat_global_target_parts = head_projection["flat_global_targets"]
            if flat_logits_parts:
                head_projection["flat_logits"] = torch.cat(flat_logits_parts, dim=0)
                head_projection["flat_targets"] = torch.cat(flat_targets_parts, dim=0)
                head_projection["flat_global_targets"] = torch.cat(flat_global_target_parts, dim=0)
            else:
                num_classes = len(getattr(model, "mod_head_defs", {}).get(head_name, []))
                head_projection["flat_logits"] = torch.zeros((0, num_classes), dtype=torch.float32)
                head_projection["flat_targets"] = torch.zeros((0,), dtype=torch.long)
                head_projection["flat_global_targets"] = torch.zeros((0,), dtype=torch.long)

        mod_summary = aggregate_modification_metrics(model, merged_projection, args.mod_threshold)
        if mod_summary["overall"]["num_sites"] == 0:
            warnings.append("No aligned valid modification sites were available for evaluation.")
        elif mod_summary["modified_vs_canonical"]["num_positive"] == 0 or mod_summary["modified_vs_canonical"]["num_negative"] == 0:
            warnings.append(
                "All aligned valid modification labels collapse to one side of the modified-vs-canonical split. "
                "Binary discrimination metrics are therefore incomplete for this dataset."
            )
    else:
        merged_projection = {"per_head": {}, "sample_records": [], "site_records": []}
        mod_summary = {
            "overall": {
                "task_type": "unavailable",
                "global_labels": [],
                "num_sites": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "mean_confidence": 0.0,
                "confusion_matrix": [],
                "per_class": [],
            },
            "modified_vs_canonical": {
                "task_type": "unavailable",
                "threshold": args.mod_threshold,
                "num_sites": 0,
                "num_positive": 0,
                "num_negative": 0,
                "positive_rate": 0.0,
                "predicted_positive_rate": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "mean_positive_prob": None,
                "mean_negative_prob": None,
                "roc_auc": None,
                "pr_auc": None,
            },
            "per_head": {},
        }

    summary = {
        "model_directory": str(Path(args.model_directory).resolve()),
        "dataset_directory": str(args.directory.resolve()),
        "dataset_split": args.dataset,
        "weights": "last" if args.weights is None else args.weights,
        "num_chunks": int(len(base_df)),
        "base": base_summary,
        "alignment": alignment_summary,
        "modification": mod_summary,
        "warnings": warnings,
    }

    base_df.to_csv(output_dir / "base_alignments.tsv", sep="\t", index=False)
    pd.DataFrame(sequence_examples).to_csv(output_dir / "sequence_examples.tsv", sep="\t", index=False)
    pd.DataFrame(mod_site_examples).to_csv(output_dir / "mod_site_examples.tsv", sep="\t", index=False)
    alignment_df.to_csv(output_dir / "mod_alignment_summary.tsv", sep="\t", index=False)
    pd.DataFrame(predicted_base_examples).to_csv(output_dir / "predicted_base_examples.tsv", sep="\t", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (output_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")

    written_plots: List[str] = []
    written_plots.extend(save_base_plots(base_df, output_dir))
    written_plots.extend(save_alignment_projection_plots(alignment_df, base_df, output_dir))
    binary_summary = summary["modification"]["modified_vs_canonical"]
    overall_summary = summary["modification"]["overall"]
    if supports_mod_eval and binary_summary["num_sites"] > 0:
        binary_true = []
        binary_prob = []
        canonical_global_ids = {
            int(getattr(model, "global_label_to_id", {}).get(f"canonical_{base}"))
            for base in getattr(model, "mod_bases", [])
            if f"canonical_{base}" in getattr(model, "global_label_to_id", {})
        }
        for head_name, head_projection in merged_projection["per_head"].items():
            flat_logits = head_projection["flat_logits"]
            flat_global_targets = head_projection["flat_global_targets"].cpu().numpy().astype(np.int64)
            if flat_global_targets.size == 0:
                continue
            _, _, local_probs = compute_head_predictions(flat_logits, args.mod_threshold)
            is_modified_true = (~np.isin(flat_global_targets, list(canonical_global_ids))).astype(np.int64)
            if local_probs.shape[1] <= 1:
                modified_prob = np.zeros(flat_global_targets.shape[0], dtype=np.float32)
            else:
                modified_prob = 1.0 - local_probs[:, 0]
            binary_true.append(is_modified_true)
            binary_prob.append(modified_prob.astype(np.float32))
        written_plots.extend(
            save_binary_mod_plots(
                np.concatenate(binary_true) if binary_true else np.array([], dtype=np.int64),
                np.concatenate(binary_prob) if binary_prob else np.array([], dtype=np.float32),
                args.mod_threshold,
                output_dir,
            )
        )
    confusion = np.array(overall_summary["confusion_matrix"], dtype=np.int64)
    if supports_mod_eval:
        written_plots.extend(save_multiclass_mod_plot(confusion, output_dir, class_labels=overall_summary.get("global_labels")))
    written_plots.extend(save_training_curves(Path(args.model_directory), output_dir))
    if supports_mod_eval:
        written_plots.extend(save_signal_alignment_examples(signal_examples, output_dir, stride=getattr(model, "stride", 1)))

    if plt is not None and not written_plots:
        warnings.append("No PNG plots were written. Check whether evaluation arrays were empty or training.csv was missing.")
        summary["warnings"] = warnings
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        (output_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")

    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {output_dir}")


def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("--model_directory", type=Path, required=True, help="The path to the model wating to be validated")
    parser.add_argument("--directory", type=Path, required=True, help="Dataset directory containing chunks.npy etc.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=["train", "valid"], default="valid")
    parser.add_argument("--weights", type=int, default=None, help="Checkpoint epoch to load. Default: latest checkpoint.")
    parser.add_argument("--chunks", type=int, default=None, help="Same meaning as train_mod --chunks when the dataset has no validation/ directory.")
    parser.add_argument("--valid-chunks", type=int, default=None, help="Same meaning as train_mod --valid-chunks when the dataset has no validation/ directory.")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--mod-threshold", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--site-report-limit", type=int, default=20000)
    parser.add_argument("--signal-example-limit", type=int, default=6, help="How many per-chunk signal/base/mod alignment plots to save.")
    parser.add_argument("--standardise", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())