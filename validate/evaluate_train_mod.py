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
import torch.nn.functional as F
from tqdm import tqdm

from bonito.data import ComputeSettings, DataSettings, ModelSetup, load_mod_data
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
    if not seq:
        return AlignResult(ref_len=len(ref), seq_len=0)

    res = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
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


def interpolate_mod_logits(mod_logits: torch.Tensor, target_width: int) -> torch.Tensor:
    mod_logits = mod_logits.permute(0, 2, 1)
    mod_logits = F.interpolate(mod_logits, size=target_width, mode="linear", align_corners=False)
    return mod_logits.permute(0, 2, 1)


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


def save_multiclass_mod_plot(confusion: np.ndarray, output_dir: Path) -> List[str]:
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


def build_text_summary(summary: Dict[str, object]) -> str:
    base = summary["base"]
    alignment = summary["alignment"]
    mod = summary["modification"]
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
        f"task_type: {mod['task_type']}",
        f"num_sites: {mod['num_sites']}",
    ]
    if mod["task_type"] == "binary":
        lines.extend([
            f"threshold: {mod['threshold']:.3f}",
            f"positive_rate: {mod['positive_rate']:.4f}",
            f"predicted_positive_rate: {mod['predicted_positive_rate']:.4f}",
            f"accuracy: {mod['accuracy']:.4f}",
            f"precision: {mod['precision']:.4f}",
            f"recall: {mod['recall']:.4f}",
            f"f1: {mod['f1']:.4f}",
            f"roc_auc: {mod['roc_auc']}",
            f"pr_auc: {mod['pr_auc']}",
        ])
    else:
        lines.extend([
            f"accuracy: {mod['accuracy']:.4f}",
            f"macro_f1: {mod['macro_f1']:.4f}",
            f"mean_confidence: {mod['mean_confidence']:.4f}",
        ])
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
    if not hasattr(model, "align_predictions_to_targets") or not hasattr(model, "predict_mods"):
        raise RuntimeError("Loaded model does not provide per-base modification alignment helpers.")

    standardisation = model.config.get("standardisation", {}) if args.standardise else {}
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=standardisation,
    )
    compute_settings = ComputeSettings(batch_size=args.batchsize, num_workers=args.num_workers, seed=args.seed)
    data_settings = resolve_data_settings(args)
    train_loader, valid_loader = load_mod_data(data_settings, model_setup, compute_settings)
    dataloader = valid_loader if args.dataset == "valid" else train_loader

    model_dtype = next(model.parameters()).dtype
    use_amp = str(args.device).startswith("cuda") and not args.no_amp
    warnings: List[str] = []
    if getattr(model, "mod_loss_weight", 1.0) == 0:
        warnings.append("model.mod_loss_weight is 0. The modification branch is not contributing to total_loss.")
    if plt is None:
        warnings.append("matplotlib is not installed. Summary files will be written, but PNG plots will be skipped.")

    base_records: List[Dict[str, object]] = []
    sequence_examples: List[Dict[str, object]] = []
    mod_site_examples: List[Dict[str, object]] = []
    mod_alignment_records: List[Dict[str, object]] = []
    predicted_base_examples: List[Dict[str, object]] = []

    binary_true: List[np.ndarray] = []
    binary_prob: List[np.ndarray] = []
    multiclass_true: List[np.ndarray] = []
    multiclass_pred: List[np.ndarray] = []
    multiclass_conf: List[np.ndarray] = []

    loss_sums = {"loss": 0.0, "mod_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    global_chunk_index = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), ascii=True, ncols=100, desc="evaluating"):
            data, targets, lengths, mod_targets, *extra = batch
            data = data.to(args.device, dtype=model_dtype, non_blocking=True)
            targets_device = targets.to(args.device, non_blocking=True)
            lengths_device = lengths.to(args.device, non_blocking=True)
            mod_targets_device = mod_targets.to(args.device, non_blocking=True)
            extra_device = [x.to(args.device, non_blocking=True) for x in extra]

            with amp.autocast("cuda", enabled=use_amp):
                outputs = model(data, *extra_device)
                losses = model.loss(outputs, targets_device, lengths_device, mod_targets_device)

            num_batches += 1
            for key in loss_sums:
                loss_sums[key] += float(losses[key].item())

            if hasattr(model, "decode_batch"):
                seqs = model.decode_batch(outputs)
            else:
                raise RuntimeError("The loaded model does not provide decode_batch().")

            refs = [decode_ref(target, model.alphabet) for target in targets]
            refs = maybe_trim_refs(refs, model)
            accs = [accuracy(ref, seq, min_coverage=0.5) if seq else 0.0 for ref, seq in zip(refs, seqs)]

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

            flat_logits = projection["flat_logits"].detach().to(torch.float32)
            flat_targets = projection["flat_targets"].detach().cpu().numpy().astype(np.int64)
            if flat_targets.size:
                if getattr(model, "num_mod_classes", 1) == 1:
                    binary_true.append(flat_targets)
                    binary_prob.append(torch.sigmoid(flat_logits.squeeze(-1)).cpu().numpy().astype(np.float32))
                else:
                    probs = torch.softmax(flat_logits, dim=-1)
                    multiclass_true.append(flat_targets)
                    multiclass_pred.append(probs.argmax(dim=-1).cpu().numpy().astype(np.int64))
                    multiclass_conf.append(probs.max(dim=-1).values.cpu().numpy().astype(np.float32))

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
                if len(predicted_base_examples) >= args.max_examples:
                    break
                if getattr(model, "num_mod_classes", 1) == 1:
                    score_preview = prediction["mod_probs"][:40]
                    pred_preview = prediction["mod_preds"][:40]
                else:
                    score_preview = [max(row) for row in prediction["mod_probs"][:40]]
                    pred_preview = prediction["mod_preds"][:40]
                predicted_base_examples.append({
                    "chunk_index": global_chunk_index + local_idx,
                    "predicted_base_len": len(prediction["sequence"]),
                    "sequence_prefix": prediction["sequence"][:80],
                    "first_20_time_steps": json.dumps(prediction["emit_positions"][:20]),
                    "first_40_mod_scores": json.dumps(score_preview),
                    "first_40_mod_preds": json.dumps(pred_preview),
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

    num_mod_classes = getattr(model, "num_mod_classes", 1)
    if num_mod_classes == 1:
        y_true = np.concatenate(binary_true) if binary_true else np.array([], dtype=np.int64)
        y_prob = np.concatenate(binary_prob) if binary_prob else np.array([], dtype=np.float32)
        mod_summary: Dict[str, object] = {
            "task_type": "binary",
            "threshold": args.mod_threshold,
            **compute_binary_mod_metrics(y_true, y_prob, args.mod_threshold),
        }
        if mod_summary["num_sites"] == 0:
            warnings.append("No aligned valid modification sites were available for evaluation.")
        elif mod_summary["num_positive"] == 0 or mod_summary["num_negative"] == 0:
            warnings.append(
                "All aligned valid modification labels belong to one class. ROC-AUC/PR-AUC are undefined, and this dataset "
                "cannot teach a real modified-vs-unmodified decision boundary on its own."
            )
    else:
        y_true = np.concatenate(multiclass_true) if multiclass_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(multiclass_pred) if multiclass_pred else np.array([], dtype=np.int64)
        y_conf = np.concatenate(multiclass_conf) if multiclass_conf else np.array([], dtype=np.float32)
        mod_summary = {
            "task_type": "multiclass",
            **compute_multiclass_mod_metrics(y_true, y_pred, y_conf, num_mod_classes),
        }
        if mod_summary["num_sites"] == 0:
            warnings.append("No aligned valid modification sites were available for evaluation.")

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
    if num_mod_classes == 1:
        y_true = np.concatenate(binary_true) if binary_true else np.array([], dtype=np.int64)
        y_prob = np.concatenate(binary_prob) if binary_prob else np.array([], dtype=np.float32)
        written_plots.extend(save_binary_mod_plots(y_true, y_prob, args.mod_threshold, output_dir))
    else:
        confusion = np.array(summary["modification"]["confusion_matrix"], dtype=np.int64)
        written_plots.extend(save_multiclass_mod_plot(confusion, output_dir))
    written_plots.extend(save_training_curves(Path(args.model_directory), output_dir))

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
    parser.add_argument("--standardise", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
