#!/usr/bin/env python3
"""
Inspect whether base and mod outputs are aligned in the current multi-head model.

This script reports three different notions of alignment:
1. Shared encoder time axis: base_scores[T, B, C] vs each mod head logits[B, T, K]
2. Predicted-base axis derived from the CRF Viterbi path: emitted base count vs decoded sequence length
3. Target-base projection coverage after mapping predicted bases back onto the target axis

Outputs:
- summary.json
- batch_summary.tsv
- sample_examples.tsv
- alignment_overview.png (if matplotlib is installed)
"""

from __future__ import annotations

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.amp as amp
from tqdm import tqdm

from bonito.data import ComputeSettings, DataSettings, ModelSetup
from bonito.train_mod_data import load_train_mod_data
from bonito.util import init, load_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def resolve_output_dir(args) -> Path:
    weights_label = "last" if args.weights is None else str(args.weights)
    if args.output_dir:
        return args.output_dir
    return Path(args.model_directory) / f"alignment_check_{args.dataset}_weights_{weights_label}"


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
            "This dataset has no validation/ directory. Pass both --chunks and --valid-chunks so the checker can "
            "reproduce the same train/valid split that train_mod used."
        )
    return DataSettings(args.directory, args.chunks, args.valid_chunks, output_dir)


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


def save_plot(sample_df: pd.DataFrame, output_dir: Path) -> List[str]:
    written = []
    if plt is None or sample_df.empty:
        return written

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(sample_df["target_len"], sample_df["decoded_len"], s=12, alpha=0.7, color="#1f77b4")
    axes[0].set_title("Decoded Length vs Target Length")
    axes[0].set_xlabel("Target length")
    axes[0].set_ylabel("Decoded length")

    axes[1].hist(sample_df["emitted_step_fraction"], bins=30, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Emitted-step Fraction on Time Axis")
    axes[1].set_xlabel("emitted_steps / time_steps")
    axes[1].set_ylabel("Samples")

    fig.tight_layout()
    path = output_dir / "alignment_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


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
    train_loader, valid_loader = load_train_mod_data(data_settings, model_setup, compute_settings)
    dataloader = valid_loader if args.dataset == "valid" else train_loader

    model_dtype = next(model.parameters()).dtype
    use_amp = str(args.device).startswith("cuda") and not args.no_amp

    batch_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    total_samples = 0
    total_time_mismatch = 0
    total_decode_mismatch = 0
    total_samples_without_valid_projection = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader, total=len(dataloader), ascii=True, ncols=100, desc="checking")):
            data, targets, lengths, mod_targets, *extra = batch
            data_device = data.to(args.device, dtype=model_dtype, non_blocking=True)
            extra_device = [x.to(args.device, non_blocking=True) for x in extra]

            with amp.autocast("cuda", enabled=use_amp):
                outputs = model(data_device, *extra_device)

            base_scores = outputs["base_scores"].detach()
            mod_logits_by_base = outputs["mod_logits_by_base"]
            per_base_predictions = model.predict_mods(outputs)
            projection = model.align_predictions_to_targets(outputs, targets.to(args.device), lengths.to(args.device), mod_targets.to(args.device))

            time_steps_base = int(base_scores.shape[0])
            head_time_steps = {
                head_name: int(head_logits.shape[1])
                for head_name, head_logits in mod_logits_by_base.items()
            }
            time_axis_match = all(head_steps == time_steps_base for head_steps in head_time_steps.values())
            if not time_axis_match:
                total_time_mismatch += int(data.shape[0])

            batch_predicted_bases = 0
            batch_decoded_len = 0
            batch_target_len = 0
            batch_target_coverage = 0.0
            batch_valid_mod_coverage = 0.0

            for sample_offset, prediction in enumerate(per_base_predictions):
                predicted_base_len = len(prediction["sequence"])
                emitted_steps = len(prediction["emit_positions"])
                target_len = int(lengths[sample_offset].item())
                time_match = emitted_steps == predicted_base_len
                if not time_match:
                    total_decode_mismatch += 1

                projection_record = projection["sample_records"][sample_offset]
                if projection_record["aligned_valid_mod_sites"] == 0:
                    total_samples_without_valid_projection += 1

                batch_predicted_bases += emitted_steps
                batch_decoded_len += predicted_base_len
                batch_target_len += target_len
                batch_target_coverage += float(projection_record["target_coverage"])
                batch_valid_mod_coverage += float(projection_record["valid_mod_coverage"])

                if len(sample_rows) < args.max_examples:
                    site_preview = prediction["sites"][:40]
                    sample_rows.append({
                        "batch_index": batch_index,
                        "sample_index_in_batch": sample_offset,
                        "global_sample_index": total_samples + sample_offset,
                        "time_steps_base": time_steps_base,
                        "head_time_steps": json.dumps(head_time_steps, sort_keys=True),
                        "time_axis_match": time_axis_match,
                        "target_len": target_len,
                        "decoded_len": predicted_base_len,
                        "emitted_steps": emitted_steps,
                        "emitted_step_fraction": emitted_steps / max(time_steps_base, 1),
                        "decoded_matches_emitted_steps": time_match,
                        "aligned_equal_bases": projection_record["aligned_equal_bases"],
                        "target_coverage": projection_record["target_coverage"],
                        "predicted_base_coverage": projection_record["predicted_base_coverage"],
                        "valid_target_mod_sites": projection_record["valid_target_mod_sites"],
                        "aligned_valid_mod_sites": projection_record["aligned_valid_mod_sites"],
                        "valid_mod_coverage": projection_record["valid_mod_coverage"],
                        "first_20_time_steps": prediction["emit_positions"][:20],
                        "first_40_pred_bases": prediction["sequence"][:40],
                        "first_40_pred_mod_labels": [item["global_pred_label"] for item in site_preview],
                        "first_40_pred_mod_scores": [float(item["score"]) for item in site_preview],
                        "target_sequence_prefix": projection_record["target_sequence_prefix"][:40],
                    })

            batch_rows.append({
                "batch_index": batch_index,
                "batch_size": int(data.shape[0]),
                "time_steps_base": time_steps_base,
                "head_time_steps": json.dumps(head_time_steps, sort_keys=True),
                "time_axis_match": time_axis_match,
                "mean_target_len": batch_target_len / max(int(data.shape[0]), 1),
                "mean_decoded_len": batch_decoded_len / max(int(data.shape[0]), 1),
                "mean_emitted_steps": batch_predicted_bases / max(int(data.shape[0]), 1),
                "mean_emitted_step_fraction": batch_predicted_bases / max(int(data.shape[0]) * max(time_steps_base, 1), 1),
                "mean_target_coverage": batch_target_coverage / max(int(data.shape[0]), 1),
                "mean_valid_mod_coverage": batch_valid_mod_coverage / max(int(data.shape[0]), 1),
            })
            total_samples += int(data.shape[0])

    batch_df = pd.DataFrame(batch_rows)
    sample_df = pd.DataFrame(sample_rows)

    if not batch_df["time_axis_match"].all():
        warnings.append("base_scores time axis and one or more mod head time axes do not always match. This should not happen in the current architecture.")
    if total_decode_mismatch > 0:
        warnings.append("For some samples, the number of emitted Viterbi steps did not equal the predicted sequence length.")
    if total_samples_without_valid_projection > 0:
        warnings.append("Some samples had no valid target-axis modification supervision after Viterbi projection.")
    warnings.append(
        "This checker validates the current per-base path: shared encoder time axis across all mod heads, predicted-base emission axis, and target-axis coverage after Viterbi projection."
    )

    summary = {
        "model_directory": str(Path(args.model_directory).resolve()),
        "dataset_directory": str(args.directory.resolve()),
        "dataset_split": args.dataset,
        "weights": "last" if args.weights is None else args.weights,
        "num_samples_checked": total_samples,
        "shared_time_axis": {
            "all_match": bool(batch_df["time_axis_match"].all()) if not batch_df.empty else False,
            "num_mismatched_samples": int(total_time_mismatch),
            "head_time_steps_example": head_time_steps if batch_rows else {},
        },
        "predicted_base_axis": {
            "num_samples_with_decode_mismatch": int(total_decode_mismatch),
            "mean_decoded_len": float(sample_df["decoded_len"].mean()) if not sample_df.empty else 0.0,
            "mean_emitted_steps": float(sample_df["emitted_steps"].mean()) if not sample_df.empty else 0.0,
            "mean_target_len": float(sample_df["target_len"].mean()) if not sample_df.empty else 0.0,
        },
        "target_axis_projection": {
            "mean_target_coverage": float(sample_df["target_coverage"].mean()) if not sample_df.empty else 0.0,
            "mean_predicted_base_coverage": float(sample_df["predicted_base_coverage"].mean()) if not sample_df.empty else 0.0,
            "mean_valid_mod_coverage": float(sample_df["valid_mod_coverage"].mean()) if not sample_df.empty else 0.0,
            "samples_without_valid_projection": int(total_samples_without_valid_projection),
        },
        "warnings": warnings,
    }

    batch_df.to_csv(output_dir / "batch_summary.tsv", sep="\t", index=False)
    sample_df.to_csv(output_dir / "sample_examples.tsv", sep="\t", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_files = save_plot(sample_df, output_dir)
    if plt is None:
        warnings.append("matplotlib is not installed. PNG plots were skipped.")
        summary["warnings"] = warnings
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"artifacts written to: {output_dir}")
    if plot_files:
        print("plots:", ", ".join(plot_files))


def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("--model_directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=["train", "valid"], default="valid")
    parser.add_argument("--weights", type=int, default=None)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--valid-chunks", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--standardise", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
