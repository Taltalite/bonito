#!/usr/bin/env python3
"""
Run basecalling and per-base modification prediction on POD5 reads.

This is the inference-only counterpart to validate/evaluate_train_mod.py:
- no reference required
- no labeled modification targets required
- input is a POD5 directory

Outputs:
- basecalls.tsv
- basecalls.fasta
- mod_site_predictions.tsv
- summary.json
- summary.txt
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
from tqdm import tqdm

from bonito.crf.basecall import stitch_results, beam_search, to_str
from bonito.multiprocessing import process_cancel, thread_iter
from bonito.reader import Reader
from bonito.util import batchify, column_to_set, init, load_model, unbatchify


def resolve_output_orientation(args, model) -> bool:
    if args.rna is not None:
        return bool(args.rna)
    run_info = model.config.get("run_info", {}) if isinstance(model.config, dict) else {}
    return str(run_info.get("sample_type", "")).lower() == "rna"


def get_reads(args, model) -> Tuple[Iterable[object], int | None]:
    reads_directory = str(args.reads_directory)
    reader = Reader(reads_directory, args.recursive)
    read_ids = column_to_set(args.read_ids)
    num_reads = None
    try:
        _, num_reads = reader.get_read_groups(
            reads_directory,
            args.model_directory,
            n_proc=8,
            recursive=args.recursive,
            read_ids=read_ids,
            skip=args.skip,
            cancel=process_cancel(),
        )
    except Exception:
        num_reads = None

    reads = reader.get_reads(
        reads_directory,
        n_proc=8,
        recursive=args.recursive,
        read_ids=read_ids,
        skip=args.skip,
        do_trim=not args.no_trim,
        scaling_strategy=model.config.get("scaling"),
        norm_params=(
            model.config.get("standardisation")
            if (
                model.config.get("scaling")
                and model.config.get("scaling", {}).get("strategy") == "pa"
            )
            else model.config.get("normalisation")
        ),
        cancel=process_cancel(),
    )
    if args.max_reads:
        reads = islice(reads, args.max_reads)
        if num_reads is not None:
            num_reads = min(num_reads, args.max_reads)
    return reads, num_reads


def iter_read_chunks(reads: Iterable[object], chunksize: int, overlap: int) -> Iterator[Tuple[Tuple[object, int, int], torch.Tensor]]:
    for read in reads:
        signal = torch.from_numpy(read.signal)
        chunks = []
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        total_length = int(signal.shape[-1])
        if chunksize == 0:
            chunks = signal[None, :]
        elif total_length < chunksize:
            from bonito.util import chunk

            chunks = chunk(signal, chunksize, overlap)
        else:
            from bonito.util import chunk

            chunks = chunk(signal, chunksize, overlap)
        yield (read, 0, total_length), chunks


def run_model_on_batch(model, batch: torch.Tensor, model_dtype: torch.dtype, device: str) -> Dict[str, object]:
    with torch.inference_mode():
        outputs = model(batch.to(device, dtype=model_dtype, non_blocking=True))
    return {
        # stitch_results expects chunk dimension first
        "base_scores": outputs["base_scores"].detach().cpu().permute(1, 0, 2).contiguous(),
        "mod_logits_by_base": {
            head_name: logits.detach().cpu().contiguous()
            for head_name, logits in outputs["mod_logits_by_base"].items()
        },
    }


def iter_stitched_outputs(
    model,
    reads: Iterable[object],
    batchsize: int,
    chunksize: int,
    overlap: int,
) -> Iterator[Tuple[object, Dict[str, object]]]:
    model_dtype = next(model.parameters()).dtype
    device = str(next(model.parameters()).device)

    chunks = thread_iter(iter_read_chunks(reads, chunksize=chunksize, overlap=overlap))
    batches = thread_iter(batchify(chunks, batchsize=batchsize))
    batch_outputs = thread_iter(
        (read_key, run_model_on_batch(model, batch, model_dtype, device))
        for read_key, batch in batches
    )
    stitched = thread_iter(
        (
            read,
            stitch_results(
                outputs,
                end - start,
                chunksize,
                overlap,
                model.stride,
                reverse=False,
            ),
        )
        for ((read, start, end), outputs) in unbatchify(batch_outputs)
    )
    yield from stitched


def build_single_read_outputs(stitched_outputs: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "base_scores": stitched_outputs["base_scores"].unsqueeze(1).to(device=device, dtype=torch.float32),
        "mod_logits_by_base": {
            head_name: logits.unsqueeze(0).to(device=device, dtype=torch.float32)
            for head_name, logits in stitched_outputs["mod_logits_by_base"].items()
        },
    }


def decode_basecall_beam_search(
    stitched_outputs: Dict[str, object],
    device: str,
    reverse_output: bool,
    beam_width: int,
    beam_cut: float,
    blank_score: float,
) -> Dict[str, object]:
    scores = stitched_outputs["base_scores"].unsqueeze(1).to(device=device)
    if not str(device).startswith("cuda"):
        raise RuntimeError("Beam-search basecalling currently requires a CUDA device in this script.")
    with torch.inference_mode():
        with torch.cuda.device(scores.device):
            sequence, qstring, moves = beam_search(
                scores,
                beam_width=beam_width,
                beam_cut=beam_cut,
                scale=1.0,
                offset=0.0,
                blank_score=blank_score,
            )
    sequence_signal_order = to_str(sequence)
    qstring_signal_order = to_str(qstring)
    return {
        "sequence_signal_order": sequence_signal_order,
        "sequence": sequence_signal_order[::-1] if reverse_output else sequence_signal_order,
        "qstring_signal_order": qstring_signal_order,
        "qstring": qstring_signal_order[::-1] if reverse_output else qstring_signal_order,
        "moves": moves.detach().cpu().numpy(),
    }


def orient_sites_for_output(site_predictions: List[Dict[str, object]], sequence_length: int, reverse_output: bool) -> List[Dict[str, object]]:
    if not reverse_output:
        ordered = []
        for idx, record in enumerate(site_predictions):
            item = dict(record)
            item["predicted_base_index_signal_order"] = idx
            item["predicted_base_index_output"] = idx
            ordered.append(item)
        return ordered

    ordered = []
    for idx, record in reversed(list(enumerate(site_predictions))):
        item = dict(record)
        item["predicted_base_index_signal_order"] = idx
        item["predicted_base_index_output"] = int(sequence_length - 1 - idx)
        ordered.append(item)
    return ordered


def write_fasta_record(handle, read_id: str, sequence: str, width: int = 80) -> None:
    handle.write(f">{read_id}\n")
    for start in range(0, len(sequence), width):
        handle.write(sequence[start:start + width] + "\n")


def positive_probability(site: Dict[str, object]) -> float | None:
    probs = site.get("local_probs")
    if not isinstance(probs, list) or len(probs) <= 1:
        return None
    return float(probs[1])


def build_text_summary(summary: Dict[str, object]) -> str:
    counts = summary["counts"]
    lines = [
        "[inputs]",
        f"model_directory: {summary['model_directory']}",
        f"reads_directory: {summary['reads_directory']}",
        "",
        "[settings]",
        f"weights: {summary['settings']['weights']}",
        f"mod_threshold: {summary['settings']['mod_threshold']}",
        f"basecall_decoder: {summary['settings']['basecall_decoder']}",
        f"mod_decoder: {summary['settings']['mod_decoder']}",
        f"beam_width: {summary['settings']['beam_width']}",
        f"beam_cut: {summary['settings']['beam_cut']}",
        f"blank_score: {summary['settings']['blank_score']}",
        f"chunksize: {summary['settings']['chunksize']}",
        f"overlap: {summary['settings']['overlap']}",
        f"batchsize: {summary['settings']['batchsize']}",
        f"output_rna_orientation: {summary['settings']['output_rna_orientation']}",
        f"only_modified_sites: {summary['settings']['only_modified_sites']}",
        "",
        "[counts]",
        f"num_reads: {counts['num_reads']}",
        f"num_called_bases: {counts['num_called_bases']}",
        f"num_mod_sites_written: {counts['num_mod_sites_written']}",
        f"num_predicted_modified_sites: {counts['num_predicted_modified_sites']}",
        "",
        "[predicted_mod_labels]",
    ]
    for label, count in summary["predicted_mod_labels"].items():
        lines.append(f"{label}: {count}")
    return "\n".join(lines) + "\n"


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if not hasattr(model, "decode_batch") or not hasattr(model, "predict_mods"):
        raise RuntimeError("Loaded model does not provide decode_batch() and predict_mods().")

    reverse_output = resolve_output_orientation(args, model)
    reads, num_reads = get_reads(args, model)
    device = str(next(model.parameters()).device)
    canonical_labels = {
        str(label)
        for label in getattr(model, "mod_global_labels", [])
        if str(label).startswith("canonical")
    }

    basecalls_tsv = out_dir / "basecalls.tsv"
    basecalls_fasta = out_dir / "basecalls.fasta"
    mod_sites_tsv = out_dir / "mod_site_predictions.tsv"

    num_called_bases = 0
    num_mod_sites_written = 0
    num_predicted_modified_sites = 0
    num_reads_processed = 0
    predicted_mod_labels: Counter[str] = Counter()

    with (
        basecalls_tsv.open("w", encoding="utf-8", newline="") as base_fh,
        basecalls_fasta.open("w", encoding="utf-8") as fasta_fh,
        mod_sites_tsv.open("w", encoding="utf-8", newline="") as mod_fh,
    ):
        base_writer = csv.DictWriter(
            base_fh,
            fieldnames=[
                "read_id",
                "filename",
                "run_id",
                "num_samples",
                "trimmed_samples",
                "signal_length",
                "sequence_length",
                "qstring",
                "sequence",
                "sequence_signal_order",
                "mod_sequence_length",
                "mod_sequence",
                "mod_sequence_signal_order",
            ],
            delimiter="\t",
        )
        base_writer.writeheader()

        mod_writer = csv.DictWriter(
            mod_fh,
            fieldnames=[
                "read_id",
                "filename",
                "run_id",
                "basecall_sequence_length",
                "sequence_length",
                "predicted_base_index_output",
                "predicted_base_index_signal_order",
                "time_step",
                "pred_base",
                "head_name",
                "pred_mod_id",
                "pred_mod_label",
                "pred_is_modified",
                "score",
                "positive_prob",
                "local_probs_json",
            ],
            delimiter="\t",
        )
        mod_writer.writeheader()

        progress = tqdm(
            iter_stitched_outputs(
                model,
                reads,
                batchsize=model.config["basecaller"]["batchsize"],
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"],
            ),
            total=num_reads,
            desc="predicting",
            ascii=True,
            ncols=100,
        )
        for read, stitched_outputs in progress:
            num_reads_processed += 1
            single_outputs = build_single_read_outputs(stitched_outputs, device=device)
            beam_basecall = decode_basecall_beam_search(
                stitched_outputs,
                device=device,
                reverse_output=reverse_output,
                beam_width=args.beam_width,
                beam_cut=args.beam_cut,
                blank_score=args.blank_score,
            )
            sequence_signal_order = model.decode_batch(single_outputs)[0]
            site_prediction_record = model.predict_mods(single_outputs, mod_threshold=args.mod_threshold)[0]

            sequence = beam_basecall["sequence"]
            sequence_length = int(len(sequence))
            mod_sequence = sequence_signal_order[::-1] if reverse_output else sequence_signal_order
            mod_sequence_length = int(len(sequence_signal_order))
            num_called_bases += sequence_length

            base_writer.writerow({
                "read_id": str(read.read_id),
                "filename": str(getattr(read, "filename", "")),
                "run_id": str(getattr(read, "run_id", "")),
                "num_samples": int(getattr(read, "num_samples", 0)),
                "trimmed_samples": int(getattr(read, "trimmed_samples", 0)),
                "signal_length": int(len(read.signal)),
                "sequence_length": sequence_length,
                "qstring": beam_basecall["qstring"],
                "sequence": sequence,
                "sequence_signal_order": beam_basecall["sequence_signal_order"],
                "mod_sequence_length": mod_sequence_length,
                "mod_sequence": mod_sequence,
                "mod_sequence_signal_order": sequence_signal_order,
            })
            write_fasta_record(fasta_fh, str(read.read_id), sequence)

            ordered_sites = orient_sites_for_output(
                site_prediction_record.get("sites", []),
                sequence_length=mod_sequence_length,
                reverse_output=reverse_output,
            )
            for site in ordered_sites:
                pred_label = str(site["global_pred_label"])
                pred_is_modified = pred_label not in canonical_labels
                if args.only_modified_sites and not pred_is_modified:
                    continue

                mod_writer.writerow({
                    "read_id": str(read.read_id),
                    "filename": str(getattr(read, "filename", "")),
                    "run_id": str(getattr(read, "run_id", "")),
                    "basecall_sequence_length": sequence_length,
                    "sequence_length": mod_sequence_length,
                    "predicted_base_index_output": int(site["predicted_base_index_output"]),
                    "predicted_base_index_signal_order": int(site["predicted_base_index_signal_order"]),
                    "time_step": int(site["emit_position"]),
                    "pred_base": str(site["base_label"]),
                    "head_name": str(site["head_name"]),
                    "pred_mod_id": int(site["global_pred_id"]),
                    "pred_mod_label": pred_label,
                    "pred_is_modified": int(pred_is_modified),
                    "score": float(site["score"]),
                    "positive_prob": positive_probability(site),
                    "local_probs_json": json.dumps(site.get("local_probs", [])),
                })
                num_mod_sites_written += 1
                predicted_mod_labels[pred_label] += 1
                if pred_is_modified:
                    num_predicted_modified_sites += 1

    summary = {
        "model_directory": str(Path(args.model_directory).resolve()),
        "reads_directory": str(Path(args.reads_directory).resolve()),
        "artifacts": {
            "basecalls_tsv": str(basecalls_tsv.resolve()),
            "basecalls_fasta": str(basecalls_fasta.resolve()),
            "mod_site_predictions_tsv": str(mod_sites_tsv.resolve()),
        },
        "settings": {
            "weights": "last" if args.weights is None else args.weights,
            "mod_threshold": float(args.mod_threshold),
            "basecall_decoder": "beam_search",
            "mod_decoder": "viterbi",
            "beam_width": int(args.beam_width),
            "beam_cut": float(args.beam_cut),
            "blank_score": float(args.blank_score),
            "chunksize": int(model.config["basecaller"]["chunksize"]),
            "overlap": int(model.config["basecaller"]["overlap"]),
            "batchsize": int(model.config["basecaller"]["batchsize"]),
            "output_rna_orientation": bool(reverse_output),
            "only_modified_sites": bool(args.only_modified_sites),
        },
        "counts": {
            "num_reads": int(num_reads_processed),
            "num_called_bases": int(num_called_bases),
            "num_mod_sites_written": int(num_mod_sites_written),
            "num_predicted_modified_sites": int(num_predicted_modified_sites),
        },
        "predicted_mod_labels": dict(sorted(predicted_mod_labels.items())),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    (out_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")
    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {out_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("reads_directory", type=Path, help="Directory containing POD5 files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=None, type=int, help="Checkpoint epoch to load; default is latest")
    parser.add_argument("--mod-threshold", default=0.5, type=float)
    parser.add_argument("--read-ids")
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--only-modified-sites", action="store_true", default=False)
    parser.add_argument("--beam-width", default=32, type=int)
    parser.add_argument("--beam-cut", default=100.0, type=float)
    parser.add_argument("--blank-score", default=2.0, type=float)
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
