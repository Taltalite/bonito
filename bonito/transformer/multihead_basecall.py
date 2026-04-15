"""
Multi-head basecalling helpers that produce standard Bonito result dictionaries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

import edlib
import numpy as np
import torch
from koi.decode import to_str

from bonito.crf.basecall import decode_scores, stitch_results
from bonito.multiprocessing import thread_iter
from bonito.util import batchify, chunk, unbatchify


MOD_CODE_BY_LABEL = {
    "m6A": "a",
    "5mC": "m",
    "5hmC": "h",
}


def _parse_cigar(cigar: str) -> List[Tuple[int, str]]:
    import re

    return [(int(count), op) for count, op in re.findall(r"(\d+)([=XID])", cigar)]


def _equal_alignment_pairs(query_seq: str, target_seq: str) -> List[Tuple[int, int]]:
    if not query_seq or not target_seq:
        return []

    result = edlib.align(query_seq, target_seq, mode="NW", task="path")
    cigar = result.get("cigar")
    if not cigar:
        return []

    query_index = 0
    target_index = 0
    pairs: List[Tuple[int, int]] = []
    for count, op in _parse_cigar(cigar):
        if op == "=":
            for _ in range(count):
                pairs.append((query_index, target_index))
                query_index += 1
                target_index += 1
        elif op == "X":
            query_index += count
            target_index += count
        elif op == "I":
            query_index += count
        elif op == "D":
            target_index += count
        else:
            raise ValueError(f"Unsupported CIGAR op from edlib: {op}")
    return pairs


def _decode_basecall_batch(model, base_scores: torch.Tensor, reverse: bool = False) -> Dict[str, object]:
    if base_scores.ndim != 3:
        raise ValueError(f"Expected base_scores with 3 dims, got shape {tuple(base_scores.shape)}")
    if not str(base_scores.device).startswith("cuda"):
        raise RuntimeError(
            "basecaller_mod currently requires a CUDA device for beam-search basecalling. "
            "Use a CUDA device or validate with CLI/import smoke checks only."
        )
    return decode_scores(base_scores, model.seqdist, reverse=reverse)


def _run_model_on_batch(model, batch: torch.Tensor, reverse: bool = False) -> Dict[str, object]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        outputs = model(batch.to(device=device, dtype=model_dtype, non_blocking=True))

    return {
        "basecall_attrs": _decode_basecall_batch(model, outputs["base_scores"], reverse=reverse),
        "model_outputs": {
            "base_scores": outputs["base_scores"].detach().cpu().permute(1, 0, 2).contiguous(),
            "mod_logits_by_base": {
                head_name: logits.detach().cpu().contiguous()
                for head_name, logits in outputs["mod_logits_by_base"].items()
            },
        },
    }


def _build_single_read_outputs(stitched_outputs: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "base_scores": stitched_outputs["base_scores"].unsqueeze(1).to(device=device, dtype=torch.float32),
        "mod_logits_by_base": {
            head_name: logits.unsqueeze(0).to(device=device, dtype=torch.float32)
            for head_name, logits in stitched_outputs["mod_logits_by_base"].items()
        },
    }


def _format_basecall_result(stride: int, attrs: Dict[str, object], rna: bool = False) -> Dict[str, object]:
    flip = (lambda x: x[::-1]) if rna else (lambda x: x)
    sequence = to_str(attrs["sequence"])
    qstring = to_str(attrs["qstring"])
    moves = attrs["moves"]
    if isinstance(moves, torch.Tensor):
        moves = moves.detach().cpu().numpy()
    elif not isinstance(moves, np.ndarray):
        moves = np.asarray(moves)

    return {
        "stride": stride,
        "moves": moves,
        "qstring": flip(qstring),
        "sequence": flip(sequence),
    }


def _oriented_mod_predictions(mod_prediction: Dict[str, object], reverse_output: bool) -> Tuple[str, List[Dict[str, object]]]:
    sites = list(mod_prediction.get("sites", []))
    sequence = str(mod_prediction.get("sequence", ""))
    if not reverse_output:
        oriented = []
        for index, site in enumerate(sites):
            oriented.append({**site, "oriented_index": index})
        return sequence, oriented

    sequence_length = len(sequence)
    oriented = []
    for index, site in reversed(list(enumerate(sites))):
        oriented.append({**site, "oriented_index": int(sequence_length - 1 - index)})
    return sequence[::-1], oriented


def _positive_mod_probability(site: Dict[str, object]) -> float:
    probs = site.get("local_probs", [])
    local_pred = int(site.get("local_pred_id", 0))
    if not isinstance(probs, list) or not probs:
        return 1.0
    if 0 <= local_pred < len(probs):
        return float(probs[local_pred])
    if len(probs) > 1:
        return float(max(probs[1:]))
    return float(probs[0])


def _build_mod_tags(sequence: str, mapped_sites: List[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)

    for site in mapped_sites:
        label = str(site.get("global_pred_label", ""))
        if label.startswith("canonical_"):
            continue
        mod_code = MOD_CODE_BY_LABEL.get(label)
        if mod_code is None:
            continue

        base_index = int(site["base_index"])
        if base_index < 0 or base_index >= len(sequence):
            continue
        base_char = str(sequence[base_index]).upper()
        if base_char not in {"A", "C", "G", "T", "U"}:
            continue

        prob = min(max(_positive_mod_probability(site), 0.0), 1.0)
        grouped[(base_char, mod_code)].append((base_index, int(round(prob * 255.0))))

    if not grouped:
        return []

    mm_parts = []
    ml_values: List[str] = []
    for (base_char, mod_code), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item[0])
        base_positions = [idx for idx, char in enumerate(sequence) if str(char).upper() == base_char]
        if not base_positions:
            continue
        occurrence_index = {pos: idx for idx, pos in enumerate(base_positions)}

        deltas = []
        previous = -1
        for pos, prob in items:
            occurrence = occurrence_index.get(pos)
            if occurrence is None:
                continue
            deltas.append(str(occurrence if previous < 0 else occurrence - previous - 1))
            previous = occurrence
            ml_values.append(str(prob))

        if deltas:
            mm_parts.append(f"{base_char}+{mod_code}.,{','.join(deltas)};")

    if not mm_parts:
        return []

    return [
        f"MM:Z:{''.join(mm_parts)}",
        f"ML:B:C,{','.join(ml_values)}",
    ]


def _result_from_stitched_outputs(
    model,
    stitched_batch_result: Dict[str, object],
    *,
    rna: bool = False,
    mod_threshold: float = 0.5,
) -> Dict[str, object]:
    base_result = _format_basecall_result(model.stride, stitched_batch_result["basecall_attrs"], rna=rna)

    device = next(model.parameters()).device
    single_outputs = _build_single_read_outputs(stitched_batch_result["model_outputs"], device=device)
    mod_prediction = model.predict_mods(single_outputs, mod_threshold=mod_threshold)[0]
    mod_sequence, oriented_sites = _oriented_mod_predictions(mod_prediction, reverse_output=rna)
    aligned_pairs = {
        query_idx: target_idx
        for query_idx, target_idx in _equal_alignment_pairs(mod_sequence, base_result["sequence"])
    }

    mapped_sites = []
    for site in oriented_sites:
        target_idx = aligned_pairs.get(int(site["oriented_index"]))
        if target_idx is None:
            continue
        mapped_sites.append({**site, "base_index": int(target_idx)})

    return {
        **base_result,
        "mods": _build_mod_tags(base_result["sequence"], mapped_sites),
    }


def basecall(
    model,
    reads: Iterable[object],
    chunksize: int = 4000,
    overlap: int = 100,
    batchsize: int = 32,
    reverse: bool = False,
    rna: bool = False,
    mod_threshold: float = 0.5,
) -> Iterator[Tuple[object, Dict[str, object]]]:
    chunks = thread_iter(
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )
    batches = thread_iter(batchify(chunks, batchsize=batchsize))
    batch_results = thread_iter(
        (read_info, _run_model_on_batch(model, batch, reverse=reverse))
        for read_info, batch in batches
    )
    per_read_results = thread_iter(
        (
            read,
            {
                "basecall_attrs": stitch_results(
                    outputs["basecall_attrs"],
                    end - start,
                    chunksize,
                    overlap,
                    model.stride,
                    reverse=False,
                ),
                "model_outputs": stitch_results(
                    outputs["model_outputs"],
                    end - start,
                    chunksize,
                    overlap,
                    model.stride,
                    reverse=False,
                ),
            },
        )
        for ((read, start, end), outputs) in unbatchify(batch_results)
    )
    return thread_iter(
        (
            read,
            _result_from_stitched_outputs(
                model,
                stitched_outputs,
                rna=rna,
                mod_threshold=mod_threshold,
            ),
        )
        for read, stitched_outputs in per_read_results
    )
