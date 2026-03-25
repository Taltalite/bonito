"""
Multi-head Transformer model with native Bonito CRF basecalling and per-base modification outputs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import edlib
import torch
import torch.nn.functional as F

from bonito.crf.model import CTC_CRF, get_stride
from bonito.nn import LinearCRFEncoder, NamedSerial, TransformerEncoderLayer, from_dict


_CIGAR_RE = re.compile(r"(\d+)([=XID])")


class MultiHeadModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        labels = list(config["labels"]["labels"])
        if not labels:
            raise ValueError("config['labels']['labels'] must not be empty")
        if labels[0] == "":
            labels[0] = "N"
        elif labels[0] in {"A", "C", "G", "T", "U"}:
            labels = ["N", *labels]
        self.alphabet = labels

        model_cfg = config["model"]
        pretrained_encoder_cfg = model_cfg.get("pretrained_encoder")
        state_len = self._resolve_state_len(config, pretrained_encoder_cfg)
        self.seqdist = CTC_CRF(state_len=state_len, alphabet=self.alphabet)

        default_context = (state_len - 1, 1)
        self.n_pre_context_bases, self.n_post_context_bases = config["input"].get(
            "n_pre_post_context_bases", default_context
        )

        if pretrained_encoder_cfg:
            self.encoder = from_dict(pretrained_encoder_cfg)
            self._native_crf_in_encoder = isinstance(self.encoder, NamedSerial) and "crf" in self.encoder._modules
            if self._native_crf_in_encoder:
                crf_layer = self.encoder._modules["crf"]
                if not isinstance(crf_layer, LinearCRFEncoder):
                    raise TypeError("Expected encoder.crf to be a LinearCRFEncoder")
                d_model = crf_layer.linear.in_features
            else:
                d_model = (
                    pretrained_encoder_cfg.get("upsample", {}).get("d_model")
                    or pretrained_encoder_cfg.get("transformer_encoder", {}).get("layer", {}).get("d_model")
                    or model_cfg.get("d_model", 256)
                )
            stride = get_stride(self.encoder)
        else:
            self._native_crf_in_encoder = False
            d_model = model_cfg.get("d_model", 256)
            nhead = model_cfg.get("nhead", 4)
            dim_feedforward = model_cfg.get("dim_feedforward", 1024)
            num_layers = model_cfg.get("num_layers", 4)
            kernel_size = model_cfg.get("kernel_size", 5)
            stride = model_cfg.get("stride", 2)

            padding = kernel_size // 2
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(config["input"]["features"], d_model, kernel_size, stride=stride, padding=padding),
                torch.nn.SiLU(),
            )
            self.encoder_layers = torch.nn.ModuleList([
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    deepnorm_alpha=1.0,
                    deepnorm_beta=1.0,
                    attn_window=model_cfg.get("attn_window"),
                )
                for _ in range(num_layers)
            ])
            self.crf = LinearCRFEncoder(
                insize=d_model,
                n_base=self.seqdist.n_base,
                state_len=state_len,
                bias=model_cfg.get("base_bias", False),
                scale=model_cfg.get("base_scale", 5.0),
                activation=model_cfg.get("base_activation", "tanh"),
                blank_score=model_cfg.get("blank_score", 2.0),
                expand_blanks=model_cfg.get("expand_blanks", True),
                permute=[1, 0, 2],
            )

        self.num_mod_classes = model_cfg.get("num_mod_classes", 1)
        self.mod_task = model_cfg.get("mod_task", "multilabel")
        self.mod_loss_weight = model_cfg.get("mod_loss_weight", 1.0)
        self.mod_target_projection = model_cfg.get("mod_target_projection", "viterbi_edlib_equal")
        self.mod_decode_projection = model_cfg.get("mod_decode_projection", "viterbi_path")
        if self.mod_target_projection != "viterbi_edlib_equal":
            raise ValueError(f"Unsupported mod_target_projection: {self.mod_target_projection}")
        if self.mod_decode_projection != "viterbi_path":
            raise ValueError(f"Unsupported mod_decode_projection: {self.mod_decode_projection}")

        self.mod_adapter = torch.nn.Linear(d_model, d_model // 2)
        self.mod_decoder = torch.nn.Linear(d_model // 2, self.num_mod_classes)

        self.stride = stride
        self.config = config

    def _resolve_state_len(self, config, pretrained_encoder_cfg):
        if pretrained_encoder_cfg and "crf" in pretrained_encoder_cfg:
            return pretrained_encoder_cfg["crf"].get("state_len", config.get("global_norm", {}).get("state_len", 5))
        return config.get("global_norm", {}).get("state_len", 5)

    def _encode_features_and_base_scores(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "encoder"):
            if self._native_crf_in_encoder:
                features = x
                for name, layer in self.encoder.named_children():
                    if name == "crf":
                        break
                    features = layer(features)
                base_scores = self.encoder.crf(features)
                return features, base_scores

            features = self.encoder(x)
            base_scores = self.crf(features)
            return features, base_scores

        features = self.conv(x)
        features = features.permute(0, 2, 1)
        for layer in self.encoder_layers:
            features = layer(features)
        base_scores = self.crf(features)
        return features, base_scores

    @staticmethod
    def _parse_cigar(cigar: str) -> List[Tuple[int, str]]:
        return [(int(count), op) for count, op in _CIGAR_RE.findall(cigar)]

    def _tokens_to_string(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            values = tokens.detach().cpu().tolist()
        else:
            values = list(tokens)
        return "".join(self.alphabet[int(token)] for token in values if int(token) != 0)

    def _decode_paths(self, base_scores: torch.Tensor) -> torch.Tensor:
        scores = self.seqdist.posteriors(base_scores.detach().to(torch.float32)) + 1e-8
        return self.seqdist.viterbi(scores.log()).to(torch.int16).T

    def _per_base_prediction_tensors(self, outputs) -> List[Dict[str, object]]:
        mod_logits = outputs["mod_logits"].to(torch.float32)
        with torch.no_grad():
            paths = self._decode_paths(outputs["base_scores"])

        predictions = []
        for sample_idx, path in enumerate(paths):
            emit_positions = torch.nonzero(path != 0, as_tuple=False).flatten()
            emit_tokens = path.index_select(0, emit_positions).to(torch.long)
            sample_logits = mod_logits[sample_idx].index_select(0, emit_positions)
            predictions.append({
                "emit_positions": emit_positions,
                "emit_tokens": emit_tokens,
                "mod_logits": sample_logits,
                "sequence": self._tokens_to_string(emit_tokens),
            })
        return predictions

    def _equal_alignment_pairs(self, query_seq: str, target_seq: str) -> List[Tuple[int, int]]:
        if not query_seq or not target_seq:
            return []

        result = edlib.align(query_seq, target_seq, mode="NW", task="path")
        cigar = result.get("cigar")
        if not cigar:
            return []

        query_index = 0
        target_index = 0
        pairs: List[Tuple[int, int]] = []
        for count, op in self._parse_cigar(cigar):
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

    def _mod_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_mod_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return torch.softmax(logits, dim=-1)

    def _mod_predictions(self, logits: torch.Tensor, mod_threshold: float = 0.5) -> torch.Tensor:
        if self.num_mod_classes == 1:
            return (torch.sigmoid(logits.squeeze(-1)) >= mod_threshold).to(torch.int64)
        return torch.softmax(logits, dim=-1).argmax(dim=-1)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features, base_scores = self._encode_features_and_base_scores(x)
        mod_hidden = F.gelu(self.mod_adapter(features))
        mod_logits = self.mod_decoder(mod_hidden)
        return {
            "base_scores": base_scores,
            "mod_logits": mod_logits,
        }

    def decode_batch(self, outputs):
        paths = self._decode_paths(outputs["base_scores"])
        return [self.seqdist.path_to_str(path.cpu().numpy()) for path in paths]

    def predict_mods(self, outputs, mod_threshold: float = 0.5) -> List[Dict[str, object]]:
        predictions = []
        for sample in self._per_base_prediction_tensors(outputs):
            logits = sample["mod_logits"].detach().to(torch.float32)
            probs = self._mod_probabilities(logits)
            preds = self._mod_predictions(logits, mod_threshold=mod_threshold)
            emit_tokens = sample["emit_tokens"].detach().cpu().tolist()
            emit_positions = sample["emit_positions"].detach().cpu().tolist()
            predictions.append({
                "sequence": sample["sequence"],
                "emit_positions": emit_positions,
                "emit_tokens": emit_tokens,
                "base_labels": [self.alphabet[int(token)] for token in emit_tokens],
                "mod_logits": logits.cpu().tolist(),
                "mod_probs": probs.cpu().tolist(),
                "mod_preds": preds.cpu().tolist(),
            })
        return predictions

    def align_predictions_to_targets(
        self,
        outputs,
        targets,
        target_lengths,
        mod_targets=None,
        mod_threshold: float = 0.5,
        include_site_records: bool = False,
        site_record_limit: int | None = None,
    ) -> Dict[str, object]:
        predictions = self._per_base_prediction_tensors(outputs)

        flat_logits_chunks: List[torch.Tensor] = []
        flat_target_chunks: List[torch.Tensor] = []
        sample_records: List[Dict[str, object]] = []
        site_records: List[Dict[str, object]] = []
        remaining_site_slots = None if site_record_limit is None else max(int(site_record_limit), 0)

        for sample_idx, sample in enumerate(predictions):
            target_len = int(target_lengths[sample_idx].item())
            target_tokens = targets[sample_idx, :target_len].to(torch.long)
            target_seq = self._tokens_to_string(target_tokens)
            equal_pairs = self._equal_alignment_pairs(sample["sequence"], target_seq)

            aligned_equal = len(equal_pairs)
            pred_len = int(sample["emit_tokens"].shape[0])
            valid_target_mod_sites = 0
            aligned_valid_mod_sites = 0

            if mod_targets is not None:
                valid_target_mod_sites = int((mod_targets[sample_idx, :target_len] != -100).sum().item())

            if equal_pairs:
                pred_indices = torch.tensor([query_idx for query_idx, _ in equal_pairs], device=sample["mod_logits"].device, dtype=torch.long)
                target_indices = torch.tensor([target_idx for _, target_idx in equal_pairs], device=targets.device, dtype=torch.long)
                matched_logits = sample["mod_logits"].index_select(0, pred_indices).to(torch.float32)

                if mod_targets is not None:
                    matched_mod_targets = mod_targets[sample_idx, :target_len].index_select(0, target_indices)
                    valid_mask = matched_mod_targets != -100
                    aligned_valid_mod_sites = int(valid_mask.sum().item())
                    if aligned_valid_mod_sites > 0:
                        valid_logits = matched_logits[valid_mask]
                        valid_targets = matched_mod_targets[valid_mask]
                        flat_logits_chunks.append(valid_logits)
                        flat_target_chunks.append(valid_targets)

                        if include_site_records and (remaining_site_slots is None or remaining_site_slots > 0):
                            if self.num_mod_classes == 1:
                                site_scores = torch.sigmoid(valid_logits.squeeze(-1))
                                site_preds = (site_scores >= mod_threshold).to(torch.int64)
                            else:
                                site_probs = torch.softmax(valid_logits, dim=-1)
                                site_scores = site_probs.max(dim=-1).values
                                site_preds = site_probs.argmax(dim=-1)

                            valid_pred_indices = pred_indices[valid_mask].detach().cpu().tolist()
                            valid_target_indices = target_indices[valid_mask].detach().cpu().tolist()
                            emit_positions = sample["emit_positions"].detach().cpu().tolist()
                            emit_tokens = sample["emit_tokens"].detach().cpu().tolist()
                            target_token_values = target_tokens.detach().cpu().tolist()
                            target_mod_values = valid_targets.detach().cpu().tolist()
                            score_values = site_scores.detach().cpu().tolist()
                            pred_values = site_preds.detach().cpu().tolist()

                            for record_index, (pred_index, target_index) in enumerate(zip(valid_pred_indices, valid_target_indices)):
                                if remaining_site_slots == 0:
                                    break
                                site_records.append({
                                    "sample_index_in_batch": sample_idx,
                                    "predicted_base_index": int(pred_index),
                                    "time_step": int(emit_positions[pred_index]),
                                    "target_pos": int(target_index),
                                    "pred_base": self.alphabet[int(emit_tokens[pred_index])],
                                    "ref_base": self.alphabet[int(target_token_values[target_index])],
                                    "true_mod": int(target_mod_values[record_index]),
                                    "pred_mod": int(pred_values[record_index]),
                                    "score": float(score_values[record_index]),
                                })
                                if remaining_site_slots is not None:
                                    remaining_site_slots -= 1
                                    if remaining_site_slots == 0:
                                        break

            sample_records.append({
                "sample_index_in_batch": sample_idx,
                "predicted_base_len": pred_len,
                "target_len": target_len,
                "aligned_equal_bases": aligned_equal,
                "target_coverage": float(aligned_equal / target_len) if target_len else 0.0,
                "predicted_base_coverage": float(aligned_equal / pred_len) if pred_len else 0.0,
                "valid_target_mod_sites": valid_target_mod_sites,
                "aligned_valid_mod_sites": aligned_valid_mod_sites,
                "valid_mod_coverage": float(aligned_valid_mod_sites / valid_target_mod_sites) if valid_target_mod_sites else 0.0,
                "pred_sequence_prefix": sample["sequence"][:80],
                "target_sequence_prefix": target_seq[:80],
            })

        feature_dim = self.num_mod_classes if self.num_mod_classes > 1 else 1
        if flat_logits_chunks:
            flat_logits = torch.cat(flat_logits_chunks, dim=0)
        else:
            flat_logits = outputs["mod_logits"].new_zeros((0, feature_dim), dtype=torch.float32)

        target_dtype = mod_targets.dtype if mod_targets is not None else targets.dtype
        if flat_target_chunks:
            flat_targets = torch.cat(flat_target_chunks, dim=0)
        else:
            flat_targets = targets.new_zeros((0,), dtype=target_dtype)

        return {
            "flat_logits": flat_logits,
            "flat_targets": flat_targets,
            "sample_records": sample_records,
            "site_records": site_records,
        }

    def loss(self, outputs, targets, target_lengths, mod_targets):
        base_scores = outputs["base_scores"]
        base_loss = self.seqdist.ctc_loss(base_scores.to(torch.float32), targets, target_lengths)

        projection = self.align_predictions_to_targets(outputs, targets, target_lengths, mod_targets)
        flat_logits = projection["flat_logits"]
        flat_targets = projection["flat_targets"]

        if flat_targets.numel() == 0:
            mod_loss = base_scores.new_zeros(())
        elif self.num_mod_classes == 1:
            mod_loss = F.binary_cross_entropy_with_logits(flat_logits.squeeze(-1), flat_targets.float())
        else:
            mod_loss = F.cross_entropy(flat_logits, flat_targets.long())

        total_loss = base_loss + (self.mod_loss_weight * mod_loss)
        return {
            "loss": base_loss,
            "mod_loss": mod_loss,
            "total_loss": total_loss,
        }


Model = MultiHeadModel
