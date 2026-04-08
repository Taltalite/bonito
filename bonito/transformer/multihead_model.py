"""
Multi-head Transformer model with native Bonito CRF basecalling and per-base modification outputs.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Tuple

import edlib
import torch
import torch.nn.functional as F

from bonito.crf.model import CTC_CRF, get_stride
from bonito.nn import LinearCRFEncoder, NamedSerial, TransformerEncoderLayer, from_dict


_CIGAR_RE = re.compile(r"(\d+)([=XID])")
DEFAULT_MOD_BASES = ("A", "C", "G", "T")
DEFAULT_MOD_GLOBAL_LABELS = (
    "canonical_A",
    "canonical_C",
    "canonical_G",
    "canonical_T",
    "m6A",
)
DEFAULT_MOD_HEAD_DEFS = {
    "A": ["canonical_A", "m6A"],
    "C": ["canonical_C"],
    "G": ["canonical_G"],
    "T": ["canonical_T"],
}
DEFAULT_BASE_SLOT_ALIASES = {
    "T": ["T", "U"],
}
IGNORE_INDEX = -100
STANDALONE_MOD_HEAD_MODE = "standalone_mod_head"
DEFAULT_ALIGNMENT_CACHE_CAPACITY = 262144


class LightweightModBlock(torch.nn.Module):
    def __init__(self, width: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"mod_trunk_kernel_size must be a positive odd integer, got {kernel_size}")

        padding = kernel_size // 2
        self.norm = torch.nn.LayerNorm(width)
        self.linear = torch.nn.Linear(width, width)
        self.depthwise = torch.nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=padding,
            groups=width,
        )
        self.pointwise = torch.nn.Conv1d(width, width, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear(x))
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return residual + x


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

        self.mod_task = model_cfg.get("mod_task", "multilabel")
        self.mod_loss_weight = model_cfg.get("mod_loss_weight", 1.0)
        self.mod_target_projection = model_cfg.get("mod_target_projection", "viterbi_edlib_equal")
        self.mod_decode_projection = model_cfg.get("mod_decode_projection", "viterbi_path")
        if self.mod_target_projection != "viterbi_edlib_equal":
            raise ValueError(f"Unsupported mod_target_projection: {self.mod_target_projection}")
        if self.mod_decode_projection != "viterbi_path":
            raise ValueError(f"Unsupported mod_decode_projection: {self.mod_decode_projection}")

        self.mod_bases = [str(base).upper() for base in model_cfg.get("mod_bases", DEFAULT_MOD_BASES)]
        self.mod_global_labels = list(model_cfg.get("mod_global_labels", DEFAULT_MOD_GLOBAL_LABELS))
        if not self.mod_global_labels:
            raise ValueError("model.mod_global_labels must not be empty")
        self.global_label_to_id = {label: idx for idx, label in enumerate(self.mod_global_labels)}
        if len(self.global_label_to_id) != len(self.mod_global_labels):
            raise ValueError("model.mod_global_labels must be unique")

        configured_aliases = model_cfg.get("base_slot_aliases", {})
        self.base_slot_aliases = {base: [base] for base in self.mod_bases}
        for base, aliases in DEFAULT_BASE_SLOT_ALIASES.items():
            if base in self.base_slot_aliases:
                self.base_slot_aliases[base] = [str(alias).upper() for alias in aliases]
        for base, aliases in configured_aliases.items():
            base_key = str(base).upper()
            if base_key not in self.base_slot_aliases:
                continue
            self.base_slot_aliases[base_key] = [str(alias).upper() for alias in aliases]
        self.base_label_to_slot = {}
        for base, aliases in self.base_slot_aliases.items():
            self.base_label_to_slot[base] = base
            for alias in aliases:
                self.base_label_to_slot[alias] = base
        self.head_name_to_id = {base: idx for idx, base in enumerate(self.mod_bases)}

        configured_head_defs = model_cfg.get("mod_head_defs", {})
        self.mod_head_defs = {}
        self.global_id_to_head_local = {}
        self.head_global_ids = {}
        self.head_display_names = {}
        for base in self.mod_bases:
            canonical_label = self._canonical_label(base)
            head_labels = list(configured_head_defs.get(base, DEFAULT_MOD_HEAD_DEFS.get(base, [canonical_label])))
            if not head_labels:
                head_labels = [canonical_label]
            if head_labels[0] != canonical_label:
                raise ValueError(f"Head {base} must start with canonical label {canonical_label}")

            local_global_ids = []
            for local_idx, label in enumerate(head_labels):
                if label not in self.global_label_to_id:
                    raise ValueError(f"Head {base} uses undefined global mod label: {label}")
                global_id = self.global_label_to_id[label]
                if global_id in self.global_id_to_head_local:
                    prev_base, _ = self.global_id_to_head_local[global_id]
                    raise ValueError(f"Global mod label {label} is assigned to multiple heads: {prev_base}, {base}")
                self.global_id_to_head_local[global_id] = (base, local_idx)
                local_global_ids.append(global_id)

            self.mod_head_defs[base] = head_labels
            self.head_global_ids[base] = local_global_ids
            aliases = self.base_slot_aliases.get(base, [base])
            if base == "T" and "U" in aliases:
                self.head_display_names[base] = "T/U slot"
            else:
                self.head_display_names[base] = base

        for base in self.mod_bases:
            canonical_id = self.global_label_to_id.get(self._canonical_label(base))
            if canonical_id is None:
                raise ValueError(f"Missing canonical global label for base {base}")
            if canonical_id not in self.global_id_to_head_local:
                raise ValueError(f"Canonical global label for base {base} is not assigned to head {base}")

        token_to_head_id = torch.full((len(self.alphabet),), -1, dtype=torch.long)
        for token_value in range(len(self.alphabet)):
            head_name = self._base_slot_for_token(token_value)
            if head_name is not None:
                token_to_head_id[token_value] = self.head_name_to_id[head_name]
        self.register_buffer("token_to_head_id", token_to_head_id, persistent=False)

        global_target_to_head_id = torch.full((len(self.mod_global_labels),), -1, dtype=torch.long)
        global_target_to_local_id = torch.full((len(self.mod_global_labels),), -1, dtype=torch.long)
        for global_id, (head_name, local_idx) in self.global_id_to_head_local.items():
            global_target_to_head_id[global_id] = self.head_name_to_id[head_name]
            global_target_to_local_id[global_id] = int(local_idx)
        self.register_buffer("global_target_to_head_id", global_target_to_head_id, persistent=False)
        self.register_buffer("global_target_to_local_id", global_target_to_local_id, persistent=False)

        self.mod_trunk_dim = int(model_cfg.get("mod_trunk_dim", 128))
        self.mod_trunk_kernel_size = int(model_cfg.get("mod_trunk_kernel_size", 5))
        self.mod_trunk_depth = int(model_cfg.get("mod_trunk_depth", 1))
        self.mod_head_dropout = float(model_cfg.get("mod_head_dropout", 0.1))

        self.mod_input_proj = torch.nn.Linear(d_model, self.mod_trunk_dim)
        self.mod_trunk = torch.nn.ModuleList([
            LightweightModBlock(self.mod_trunk_dim, self.mod_trunk_kernel_size, self.mod_head_dropout)
            for _ in range(max(self.mod_trunk_depth, 0))
        ])
        self.mod_heads = torch.nn.ModuleDict({
            base: torch.nn.Linear(self.mod_trunk_dim, len(self.mod_head_defs[base]))
            for base in self.mod_bases
        })

        self.stride = stride
        self.config = config
        self.training_mode = str(config.get("training", {}).get("mode", "")).strip()
        self.standalone_mod_head = self.training_mode == STANDALONE_MOD_HEAD_MODE
        self.alignment_cache_capacity = max(
            int(config.get("training", {}).get("alignment_cache_capacity", DEFAULT_ALIGNMENT_CACHE_CAPACITY)),
            0,
        )
        self._alignment_cache = OrderedDict()
        if self.standalone_mod_head:
            self._freeze_basecaller_parameters()
            self._set_basecaller_eval()

    @staticmethod
    def _resolve_state_len(config, pretrained_encoder_cfg):
        if pretrained_encoder_cfg and "crf" in pretrained_encoder_cfg:
            return pretrained_encoder_cfg["crf"].get("state_len", config.get("global_norm", {}).get("state_len", 5))
        return config.get("global_norm", {}).get("state_len", 5)

    @staticmethod
    def _canonical_label(base: str) -> str:
        return f"canonical_{base}"

    def _base_label(self, token_value: int) -> str | None:
        if token_value <= 0 or token_value >= len(self.alphabet):
            return None
        return str(self.alphabet[int(token_value)]).upper()

    def _base_slot_for_label(self, base_label: str | None) -> str | None:
        if base_label is None:
            return None
        return self.base_label_to_slot.get(str(base_label).upper())

    def _base_slot_for_token(self, token_value: int) -> str | None:
        return self._base_slot_for_label(self._base_label(token_value))

    def _basecalling_modules(self):
        modules = []
        for attr_name in ("encoder", "conv", "encoder_layers", "crf"):
            module = getattr(self, attr_name, None)
            if module is not None:
                modules.append(module)
        return modules

    @staticmethod
    def _freeze_module(module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_basecaller_parameters(self) -> None:
        for module in self._basecalling_modules():
            self._freeze_module(module)

    def _set_basecaller_eval(self) -> None:
        for module in self._basecalling_modules():
            module.eval()

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    @staticmethod
    def _mod_branch_prefixes() -> Tuple[str, ...]:
        return ("mod_input_proj.", "mod_trunk.", "mod_heads.")

    def mod_branch_state_dict(self):
        prefixes = self._mod_branch_prefixes()
        return OrderedDict(
            (name, value)
            for name, value in self.state_dict().items()
            if name.startswith(prefixes)
        )

    def load_mod_branch_state_dict(self, state_dict):
        current_state = self.state_dict()
        expected = set(self.mod_branch_state_dict().keys())
        matched = set()
        unexpected = []
        shape_mismatches = []

        for name, value in state_dict.items():
            if name not in expected:
                unexpected.append(name)
                continue
            if current_state[name].shape != value.shape:
                shape_mismatches.append(f"{name}: expected {tuple(current_state[name].shape)} got {tuple(value.shape)}")
                continue
            current_state[name] = value
            matched.add(name)

        missing = sorted(expected - matched)
        if unexpected or shape_mismatches or missing:
            parts = []
            if unexpected:
                parts.append(f"unexpected={unexpected}")
            if shape_mismatches:
                parts.append(f"shape_mismatches={shape_mismatches}")
            if missing:
                parts.append(f"missing={missing}")
            raise ValueError("Invalid standalone mod-head checkpoint: " + "; ".join(parts))

        self.load_state_dict(current_state)

    def checkpoint_state_dict(self):
        if self.standalone_mod_head:
            return self.mod_branch_state_dict()
        return self.state_dict()

    def load_checkpoint_state_dict(self, state_dict):
        if self.standalone_mod_head:
            self.load_mod_branch_state_dict(state_dict)
            return
        self.load_state_dict(state_dict)

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

    def _decoded_base_predictions(self, outputs) -> List[Dict[str, object]]:
        cached = outputs.get("_decoded_base_predictions")
        if cached is not None:
            return cached

        with torch.no_grad():
            paths = self._decode_paths(outputs["base_scores"])

        predictions = []
        for path in paths:
            emit_positions = torch.nonzero(path != 0, as_tuple=False).flatten()
            emit_tokens = path.index_select(0, emit_positions).to(torch.long)
            predictions.append({
                "emit_positions": emit_positions,
                "emit_tokens": emit_tokens,
                "sequence": self._tokens_to_string(emit_tokens),
            })

        outputs["_decoded_base_predictions"] = predictions
        return predictions

    def _sample_keys(self, outputs):
        cached = outputs.get("_sample_keys_cpu")
        if cached is not None:
            return cached

        sample_keys = outputs.get("sample_keys")
        if sample_keys is None:
            return None
        cached = tuple(int(value) for value in sample_keys.detach().to("cpu").tolist())
        outputs["_sample_keys_cpu"] = cached
        return cached

    def _alignment_cache_enabled(self, outputs, mod_targets, include_site_records: bool) -> bool:
        return (
            self.standalone_mod_head
            and mod_targets is not None
            and not include_site_records
            and self.alignment_cache_capacity > 0
            and outputs.get("sample_keys") is not None
        )

    def _alignment_cache_get(self, sample_key: int):
        entry = self._alignment_cache.get(sample_key)
        if entry is None:
            return None
        self._alignment_cache.move_to_end(sample_key)
        return entry

    def _alignment_cache_put(self, sample_key: int, entry) -> None:
        if self.alignment_cache_capacity <= 0:
            return
        self._alignment_cache[sample_key] = entry
        self._alignment_cache.move_to_end(sample_key)
        while len(self._alignment_cache) > self.alignment_cache_capacity:
            self._alignment_cache.popitem(last=False)

    def _head_probabilities(self, head_name: str, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return torch.ones_like(logits, dtype=torch.float32)
        return torch.softmax(logits, dim=-1)

    def _head_predictions(self, head_name: str, logits: torch.Tensor, mod_threshold: float = 0.5) -> torch.Tensor:
        num_classes = logits.shape[-1]
        if num_classes == 1:
            return logits.new_zeros((logits.shape[0],), dtype=torch.int64)

        probs = torch.softmax(logits, dim=-1)
        if num_classes == 2:
            modified_probs = probs[:, 1]
            return torch.where(
                modified_probs >= mod_threshold,
                torch.ones_like(modified_probs, dtype=torch.int64),
                torch.zeros_like(modified_probs, dtype=torch.int64),
            )
        return probs.argmax(dim=-1)

    def _prediction_scores(self, probs: torch.Tensor, local_preds: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            return probs
        if probs.shape[-1] == 1:
            return probs.squeeze(-1)
        return probs.gather(1, local_preds.unsqueeze(-1)).squeeze(-1)

    def _local_to_global_id(self, head_name: str, local_idx: int) -> int:
        return self.head_global_ids[head_name][int(local_idx)]

    def _global_to_local(self, global_id: int) -> Tuple[str, int] | None:
        return self.global_id_to_head_local.get(int(global_id))

    def _empty_head_projection(self, outputs) -> Dict[str, Dict[str, torch.Tensor]]:
        device = outputs["base_scores"].device
        projections = {}
        for head_name in self.mod_bases:
            num_classes = len(self.mod_head_defs[head_name])
            projections[head_name] = {
                "flat_logits": torch.zeros((0, num_classes), device=device, dtype=torch.float32),
                "flat_targets": torch.zeros((0,), device=device, dtype=torch.long),
                "flat_global_targets": torch.zeros((0,), device=device, dtype=torch.long),
            }
        return projections

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

    def forward(self, x, *extra) -> Dict[str, torch.Tensor]:
        features, base_scores = self._encode_features_and_base_scores(x)
        mod_hidden = self.mod_input_proj(features)
        for block in self.mod_trunk:
            mod_hidden = block(mod_hidden)
        mod_logits_by_base = {
            base: head(mod_hidden)
            for base, head in self.mod_heads.items()
        }
        outputs = {
            "base_scores": base_scores,
            "mod_features": mod_hidden,
            "mod_logits_by_base": mod_logits_by_base,
        }
        if extra:
            outputs["sample_keys"] = extra[0]
        return outputs

    def decode_batch(self, outputs):
        return [sample["sequence"] for sample in self._decoded_base_predictions(outputs)]

    def predict_mods(self, outputs, mod_threshold: float = 0.5) -> List[Dict[str, object]]:
        predictions = []
        mod_logits_by_base = outputs["mod_logits_by_base"]
        for sample_idx, sample in enumerate(self._decoded_base_predictions(outputs)):
            emit_tokens = sample["emit_tokens"].detach().cpu().tolist()
            emit_positions = sample["emit_positions"].detach().cpu().tolist()
            site_predictions = []

            for record_idx, token_value in enumerate(emit_tokens):
                base_label = self._base_label(int(token_value))
                head_name = self._base_slot_for_label(base_label)
                if head_name is None:
                    continue

                logits = mod_logits_by_base[head_name][sample_idx, emit_positions[record_idx]].detach().to(torch.float32).unsqueeze(0)
                probs = self._head_probabilities(head_name, logits)
                local_preds = self._head_predictions(head_name, logits, mod_threshold=mod_threshold)
                local_pred = int(local_preds.item())
                global_pred_id = self._local_to_global_id(head_name, local_pred)
                score = float(self._prediction_scores(probs, local_preds).item())

                site_predictions.append({
                    "base_label": base_label,
                    "head_name": self.head_display_names[head_name],
                    "global_pred_id": global_pred_id,
                    "global_pred_label": self.mod_global_labels[global_pred_id],
                    "local_pred_id": local_pred,
                    "local_probs": probs.squeeze(0).detach().cpu().tolist(),
                    "score": score,
                    "emit_position": int(emit_positions[record_idx]),
                })

            predictions.append({
                "sequence": sample["sequence"],
                "emit_positions": emit_positions,
                "emit_tokens": emit_tokens,
                "base_labels": [self._base_label(int(token)) for token in emit_tokens],
                "sites": site_predictions,
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
        per_head_logits = {base: [] for base in self.mod_bases}
        per_head_targets = {base: [] for base in self.mod_bases}
        per_head_global_targets = {base: [] for base in self.mod_bases}
        sample_records: List[Dict[str, object]] = []
        site_records: List[Dict[str, object]] = []
        remaining_site_slots = None if site_record_limit is None else max(int(site_record_limit), 0)
        cache_enabled = self._alignment_cache_enabled(outputs, mod_targets, include_site_records)
        sample_keys = self._sample_keys(outputs) if cache_enabled else None
        predictions = None

        for sample_idx in range(target_lengths.shape[0]):
            target_len = int(target_lengths[sample_idx].item())
            cached_entry = self._alignment_cache_get(sample_keys[sample_idx]) if sample_keys is not None else None
            if cached_entry is None:
                if predictions is None:
                    predictions = self._decoded_base_predictions(outputs)
                sample = predictions[sample_idx]
                target_tokens = targets[sample_idx, :target_len].to(torch.long)
                target_seq = self._tokens_to_string(target_tokens)
                equal_pairs = self._equal_alignment_pairs(sample["sequence"], target_seq)
                aligned_equal = len(equal_pairs)
                pred_len = int(sample["emit_tokens"].shape[0])
                valid_target_mod_sites = 0
                aligned_valid_mod_sites = 0
                cached_heads = {}

                if mod_targets is not None:
                    valid_target_mod_sites = int((mod_targets[sample_idx, :target_len] != IGNORE_INDEX).sum().item())

                if equal_pairs and mod_targets is not None:
                    sample_device = outputs["base_scores"].device
                    pred_indices = torch.tensor([query_idx for query_idx, _ in equal_pairs], device=sample_device, dtype=torch.long)
                    target_indices = torch.tensor([target_idx for _, target_idx in equal_pairs], device=targets.device, dtype=torch.long)
                    matched_mod_targets = mod_targets[sample_idx, :target_len].index_select(0, target_indices)
                    matched_target_tokens = target_tokens.index_select(0, target_indices)
                    valid_mask = matched_mod_targets != IGNORE_INDEX
                    aligned_valid_mod_sites = int(valid_mask.sum().item())
                    if aligned_valid_mod_sites > 0:
                        valid_pred_indices = pred_indices[valid_mask]
                        valid_target_indices = target_indices[valid_mask]
                        valid_targets = matched_mod_targets[valid_mask].to(torch.long)
                        valid_target_tokens = matched_target_tokens[valid_mask].to(torch.long)
                        valid_time_indices = sample["emit_positions"].to(sample_device).index_select(0, valid_pred_indices)
                        valid_head_ids = self.token_to_head_id.index_select(0, valid_target_tokens)
                        valid_target_head_ids = self.global_target_to_head_id.index_select(0, valid_targets)
                        valid_local_targets = self.global_target_to_local_id.index_select(0, valid_targets)

                        for head_name, head_id in self.head_name_to_id.items():
                            keep_mask = (valid_head_ids == head_id) & (valid_target_head_ids == head_id)
                            if not bool(keep_mask.any()):
                                continue

                            selected_pred_indices = valid_pred_indices[keep_mask]
                            selected_target_indices = valid_target_indices[keep_mask]
                            selected_time_indices = valid_time_indices[keep_mask]
                            selected_global_targets = valid_targets[keep_mask]
                            selected_local_targets = valid_local_targets[keep_mask]
                            selected_logits = outputs["mod_logits_by_base"][head_name][sample_idx].to(torch.float32).index_select(0, selected_time_indices)

                            per_head_logits[head_name].append(selected_logits)
                            per_head_targets[head_name].append(selected_local_targets)
                            per_head_global_targets[head_name].append(selected_global_targets)
                            cached_heads[head_name] = {
                                "time_indices": selected_time_indices.detach().to("cpu"),
                                "local_targets": selected_local_targets.detach().to("cpu"),
                                "global_targets": selected_global_targets.detach().to("cpu"),
                            }

                            if include_site_records and (remaining_site_slots is None or remaining_site_slots > 0):
                                probs = self._head_probabilities(head_name, selected_logits)
                                local_preds = self._head_predictions(head_name, selected_logits, mod_threshold=mod_threshold)
                                scores = self._prediction_scores(probs, local_preds)
                                emit_positions = sample["emit_positions"].detach().cpu().tolist()
                                emit_tokens = sample["emit_tokens"].detach().cpu().tolist()
                                target_token_values = target_tokens.detach().cpu().tolist()
                                selected_pred_indices_list = selected_pred_indices.detach().cpu().tolist()
                                selected_target_indices_list = selected_target_indices.detach().cpu().tolist()
                                true_global_targets = selected_global_targets.detach().cpu().tolist()
                                pred_local_values = local_preds.detach().cpu().tolist()
                                score_values = scores.detach().cpu().tolist()

                                for record_idx, (pred_index, target_index) in enumerate(zip(selected_pred_indices_list, selected_target_indices_list)):
                                    if remaining_site_slots == 0:
                                        break
                                    pred_global_id = self._local_to_global_id(head_name, pred_local_values[record_idx])
                                    site_records.append({
                                        "sample_index_in_batch": sample_idx,
                                        "predicted_base_index": int(pred_index),
                                        "time_step": int(emit_positions[pred_index]),
                                        "target_pos": int(target_index),
                                        "pred_base": self._base_label(int(emit_tokens[pred_index])),
                                        "ref_base": self._base_label(int(target_token_values[target_index])),
                                        "head_name": self.head_display_names[head_name],
                                        "true_mod": int(true_global_targets[record_idx]),
                                        "true_mod_label": self.mod_global_labels[int(true_global_targets[record_idx])],
                                        "pred_mod": int(pred_global_id),
                                        "pred_mod_label": self.mod_global_labels[int(pred_global_id)],
                                        "score": float(score_values[record_idx]),
                                    })
                                    if remaining_site_slots is not None:
                                        remaining_site_slots -= 1
                                        if remaining_site_slots == 0:
                                            break

                sample_record = {
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
                }
                cached_entry = {
                    "per_head": cached_heads,
                    "sample_record": {k: v for k, v in sample_record.items() if k != "sample_index_in_batch"},
                }
                if sample_keys is not None:
                    self._alignment_cache_put(sample_keys[sample_idx], cached_entry)
            else:
                sample_record = {
                    "sample_index_in_batch": sample_idx,
                    **cached_entry["sample_record"],
                }
                sample_device = outputs["base_scores"].device
                for head_name, head_cache in cached_entry["per_head"].items():
                    time_indices = head_cache["time_indices"].to(sample_device)
                    local_targets = head_cache["local_targets"].to(sample_device)
                    global_targets = head_cache["global_targets"].to(sample_device)
                    selected_logits = outputs["mod_logits_by_base"][head_name][sample_idx].to(torch.float32).index_select(0, time_indices)
                    per_head_logits[head_name].append(selected_logits)
                    per_head_targets[head_name].append(local_targets)
                    per_head_global_targets[head_name].append(global_targets)

            sample_records.append(sample_record)

        head_projections = self._empty_head_projection(outputs)
        for head_name in self.mod_bases:
            if per_head_logits[head_name]:
                head_projections[head_name]["flat_logits"] = torch.cat(per_head_logits[head_name], dim=0)
                head_projections[head_name]["flat_targets"] = torch.cat(per_head_targets[head_name], dim=0)
                head_projections[head_name]["flat_global_targets"] = torch.cat(per_head_global_targets[head_name], dim=0)

        return {
            "per_head": head_projections,
            "sample_records": sample_records,
            "site_records": site_records,
        }

    def loss(self, outputs, targets, target_lengths, mod_targets):
        base_scores = outputs["base_scores"]
        if self.standalone_mod_head:
            base_loss = base_scores.new_zeros((), dtype=torch.float32)
        else:
            base_loss = self.seqdist.ctc_loss(base_scores.to(torch.float32), targets, target_lengths)

        projection = self.align_predictions_to_targets(outputs, targets, target_lengths, mod_targets)
        weighted_mod_loss = base_scores.new_zeros((), dtype=torch.float32)
        contributing_sites = 0

        for head_name, head_projection in projection["per_head"].items():
            flat_logits = head_projection["flat_logits"]
            flat_targets = head_projection["flat_targets"]
            if flat_targets.numel() == 0 or flat_logits.shape[-1] <= 1:
                continue
            head_loss = F.cross_entropy(flat_logits, flat_targets.long())
            site_count = int(flat_targets.numel())
            weighted_mod_loss = weighted_mod_loss + (head_loss * site_count)
            contributing_sites += site_count

        if contributing_sites == 0:
            mod_loss = base_scores.new_zeros(())
        else:
            mod_loss = weighted_mod_loss / contributing_sites

        if self.standalone_mod_head:
            total_loss = self.mod_loss_weight * mod_loss
        else:
            total_loss = base_loss + (self.mod_loss_weight * mod_loss)
        return {
            "loss": base_loss,
            "base_loss": base_loss,
            "mod_loss": mod_loss,
            "total_loss": total_loss,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.standalone_mod_head:
            self._set_basecaller_eval()
        return self


Model = MultiHeadModel
