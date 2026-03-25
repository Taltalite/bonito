"""
Multi-head Transformer model with native Bonito CRF basecalling + modification outputs.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from bonito.crf.model import CTC_CRF, get_stride
from bonito.nn import LinearCRFEncoder, NamedSerial, TransformerEncoderLayer, from_dict


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

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features, base_scores = self._encode_features_and_base_scores(x)
        mod_hidden = F.gelu(self.mod_adapter(features))
        mod_logits = self.mod_decoder(mod_hidden)
        return {
            "base_scores": base_scores,
            "mod_logits": mod_logits,
        }

    def decode_batch(self, outputs):
        base_scores = outputs["base_scores"]
        scores = self.seqdist.posteriors(base_scores.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def loss(self, outputs, targets, target_lengths, mod_targets):
        base_scores = outputs["base_scores"]
        mod_logits = outputs["mod_logits"]

        base_loss = self.seqdist.ctc_loss(base_scores.to(torch.float32), targets, target_lengths)

        max_len = mod_targets.size(1)
        mod_logits = mod_logits.permute(0, 2, 1)
        mod_logits = F.interpolate(mod_logits, size=max_len, mode="linear", align_corners=False)
        mod_logits = mod_logits.permute(0, 2, 1)

        length_mask = torch.arange(max_len, device=mod_targets.device)[None, :] < target_lengths[:, None]
        valid_mask = mod_targets != -100
        mask = length_mask & valid_mask

        if self.num_mod_classes == 1:
            mod_targets = mod_targets.float()
            loss_raw = F.binary_cross_entropy_with_logits(mod_logits.squeeze(-1), mod_targets, reduction="none")
            mod_loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1)
        else:
            mod_targets_masked = mod_targets.clone()
            mod_targets_masked[~mask] = -100
            mod_loss = F.cross_entropy(
                mod_logits.reshape(-1, self.num_mod_classes),
                mod_targets_masked.reshape(-1),
                ignore_index=-100,
            )

        total_loss = base_loss + (self.mod_loss_weight * mod_loss)
        return {
            "loss": base_loss,
            "mod_loss": mod_loss,
            "total_loss": total_loss,
        }


Model = MultiHeadModel
