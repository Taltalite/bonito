"""
Multi-head Transformer model with base + modification outputs.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from bonito.crf.model import get_stride
from bonito.nn import NamedSerial, TransformerEncoderLayer, from_dict


def _ctc_greedy_decode(logits, labels):
    preds = logits.argmax(dim=-1)
    sequences = []
    for pred in preds:
        prev = None
        seq = []
        for idx in pred.tolist():
            if idx != prev and idx != 0:
                seq.append(labels[idx])
            prev = idx
        sequences.append("".join(seq))
    return sequences


class MultiHeadModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        labels = list(config["labels"]["labels"])
        if not labels or labels[0] != "":
            labels = ["", *labels]
        self.alphabet = labels
        self.n_pre_context_bases, self.n_post_context_bases = config["input"].get("n_pre_post_context_bases", (0, 0))

        model_cfg = config["model"]
        pretrained_encoder_cfg = model_cfg.get("pretrained_encoder")
        if pretrained_encoder_cfg:
            encoder = from_dict(pretrained_encoder_cfg)
            if isinstance(encoder, NamedSerial) and "crf" in encoder._modules:
                encoder = NamedSerial({name: layer for name, layer in encoder.named_children() if name != "crf"})
            self.encoder = encoder
            stride = get_stride(self.encoder)
            d_model = (
                pretrained_encoder_cfg.get("upsample", {}).get("d_model")
                or pretrained_encoder_cfg.get("transformer_encoder", {}).get("layer", {}).get("d_model")
                or model_cfg.get("d_model", 256)
            )
        else:
            d_model = model_cfg.get("d_model", 256)
            nhead = model_cfg.get("nhead", 4)
            dim_feedforward = model_cfg.get("dim_feedforward", 1024)
            num_layers = model_cfg.get("num_layers", 4)
            kernel_size = model_cfg.get("kernel_size", 5)
            stride = model_cfg.get("stride", 2)

        self.num_mod_classes = model_cfg.get("num_mod_classes", 1)
        self.mod_task = model_cfg.get("mod_task", "multilabel")
        self.mod_loss_weight = model_cfg.get("mod_loss_weight", 1.0)

        if not pretrained_encoder_cfg:
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

        self.base_decoder = torch.nn.Linear(d_model, len(self.alphabet))
        self.mod_adapter = torch.nn.Linear(d_model, d_model // 2)
        self.mod_decoder = torch.nn.Linear(d_model // 2, self.num_mod_classes)

        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        self.stride = stride
        self.config = config

    def encode(self, x):
        if hasattr(self, "encoder"):
            x = self.encoder(x)
        else:
            x = self.conv(x)
            x = x.permute(0, 2, 1)
            for layer in self.encoder_layers:
                x = layer(x)
        return x

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.encode(x)
        base_logits = self.base_decoder(features)
        mod_hidden = F.gelu(self.mod_adapter(features))
        mod_logits = self.mod_decoder(mod_hidden)
        return {
            "base_logits": base_logits,
            "mod_logits": mod_logits,
        }

    def decode_batch(self, outputs):
        base_logits = outputs["base_logits"]
        return _ctc_greedy_decode(base_logits, self.alphabet)

    def loss(self, outputs, targets, target_lengths, mod_targets):
        base_logits = outputs["base_logits"]
        mod_logits = outputs["mod_logits"]

        log_probs = base_logits.log_softmax(dim=-1).permute(1, 0, 2)
        input_lengths = torch.full(
            (log_probs.size(1),),
            log_probs.size(0),
            device=log_probs.device,
            dtype=torch.long,
        )
        base_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

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
