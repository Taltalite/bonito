"""
Configurable Bonito model with pluggable backbones.

The module follows the standard Bonito input/output conventions
((N, C, L) tensors in, (T, N, C) score tensors out) and is compatible
with the training/basecalling pipeline. The backbone itself can be
defined in the model config using existing ``bonito.nn`` layer types;
if none is provided a small CNN is used by default.
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import tomllib
import torch
from torchinfo import summary

from bonito.crf.model import CTC_CRF, SeqdistModel
from bonito.nn import Clamp, Convolution, LinearCRFEncoder, Permute, Serial, from_dict


def _infer_feature_size(backbone: torch.nn.Module) -> int:
    """Infer the feature dimension emitted by the backbone."""
    for module in reversed(list(backbone.modules())):
        # Explicit Conv1d or nested Convolution wrappers
        if isinstance(module, torch.nn.Conv1d):
            return module.out_channels
        if isinstance(module, Convolution):
            return module.conv.out_channels
        # Linear-style modules also define an output dimension that is valid
        if isinstance(module, torch.nn.Linear):
            return module.out_features
        if isinstance(module, LinearCRFEncoder):
            return module.linear.in_features
    raise ValueError("Could not infer feature size from backbone; please set a simpler backbone or specify one that ends in Conv1d/Linear.")


def _default_backbone(insize: int, features: int, activation: str = "swish", norm: str = "batchnorm") -> Serial:
    """Fallback CNN backbone kept intentionally simple."""
    return Serial([
        Convolution(insize, features, winlen=5, stride=1, padding=2, bias=True, activation=activation, norm=norm),
        Clamp(min=-1.0, max=1.0),
        Convolution(features, features, winlen=5, stride=2, padding=2, bias=True, activation=activation, norm=norm),
    ])


class Model(SeqdistModel):
    """
    Sequence-distance model with a configurable backbone.

    The backbone is expected to emit ``(N, C, L)`` features.  These are
    converted to ``(T, N, C)`` scores via a ``LinearCRFEncoder`` and the
    usual CTC-CRF sequence distance.
    """

    def __init__(self, config: dict):
        seqdist = CTC_CRF(
            state_len=config["global_norm"]["state_len"],
            alphabet=config["labels"]["labels"],
        )

        backbone: Optional[torch.nn.Module]
        if config.get("backbone"):
            backbone = from_dict(config["backbone"])
        else:
            features = config.get("model", {}).get("features", 64)
            backbone = _default_backbone(
                config["input"]["features"],
                features,
                activation=config.get("model", {}).get("activation", "swish"),
                norm=config.get("model", {}).get("norm", "batchnorm"),
            )

        feature_size = _infer_feature_size(backbone)
        self.feature_stack = Serial([
            backbone,
            Permute([2, 0, 1]),
        ])
        self.base_head = LinearCRFEncoder(
            feature_size,
            seqdist.n_base,
            seqdist.state_len,
            activation=config.get("model", {}).get("head_activation", "tanh"),
            scale=config.get("model", {}).get("scale", 5.0),
            blank_score=config.get("model", {}).get("blank_score"),
            expand_blanks=config.get("model", {}).get("expand_blanks", True),
        )

        encoder = Serial([
            self.feature_stack,
            self.base_head,
        ])

        super().__init__(
            encoder,
            seqdist,
            n_pre_post_context_bases=config["input"].get("n_pre_post_context_bases"),
        )
        self.extra_heads = self._build_extra_heads(config, feature_size)
        self.head_labels = {name: head_cfg["labels"] for name, head_cfg in self._head_configs(config).items()}
        self.config = config

    @staticmethod
    def _head_configs(config: dict) -> Dict[str, dict]:
        heads = {}
        for head_cfg in config.get("model", {}).get("heads", []):
            name = head_cfg["name"]
            heads[name] = head_cfg
        return heads

    def _build_extra_heads(self, config: dict, feature_size: int) -> torch.nn.ModuleDict:
        head_layers: Dict[str, torch.nn.Module] = {}
        for head_cfg in config.get("model", {}).get("heads", []):
            name = head_cfg["name"]
            labels = head_cfg["labels"]
            head_layers[name] = torch.nn.Linear(feature_size, len(labels))
        return torch.nn.ModuleDict(head_layers)

    def forward(self, x, return_all: bool = False):
        features = self.feature_stack(x)
        base_scores = self.base_head(features)
        if not (return_all and self.extra_heads):
            return base_scores
        outputs = {name: head(features) for name, head in self.extra_heads.items()}
        outputs["basecall"] = base_scores
        return outputs


def _load_config(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Inspect configurable Bonito model with optional multi-head outputs.")
    parser.add_argument("config", help="Path to the model TOML config.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch dimension for the torchinfo summary.")
    parser.add_argument("--signal-length", type=int, default=4000, help="Signal length to use for the summary input shape.")
    parser.add_argument(
        "--return-all-heads",
        action="store_true",
        help="Show all configured heads (basecall + modification/classification heads) in the summary output.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    model = Model(config)
    input_shape = (args.batch_size, config["input"]["features"], args.signal_length)

    summary(
        model,
        input_size=input_shape,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=4,
        device="cpu",
        verbose=1,
        kwargs={"return_all": args.return_all_heads},
    )


if __name__ == "__main__":
    main()
