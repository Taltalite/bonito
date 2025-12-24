"""
Simple CNN basecaller model for pluggable training/inspection.

Required TOML keys:
- [model] file (points to this file), hidden_size, kernel_size, stride
- [input] features, n_pre_post_context_bases (optional)
- [labels] labels
- [global_norm] state_len
"""

import torch

from bonito.crf.model import CTC_CRF, SeqdistModel, get_stride
from bonito.nn import LinearCRFEncoder, Permute


class SimpleCNNBasecallerEncoder(torch.nn.Module):
    def __init__(self, in_features, hidden_size, kernel_size, stride, n_base, state_len):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv1d(in_features, hidden_size, kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=stride, padding=padding)
        self.conv3 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.act = torch.nn.SiLU()
        self.permute = Permute([2, 0, 1])
        self.encoder = LinearCRFEncoder(hidden_size, n_base, state_len, activation="tanh")

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.permute(x)
        return self.encoder(x)


class Model(SeqdistModel):
    def __init__(self, config):
        labels = list(config["labels"]["labels"])
        if not labels or labels[0] != "":
            labels = ["", *labels]
        seqdist = CTC_CRF(
            state_len=config["global_norm"]["state_len"],
            alphabet=labels,
        )

        model_cfg = config["model"]
        encoder = SimpleCNNBasecallerEncoder(
            in_features=config["input"]["features"],
            hidden_size=model_cfg.get("hidden_size", 64),
            kernel_size=model_cfg.get("kernel_size", 5),
            stride=model_cfg.get("stride", 2),
            n_base=seqdist.n_base,
            state_len=seqdist.state_len,
        )

        super().__init__(
            encoder,
            seqdist,
            n_pre_post_context_bases=config["input"].get("n_pre_post_context_bases"),
        )
        self.stride = get_stride(encoder)
        self.config = config
