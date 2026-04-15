#!/usr/bin/env python3

"""
Bonito training for multi-head (base + modification) models.
"""

import os
from copy import deepcopy
from pathlib import Path
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import toml
import torch

from bonito.training_mod import TrainerMod
from bonito.data import ModelSetup, ComputeSettings, DataSettings
from bonito.train_mod_data import load_train_mod_data
from bonito.util import STANDALONE_MOD_HEAD_MODE, load_pretrained_weights, load_symbol, init, resolve_model_dir


train_mod_default_config = Path(__file__).resolve().parent.parent / "models/configs/multihead_transformer.toml"
RUNTIME_CONFIG_KEYS = (
    "basecaller",
    "scaling",
    "standardisation",
    "normalisation",
    "run_info",
    "qscore",
)


def load_pretrained_config(pretrained):
    dirname = resolve_model_dir(pretrained)
    pretrain_file = os.path.join(dirname, "config.toml")
    if not os.path.exists(pretrain_file):
        raise FileNotFoundError(f"Pretrained config not found: {pretrain_file}")
    return toml.load(pretrain_file)


def load_pretrained_encoder_config(pretrained):
    pretrained_config = load_pretrained_config(pretrained)
    return pretrained_config.get("model", {}).get("encoder")


def extract_pretrained_state_len(pretrained_config):
    model_cfg = pretrained_config.get("model", {})
    seqdist_cfg = model_cfg.get("seqdist", {})
    if "state_len" in seqdist_cfg:
        return int(seqdist_cfg["state_len"])
    encoder_crf = model_cfg.get("encoder", {}).get("crf", {})
    if "state_len" in encoder_crf:
        return int(encoder_crf["state_len"])
    global_norm = pretrained_config.get("global_norm", {})
    if "state_len" in global_norm:
        return int(global_norm["state_len"])
    return None


def extract_pretrained_labels(pretrained_config):
    labels_cfg = pretrained_config.get("labels", {}).get("labels")
    if labels_cfg is not None:
        return list(labels_cfg)
    seqdist_alphabet = pretrained_config.get("model", {}).get("seqdist", {}).get("alphabet")
    if seqdist_alphabet is not None:
        return list(seqdist_alphabet)
    return None


def extract_pretrained_input_features(pretrained_config):
    input_cfg = pretrained_config.get("input", {})
    if "features" in input_cfg:
        return int(input_cfg["features"])

    conv_sublayers = pretrained_config.get("model", {}).get("encoder", {}).get("conv", {}).get("sublayers", [])
    for layer in conv_sublayers:
        if layer.get("type") == "convolution" and "insize" in layer:
            return int(layer["insize"])
    return None


def merge_pretrained_runtime_config(config, pretrained_config):
    for key in RUNTIME_CONFIG_KEYS:
        if key in pretrained_config:
            config[key] = deepcopy(pretrained_config[key])

    pretrained_labels = extract_pretrained_labels(pretrained_config)
    if pretrained_labels is not None:
        merged_labels = deepcopy(config.get("labels", {}))
        merged_labels["labels"] = list(pretrained_labels)
        config["labels"] = merged_labels

    pretrained_input_features = extract_pretrained_input_features(pretrained_config)
    if pretrained_input_features is not None:
        merged_input = deepcopy(config.get("input", {}))
        merged_input["features"] = int(pretrained_input_features)
        config["input"] = merged_input

    pretrained_state_len = extract_pretrained_state_len(pretrained_config)
    if pretrained_state_len is not None:
        merged_global_norm = deepcopy(config.get("global_norm", {}))
        merged_global_norm["state_len"] = int(pretrained_state_len)
        config["global_norm"] = merged_global_norm

    return config


def validate_pretrained_runtime_config(config, pretrained_config, model=None):
    pretrained_state_len = extract_pretrained_state_len(pretrained_config)
    if pretrained_state_len is None:
        raise ValueError("Unable to resolve pretrained state_len from pretrained config.")

    current_labels = list(config.get("labels", {}).get("labels", []) or [])
    pretrained_labels = extract_pretrained_labels(pretrained_config)
    if pretrained_labels is None:
        raise ValueError("Unable to resolve pretrained labels/alphabet from pretrained config.")
    if current_labels != list(pretrained_labels):
        raise ValueError(
            "train_mod config labels do not match the pretrained basecaller alphabet. "
            f"current={current_labels} pretrained={list(pretrained_labels)}"
        )

    current_features = config.get("input", {}).get("features")
    pretrained_features = extract_pretrained_input_features(pretrained_config)
    if pretrained_features is None:
        raise ValueError("Unable to resolve pretrained input.features from pretrained config.")
    if current_features != pretrained_features:
        raise ValueError(
            "train_mod config input.features does not match the pretrained basecaller. "
            f"current={current_features} pretrained={pretrained_features}"
        )

    pretrained_sample_type = str(pretrained_config.get("run_info", {}).get("sample_type", "")).strip()
    current_sample_type = str(config.get("run_info", {}).get("sample_type", "")).strip()
    if pretrained_sample_type and current_sample_type != pretrained_sample_type:
        raise ValueError(
            "train_mod config run_info.sample_type does not match the pretrained basecaller. "
            f"current={current_sample_type!r} pretrained={pretrained_sample_type!r}"
        )

    scaling_cfg = config.get("scaling")
    if not scaling_cfg:
        raise ValueError("train_mod config is missing scaling copied from the pretrained basecaller.")
    if str(scaling_cfg.get("strategy", "")).strip() == "pa" and not config.get("standardisation"):
        raise ValueError(
            "train_mod config uses scaling.strategy='pa' but standardisation is missing. "
            "Standalone mod-head inference would use incorrect signal normalization."
        )
    if not config.get("standardisation") and not config.get("normalisation"):
        raise ValueError(
            "train_mod config is missing both standardisation and normalisation. "
            "Standalone mod-head inference would fall back to inconsistent runtime behavior."
        )

    if model is not None:
        model_state_len = int(model.seqdist.state_len)
        if model_state_len != pretrained_state_len:
            raise ValueError(
                "Reconstructed multi-head model state_len does not match the pretrained basecaller. "
                f"model={model_state_len} pretrained={pretrained_state_len}"
            )


def main(args):
    workdir = os.path.expanduser(args.training_directory)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)
    os.makedirs(workdir, exist_ok=True)

    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    config = toml.load(args.config)
    config["__config_dir__"] = str(Path(args.config).resolve().parent)
    pretrained_config = load_pretrained_config(args.pretrained)
    pretrained_encoder = config.get("model", {}).get("pretrained_encoder")
    if pretrained_encoder is None:
        pretrained_encoder = pretrained_config.get("model", {}).get("encoder")
        if not pretrained_encoder:
            raise ValueError(
                "train_mod requires a pretrained basecaller with model.encoder in config.toml "
                "so standalone mod-head training can reconstruct the frozen encoder."
            )
        config.setdefault("model", {})["pretrained_encoder"] = pretrained_encoder

    config = merge_pretrained_runtime_config(config, pretrained_config)
    validate_pretrained_runtime_config(config, pretrained_config)

    training_cfg = {
        **config.get("training", {}),
        **vars(args),
        "pwd": os.getcwd(),
        "mode": STANDALONE_MOD_HEAD_MODE,
        "pretrained": args.pretrained,
        "pretrained_basecaller": args.pretrained,
        "pretrained_basecaller_dir": resolve_model_dir(args.pretrained),
        "mod_head_weights_pattern": "weights_{epoch}.tar",
    }
    config["training"] = training_cfg

    print("[loading model]")
    model = load_symbol(config, 'Model')(config)
    validate_pretrained_runtime_config(config, pretrained_config, model=model)
    preload_stats = load_pretrained_weights(model, args.pretrained, device)

    try:
        model = torch.compile(model)
    except RuntimeError as e:
        print(f"[warning] Torch model failed to compile, performance may be degraded. {e}")

    print("[loading data]")
    data = DataSettings(
        training_data=args.directory,
        num_train_chunks=args.chunks,
        num_valid_chunks=args.valid_chunks,
        output_dir=workdir
    )
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=config.get("standardisation", {}),
    )
    compute_settings = ComputeSettings(
        batch_size=args.batch,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    train_loader, valid_loader = load_train_mod_data(data, model_setup, compute_settings)

    try:
        dataset_cfg = train_loader.dataset.dataset_config
    except AttributeError:
        dataset_cfg = {}
    if preload_stats:
        config["training"]["pretrained_weights"] = preload_stats
    toml.dump({**config, **dataset_cfg}, open(os.path.join(workdir, 'config.toml'), 'w'))

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = TrainerMod(
        model, device, train_loader, valid_loader,
        use_amp=not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip,
        chunks_per_epoch=args.chunks,
        batch_size=args.batch,
        profile_flush_chunks=args.profile_chunks,
    )

    if (',' in args.lr):
        lr = [float(x) for x in args.lr.split(',')]
    else:
        lr = float(args.lr)
    optim_kwargs = config.get("optim", {})
    trainer.fit(workdir, args.epochs, lr, **optim_kwargs)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    parser.add_argument('--config', default=str(train_mod_default_config))
    parser.add_argument('--pretrained', required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", type=int, help="Number of training chunks per epoch")
    parser.add_argument("--valid-chunks", type=int, help="Number of validation chunks per epoch", )
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    quantile_group = parser.add_mutually_exclusive_group()
    quantile_group.add_argument('--quantile-grad-clip', dest='quantile_grad_clip', action='store_true')
    quantile_group.add_argument('--no-quantile-grad-clip', dest='quantile_grad_clip', action='store_false')
    quantile_group.set_defaults(quantile_grad_clip=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--profile-chunks", default=10000, type=int, help="Flush training profiling stats every N chunks; set 0 to disable.")
    return parser
