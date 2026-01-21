#!/usr/bin/env python3

"""
Bonito finetuning.
"""

import os
from fnmatch import fnmatch
from pathlib import Path
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import toml
import torch

from bonito.training import Trainer
from bonito.data import load_data, ModelSetup, ComputeSettings, DataSettings
from bonito.util import __models_dir__, default_config, get_last_checkpoint, load_symbol, init, match_names


def resolve_pretrained_dir(pretrained):
    dirname = os.path.expanduser(pretrained)
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
        dirname = os.path.join(__models_dir__, dirname)
    return dirname


def load_pretrained_weights(model, pretrained, device):
    dirname = resolve_pretrained_dir(pretrained)
    weights = get_last_checkpoint(dirname)
    print(f"[loading pretrained weights] - {weights}")
    state_dict = torch.load(weights, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    matched = 0
    remapped = None

    if set(state_dict.keys()) != set(model_state.keys()):
        try:
            remapped = {
                k2: state_dict[k1]
                for k1, k2 in match_names(state_dict, model).items()
            }
        except AssertionError:
            remapped = None

    to_load = remapped if remapped is not None else state_dict
    for name, value in to_load.items():
        if name in model_state and model_state[name].shape == value.shape:
            model_state[name] = value
            matched += 1

    model.load_state_dict(model_state)
    skipped = len(to_load) - matched
    print(f"[loading pretrained weights] - matched={matched} skipped={skipped}")
    if matched == 0:
        print("[warning] No pretrained weights matched current model parameters.")
    return {"path": str(weights), "matched": matched, "skipped": skipped}


def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False


def resolve_module_path(model, path):
    current = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def freeze_encoder_layers(model, count):
    if count <= 0:
        return None
    if hasattr(model, "encoder_layers"):
        layers = list(model.encoder_layers)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "transformer_encoder"):
        layers = list(model.encoder.transformer_encoder)
    else:
        return None
    for layer in layers[:count]:
        freeze_parameters(layer)
    return f"encoder_layers=first_{count}"


def freeze_all_encoder(model):
    if hasattr(model, "encoder_layers"):
        freeze_parameters(model.encoder_layers)
        return "encoder_layers=all"
    if hasattr(model, "encoder") and hasattr(model.encoder, "transformer_encoder"):
        freeze_parameters(model.encoder.transformer_encoder)
        return "encoder.transformer_encoder=all"
    return None


def freeze_conv(model):
    if hasattr(model, "conv"):
        freeze_parameters(model.conv)
        return "conv"
    if hasattr(model, "encoder") and hasattr(model.encoder, "conv"):
        freeze_parameters(model.encoder.conv)
        return "encoder.conv"
    return None


def apply_freeze_settings(model, config):
    freeze_cfg = config.get("finetune", {}).get("freeze", {})
    if not freeze_cfg:
        return

    frozen = []
    missing = []

    if freeze_cfg.get("conv"):
        frozen_item = freeze_conv(model)
        if frozen_item:
            frozen.append(frozen_item)
        else:
            missing.append("conv")

    if freeze_cfg.get("encoder"):
        frozen_item = freeze_all_encoder(model)
        if frozen_item:
            frozen.append(frozen_item)
        else:
            missing.append("encoder")
    elif freeze_cfg.get("encoder_layers"):
        frozen_item = freeze_encoder_layers(model, int(freeze_cfg["encoder_layers"]))
        if frozen_item:
            frozen.append(frozen_item)
        else:
            missing.append("encoder_layers")

    for module_path in freeze_cfg.get("modules", []) or []:
        module = resolve_module_path(model, module_path)
        if module is None:
            missing.append(module_path)
            continue
        freeze_parameters(module)
        frozen.append(module_path)

    parameter_names = set(freeze_cfg.get("parameter_names", []) or [])
    parameter_patterns = list(freeze_cfg.get("parameter_patterns", []) or [])
    if parameter_names or parameter_patterns:
        for name, param in model.named_parameters():
            if name in parameter_names:
                param.requires_grad = False
                frozen.append(f"param:{name}")
                continue
            if any(fnmatch(name, pattern) for pattern in parameter_patterns):
                param.requires_grad = False
                frozen.append(f"param:{name}")

        unresolved = [name for name in parameter_names if name not in dict(model.named_parameters())]
        missing.extend(unresolved)

    if frozen:
        print(f"[freezing parameters] - {', '.join(dict.fromkeys(frozen))}")
    if missing:
        print(f"[freezing parameters] - missing: {', '.join(dict.fromkeys(missing))}")


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

    argsdict = dict(training=vars(args))
    argsdict["training"]["pwd"] = os.getcwd()

    print("[loading model]")
    model = load_symbol(config, 'Model')(config)
    preload_stats = {}
    if args.pretrained:
        preload_stats = load_pretrained_weights(model, args.pretrained, device)
    apply_freeze_settings(model, config)

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

    train_loader, valid_loader = load_data(data, model_setup, compute_settings)

    try:
        dataset_cfg = train_loader.dataset.dataset_config
    except AttributeError:
        dataset_cfg = {}
    if preload_stats:
        argsdict["training"]["pretrained_weights"] = preload_stats
    toml.dump({**config, **argsdict, **dataset_cfg}, open(os.path.join(workdir, 'config.toml'), 'w'))

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        model, device, train_loader, valid_loader,
        use_amp=not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip,
        chunks_per_epoch=args.chunks,
        batch_size=args.batch,
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
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--pretrained', default="")
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
    return parser
