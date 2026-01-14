#!/usr/bin/env python3

"""
Bonito training for multi-head (base + modification) models.
"""

import os
from pathlib import Path
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import toml
import torch

from bonito.training_mod import TrainerMod
from bonito.data import load_mod_data, ModelSetup, ComputeSettings, DataSettings
from bonito.util import __models_dir__, default_config, get_last_checkpoint, load_symbol, init


def load_pretrained_weights(model, pretrained, device):
    dirname = pretrained
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
        dirname = os.path.join(__models_dir__, dirname)

    weights = get_last_checkpoint(dirname)
    print(f"[loading pretrained weights] - {weights}")
    state_dict = torch.load(weights, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    matched = 0
    for name, value in state_dict.items():
        if name in model_state and model_state[name].shape == value.shape:
            model_state[name] = value
            matched += 1

    model.load_state_dict(model_state)
    skipped = len(state_dict) - matched
    print(f"[loading pretrained weights] - matched={matched} skipped={skipped}")
    if matched == 0:
        print("[warning] No pretrained weights matched current model parameters.")
    return {"path": str(weights), "matched": matched, "skipped": skipped}


def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False


def apply_freeze_settings(model, args):
    frozen = []
    if args.freeze_conv and hasattr(model, "conv"):
        freeze_parameters(model.conv)
        frozen.append("conv")

    if hasattr(model, "encoder_layers"):
        if args.freeze_encoder:
            for layer in model.encoder_layers:
                freeze_parameters(layer)
            frozen.append("encoder_layers=all")
        elif args.freeze_encoder_layers > 0:
            for layer in model.encoder_layers[:args.freeze_encoder_layers]:
                freeze_parameters(layer)
            frozen.append(f"encoder_layers=first_{args.freeze_encoder_layers}")

    if frozen:
        print(f"[freezing parameters] - {', '.join(frozen)}")


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
    if args.pretrained:
        preload_stats = load_pretrained_weights(model, args.pretrained, device)
    else:
        preload_stats = {}
    apply_freeze_settings(model, args)

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

    train_loader, valid_loader = load_mod_data(data, model_setup, compute_settings)

    try:
        dataset_cfg = train_loader.dataset.dataset_config
    except AttributeError:
        dataset_cfg = {}
    if preload_stats:
        argsdict["training"]["pretrained_weights"] = preload_stats
    toml.dump({**config, **argsdict, **dataset_cfg}, open(os.path.join(workdir, 'config.toml'), 'w'))

    trainer = TrainerMod(
        model, device, train_loader, valid_loader,
        use_amp=not args.no_amp,
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
    parser.add_argument("--freeze-conv", action="store_true", default=False)
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--freeze-encoder-layers", type=int, default=0)
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
