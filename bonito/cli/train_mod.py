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
from bonito.util import default_config, load_symbol, init


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
