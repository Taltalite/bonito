#!/usr/bin/env python3

"""
Bonito training for multi-head (base + modification) models.
"""

import os
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


def load_pretrained_encoder_config(pretrained):
    dirname = resolve_model_dir(pretrained)
    pretrain_file = os.path.join(dirname, "config.toml")
    if not os.path.exists(pretrain_file):
        raise FileNotFoundError(f"Pretrained config not found: {pretrain_file}")
    pretrained_config = toml.load(pretrain_file)
    return pretrained_config.get("model", {}).get("encoder")


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
    pretrained_encoder = config.get("model", {}).get("pretrained_encoder")
    if pretrained_encoder is None:
        pretrained_encoder = load_pretrained_encoder_config(args.pretrained)
        if not pretrained_encoder:
            raise ValueError(
                "train_mod requires a pretrained basecaller with model.encoder in config.toml "
                "so standalone mod-head training can reconstruct the frozen encoder."
            )
        config.setdefault("model", {})["pretrained_encoder"] = pretrained_encoder

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
    return parser
