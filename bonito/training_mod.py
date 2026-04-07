"""
Bonito training for multi-head models with base + modification targets.
"""

import math
import os
from itertools import islice
from time import perf_counter
from datetime import datetime

from bonito.schedule import linear_warmup_cosine_decay
from pathlib import Path

from bonito.util import permute, decode_ref, accuracy, tqdm_environ, load_object, match_names, strip_module_prefix
import bonito

import torch
import numpy as np
from tqdm import tqdm
import torch.amp as amp


class ClipGrad:
    def __init__(self, quantile=0.5, factor=2.0, buffer_size=100):
        self.buffer = np.full(buffer_size, fill_value=1e6)
        self.quantile = quantile
        self.factor = factor
        self.i = 0

    def append(self, grad_norm):
        self.buffer[self.i] = grad_norm
        self.i = (self.i + 1) % len(self.buffer)

    def __call__(self, parameters):
        max_norm = self.factor * np.quantile(self.buffer, self.quantile)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm).item()
        if not math.isnan(grad_norm):
            self.append(grad_norm)
        return grad_norm


def load_state(dirname, device, model, optim=None):
    """
    Load a model state dict from disk, supporting standalone mod-head checkpoints.
    """
    dirname = Path(dirname)
    model.to(device)
    if hasattr(model, "module"):
        model = model.module
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod

    to_load = [("weights", model)]
    weight_files = dirname.glob("weights_*.tar")
    weight_nos = {int(w.stem.split("_")[-1]) for w in weight_files}

    if optim is not None:
        to_load.append(("optim", optim))
        optim_files = dirname.glob("optim_*.tar")
        optim_nos = {int(w.stem.split("_")[-1]) for w in optim_files}
        weight_no = max(optim_nos & weight_nos, default=None)
    else:
        weight_no = max(weight_nos, default=None)

    if weight_no is None:
        return 0

    for name, obj in to_load:
        print(f"[loading state] - {name}_{weight_no}.tar")
        state_dict = torch.load(
            dirname / f"{name}_{weight_no}.tar",
            map_location=device,
            weights_only=False,
        )
        if name == "weights":
            state_dict = strip_module_prefix(state_dict)
            if hasattr(obj, "load_checkpoint_state_dict"):
                obj.load_checkpoint_state_dict(state_dict)
            else:
                state_dict = {
                    k2.replace('module.', ''): state_dict[k1]
                    for k1, k2 in match_names(state_dict, obj).items()
                }
                obj.load_state_dict(state_dict)
        else:
            obj.load_state_dict(state_dict)

    return weight_no


class TrainerMod:
    def __init__(
        self, model, device, train_loader, valid_loader, criterion=None,
        use_amp=True, lr_scheduler_fn=None, restore_optim=False,
        save_optim_every=10, grad_accum_split=1, quantile_grad_clip=False,
        chunks_per_epoch=None, batch_size=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or model.loss
        self.use_amp = use_amp
        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.save_optim_every = save_optim_every
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.optimizer = None
        if quantile_grad_clip:
            self.clip_grad = ClipGrad()
        else:
            self.clip_grad = lambda parameters: torch.nn.utils.clip_grad_norm_(parameters, max_norm=2.0).item()

        self.batch_size = batch_size
        self.chunks_per_epoch = chunks_per_epoch
        self.steps_per_epoch = chunks_per_epoch // batch_size

    def _trainable_parameters(self):
        if hasattr(self.model, "trainable_parameters"):
            params = list(self.model.trainable_parameters())
        else:
            params = [param for param in self.model.parameters() if param.requires_grad]
        return params

    def train_one_step(self, batch):
        self.optimizer.zero_grad()
        losses = None

        for batch_ in zip(
            *map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)
        ):
            data_, targets_, lengths_, mod_targets_, *args = (x.to(self.device) for x in batch_)

            with amp.autocast("cuda", enabled=self.use_amp):
                outputs_ = self.model(data_, *args)
                losses_ = self.criterion(outputs_, targets_, lengths_, mod_targets_)

            if not isinstance(losses_, dict):
                losses_ = {'loss': losses_}

            total_loss = losses_.get('total_loss', losses_['loss']) / self.grad_accum_split
            self.scaler.scale(total_loss).backward()

            losses = {
                k: ((v.item() / self.grad_accum_split) if losses is None else (v.item() / self.grad_accum_split) + losses[k])
                for k, v in losses_.items()
            }

        scale = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        grad_norm = self.clip_grad(self._trainable_parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return losses, grad_norm, scale

    def train_one_epoch(self, loss_log, lr_scheduler):
        t0 = perf_counter()
        chunks = 0
        self.model.train()

        progress_bar = tqdm(
            total=self.steps_per_epoch, desc='[0/{}]'.format(self.chunks_per_epoch),
            ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]',
            **tqdm_environ()
        )
        smoothed_losses = None

        with progress_bar:

            for batch in islice(self.train_loader, self.steps_per_epoch):
                chunks += batch[0].shape[0]
                losses, grad_norm, scale = self.train_one_step(batch)

                if smoothed_losses is None:
                    smoothed_losses = dict(losses)
                else:
                    smoothed_losses = {
                        key: (0.01 * value + 0.99 * smoothed_losses[key])
                        for key, value in losses.items()
                    }

                postfix = {"loss": "%.4f" % smoothed_losses.get("loss", 0.0)}
                if "mod_loss" in smoothed_losses:
                    postfix["mod_loss"] = "%.4f" % smoothed_losses["mod_loss"]
                if "total_loss" in smoothed_losses:
                    postfix["total_loss"] = "%.4f" % smoothed_losses["total_loss"]
                progress_bar.set_postfix(postfix)
                progress_bar.set_description("[{}/{}]".format(chunks, self.chunks_per_epoch))
                progress_bar.update()

                if loss_log is not None:
                    lr = lr_scheduler.get_last_lr()
                    if len(lr) == 1: lr = lr[0]
                    loss_log.append({
                        'chunks': chunks,
                        'time': perf_counter() - t0,
                        'grad_norm': grad_norm,
                        'lr': lr,
                        'scale': scale,
                        **losses
                    })

                if lr_scheduler is not None: lr_scheduler.step()

        return smoothed_losses or {}, perf_counter() - t0

    def validate_one_step(self, batch):
        data, targets, lengths, mod_targets, *args = batch
        with amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(data.to(self.device), *(x.to(self.device) for x in args))
            losses = self.criterion(outputs, targets.to(self.device), lengths.to(self.device), mod_targets.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()

        if hasattr(self.model, 'decode_batch'):
            seqs = self.model.decode_batch(outputs)
        else:
            base_logits = outputs["base_logits"]
            seqs = [self.model.decode(x) for x in permute(base_logits, 'NTC', 'TNC')]

        refs = [decode_ref(target, self.model.alphabet) for target in targets]
        n_pre = getattr(self.model, "n_pre_context_bases", 0)
        n_post = getattr(self.model, "n_post_context_bases", 0)
        if n_pre > 0 or n_post > 0:
            refs = [ref[n_pre:len(ref)-n_post] for ref in refs]

        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]
        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            seqs, refs, accs, losses = zip(*(self.validate_one_step(batch) for batch in self.valid_loader))
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss_metrics = {}
        if all(isinstance(x, dict) for x in losses):
            for key in losses[0]:
                loss_metrics[key] = np.mean([x[key] for x in losses])
        else:
            loss_metrics["loss"] = np.mean(losses)
        return loss_metrics, np.mean(accs), np.median(accs)

    def init_optimizer(self, lr, **optim_kwargs):
        if "package" in optim_kwargs:
            optim_cls = load_object(optim_kwargs.pop('package'), optim_kwargs.pop('symbol'))
        else:
            optim_cls = torch.optim.AdamW

        print(f"[loading optim] - '{optim_cls.__name__}' with args: {optim_kwargs}")
        optim_kwargs["lr"] = lr
        trainable_parameters = self._trainable_parameters()
        if not trainable_parameters:
            raise ValueError("No trainable parameters were found for train_mod.")
        self.optimizer = optim_cls(trainable_parameters, **optim_kwargs)

    def get_lr_scheduler(self, epochs, last_epoch=0):
        return self.lr_scheduler_fn(self.optimizer, self.steps_per_epoch, epochs, last_epoch)

    def fit(self, workdir, epochs=1, lr=2e-3, **optim_kwargs):
        if self.optimizer is None:
            self.init_optimizer(lr, **optim_kwargs)

        last_epoch = load_state(workdir, self.device, self.model, self.optimizer if self.restore_optim else None)

        if self.restore_optim:
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["initial_lr"] = pg["lr"] = lr[i] if isinstance(lr, (list, tuple)) else lr

        lr_scheduler = self.get_lr_scheduler(epochs, last_epoch=last_epoch)

        for epoch in range(1 + last_epoch, epochs + 1):
            try:
                with bonito.io.CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                    train_losses, duration = self.train_one_epoch(loss_log, lr_scheduler)

                if hasattr(self.model, "module"):
                    checkpoint_model = self.model.module
                elif hasattr(self.model, "_orig_mod"):
                    checkpoint_model = self.model._orig_mod
                else:
                    checkpoint_model = self.model
                if hasattr(checkpoint_model, "checkpoint_state_dict"):
                    model_state = checkpoint_model.checkpoint_state_dict()
                else:
                    model_state = checkpoint_model.state_dict()
                torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))
                if epoch % self.save_optim_every == 0:
                    torch.save(self.optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))

                val_losses, val_mean, val_median = self.validate_one_epoch()
            except KeyboardInterrupt:
                break

            val_loss = val_losses.get("loss", 0.0)
            val_mod_loss = val_losses.get("mod_loss", None)
            val_total_loss = val_losses.get("total_loss", None)
            parts = [f"loss={val_loss:.4f}"]
            if val_mod_loss is not None:
                parts.append(f"mod_loss={val_mod_loss:.4f}")
            if val_total_loss is not None:
                parts.append(f"total_loss={val_total_loss:.4f}")
            parts.append(f"mean_acc={val_mean:.3f}% median_acc={val_median:.3f}%")
            print("[epoch {}] directory={} {}".format(epoch, workdir, " ".join(parts)))

            with bonito.io.CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                train_loss = train_losses.get("loss", None)
                train_base_loss = train_losses.get("base_loss", None)
                train_mod_loss = train_losses.get("mod_loss", None)
                train_total_loss = train_losses.get("total_loss", None)
                val_base_loss = val_losses.get("base_loss", None)
                training_log.append({
                    "time": datetime.today(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_base_loss": train_base_loss,
                    "train_mod_loss": train_mod_loss,
                    "train_total_loss": train_total_loss,
                    "train_duration": duration,
                    "val_loss": val_loss,
                    "val_base_loss": val_base_loss,
                    "val_mod_loss": val_mod_loss,
                    "val_total_loss": val_total_loss,
                    "val_mean": val_mean,
                    "val_median": val_median,
                })

