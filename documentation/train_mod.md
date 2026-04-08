# `bonito train_mod`

This command trains a standalone shared-encoder multi-head model that predicts:

- a basecalled sequence from `base_logits`
- base-aligned modification labels from a shared modification trunk plus
  base-specific modification heads

Use it for models such as [`bonito/transformer/multihead_model.py`](../bonito/transformer/multihead_model.py).

## Model Architecture

The current `multihead_model.py` implementation uses:

- one shared encoder for basecalling and modification prediction
- one shared modification trunk that consumes encoder features
- one modification head per base slot listed in `model.mod_bases`

With the default config, the model defines four base slots:

- `A`
- `C`
- `G`
- `T`

and four corresponding heads:

- `A` head: `["canonical_A", "m6A"]`
- `C` head: `["canonical_C"]`
- `G` head: `["canonical_G"]`
- `T` head: `["canonical_T"]`

So in the default m6A setup, only the `A` head has a real supervised
modified-vs-canonical decision. The `C/G/T` heads exist so the model can keep a
consistent global label space and base-slot mapping, but they do not contribute
modification loss unless you configure additional classes for them.

## Directory Layout

Training data can be provided either as numpy files or through a custom `dataset.py`.

Numpy layout:

```text
/path/to/train_data/
|-- chunks.npy
|-- references.npy
|-- reference_lengths.npy
|-- mod_targets.npy
`-- validation/              # optional
    |-- chunks.npy
    |-- references.npy
    |-- reference_lengths.npy
    `-- mod_targets.npy
```

If `validation/` is missing, Bonito splits the training arrays into train/valid subsets.

## Required Array Shapes

The built-in numpy loader expects:

| file | dtype | shape | meaning |
|---|---|---|---|
| `chunks.npy` | `float32` or `float16` | `[N, chunk_length]` | raw signal chunks |
| `references.npy` | integer | `[N, max_target_len]` | padded base labels for CTC |
| `reference_lengths.npy` | integer | `[N]` | true target length for each row in `references.npy` |
| `mod_targets.npy` | integer | `[N, max_target_len]` or `[N, mod_target_len]` with `mod_target_len >= max(reference_lengths)` | padded modification labels aligned to the same target axis |

Constraints checked by the loader:

- all files must share the same first dimension `N`
- `chunks.npy` must be 2D
- `references.npy` must be 2D
- `reference_lengths.npy` must be 1D
- `mod_targets.npy` must be 2D
- `max(reference_lengths.npy)` must not exceed the width of `references.npy`
- `max(reference_lengths.npy)` must not exceed the width of `mod_targets.npy`

## Label Semantics

`references.npy`

- Use integer-encoded base labels.
- Padding uses `0`.
- For the default config, the effective alphabet is `["", "A", "C", "G", "T"]`.
- `reference_lengths.npy[i]` is the number of valid target positions in `references.npy[i]`.

`mod_targets.npy`

- One row per sample, aligned to the target sequence rather than to raw signal time steps.
- Padding or ignored positions should use `-100` if you want them excluded from the modification loss.
- Labels are global modification-label ids, not per-head local ids.
- The available global ids are defined by `model.mod_global_labels`.
- Each base slot maps a subset of those global ids into its own local head space
  through `model.mod_head_defs`.
- A target position contributes modification loss only if:
  - its label is not `-100`
  - the base at that target position maps to a configured base slot
  - that slot's head contains more than one class
  - the global label is assigned to that same slot

For the default m6A config:

- `0` = `canonical_A`
- `1` = `canonical_C`
- `2` = `canonical_G`
- `3` = `canonical_T`
- `4` = `m6A`

Practical interpretation with the default config:

- `A` positions should usually be labeled as either `canonical_A` (`0`) or `m6A` (`4`)
- `C/G/T` positions may be labeled with their canonical global ids if you want a fully
  populated target tensor
- `C/G/T` positions can also be set to `-100` if you want them ignored for an m6A-only task

Important:

- do not use per-head local ids in `mod_targets.npy`
- do not assume every non-ignored target position contributes mod loss
- the valid prefix is still controlled by `reference_lengths.npy`

In the current implementation, `target_lengths` defines the valid prefix for both base and modification targets.

## Custom Dataset Loader

Instead of numpy files, you can place a `dataset.py` beside your data. Its loader must provide batches shaped like:

```python
(chunks, targets, lengths, mod_targets)
```

with the same semantics as the numpy layout above.

## Config Example

Example config:

```toml
[model]
file = "../../transformer/multihead_model.py"
d_model = 256
nhead = 4
dim_feedforward = 1024
num_layers = 4
kernel_size = 5
stride = 2
mod_task = "multilabel"
mod_loss_weight = 1.0
mod_target_projection = "viterbi_edlib_equal"
mod_decode_projection = "viterbi_path"
mod_bases = ["A", "C", "G", "T"]
mod_global_labels = ["canonical_A", "canonical_C", "canonical_G", "canonical_T", "m6A"]
mod_trunk_dim = 128
mod_trunk_kernel_size = 5
mod_trunk_depth = 1
mod_head_dropout = 0.1

[model.mod_head_defs]
A = ["canonical_A", "m6A"]
C = ["canonical_C"]
G = ["canonical_G"]
T = ["canonical_T"]

[model.base_slot_aliases]
T = ["T", "U"]

[input]
features = 1
n_pre_post_context_bases = [0, 0]

[labels]
labels = ["", "A", "C", "G", "T"]

[global_norm]
state_len = 4
```

Notes:

- `model.file` is resolved relative to the config file directory.
- `model.mod_global_labels` defines the shared global label vocabulary.
- `model.mod_head_defs` defines which labels belong to each base-specific head.
- The first label in each head must be that base's canonical label.
- `model.base_slot_aliases` lets one slot accept multiple emitted base symbols, such as
  `T` and `U` sharing the same head.
- Heads with only one class are structurally present but do not contribute cross-entropy loss.
- `mod_loss_weight` scales the modification loss contribution.

## Label Mapping Example

With the default config:

- a reference `A` with canonical label `0` is routed to the `A` head as local class `0`
- a reference `A` with m6A label `4` is routed to the `A` head as local class `1`
- a reference `C` with canonical label `1` is routed to the `C` head as local class `0`
- a reference `C` with label `4` is invalid for the `C` head and is ignored by the mod-loss projection

This means `mod_targets.npy` should be thought of as a target-axis tensor in the
shared global label space, while the model internally converts those labels into
per-head local targets after aligning predicted bases back to the reference.

## Training Command

`train_mod` now supports exactly one workflow:

- `--pretrained` is required
- the pretrained basecaller is frozen
- only the modification branch is trainable

```bash
bonito train_mod runs/multihead_r9 \
  --directory /path/to/train_data \
  --config bonito/models/configs/multihead_transformer.toml \
  --pretrained dna_r10.4.1@v5.0 \
  --epochs 30 \
  --chunks 100000 \
  --valid-chunks 10000 \
  --batch 64 \
  --device cuda
```

`train_mod` always runs in **standalone mod-head training** mode:

- the pretrained basecaller is treated as immutable
- only the modification branch is trainable
- checkpoints store the modification branch weights only
- loading the run later reconstructs the model by combining the recorded pretrained basecaller with the saved mod-head weights
- base decode results are cached per forward output and reused by loss/projection/prediction helpers
- standalone training reuses per-sample alignment/projection results when stable sample keys are available from the built-in numpy loader
- standalone training skips base-loss computation because the frozen basecaller is not optimized
- standalone training prints profiling summaries every `10000` chunks by default, including dataloader/transfer/forward/criterion/backward/optimizer timings and alignment-cache hit rate

Use `--profile-chunks 0` to disable the periodic profiling output or set a different flush interval.

This is the recommended mode when you want to benchmark modification detection against the unchanged official basecaller output.

## Outputs

The training directory will contain files such as:

```text
runs/multihead_r9/
|-- config.toml
|-- training.csv
|-- losses_1.csv
|-- weights_1.tar
|-- weights_2.tar
`-- optim_10.tar
```

`config.toml` now records saved dataset shape metadata for numpy datasets, which is useful when checking whether a run used the expected target width and modification target width.

For standalone mod-head runs, `weights_*.tar` contains only the modification branch state. The run `config.toml` records the pretrained basecaller reference needed to reconstruct the full composite model during evaluation or inference.

## Resume Behavior

- `weights_*.tar` checkpoints can be resumed.
- `--restore-optim` also restores the optimizer state and continues the scheduler from the last saved epoch.

## Practical Size Guidance

There is no hard-coded fixed size, but a reasonable starting point is:

- `chunk_length`: typically a few thousand signal samples per chunk
- `max_target_len`: roughly `chunk_length / model_stride`, adjusted for your labeling pipeline
- `N`: as large as your GPU budget and training objective require

For the provided example config with `stride = 2`, the encoded time axis is about half the raw signal length before any target-side padding or trimming choices in your dataset pipeline.

The most important rule is consistency:

- every sample must have one signal chunk
- one base target row
- one target length
- one modification target row aligned to the same base target axis
