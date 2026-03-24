# `bonito train_mod`

This command trains a shared-encoder multi-head model that predicts:

- a basecalled sequence from `base_logits`
- base-aligned modification labels from `mod_logits`

Use it for models such as [`bonito/transformer/multihead_model.py`](../bonito/transformer/multihead_model.py).

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
- For binary modification detection with `num_mod_classes = 1`:
  - use `0` for unmodified
  - use `1` for modified
- For multi-class modification detection with `num_mod_classes > 1`:
  - use class ids in `[0, num_mod_classes - 1]`
  - use `-100` for ignored positions

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
num_mod_classes = 1
mod_task = "multilabel"
mod_loss_weight = 1.0

[input]
features = 1
n_pre_post_context_bases = [3, 1]

[labels]
labels = ["", "A", "C", "G", "T"]
```

Notes:

- `model.file` is resolved relative to the config file directory.
- `num_mod_classes = 1` selects binary modification loss.
- `num_mod_classes > 1` selects multi-class modification loss.
- `mod_loss_weight` scales the modification loss contribution.

## Training Command

```bash
bonito train_mod runs/multihead_r9 \
  --directory /path/to/train_data \
  --config bonito/models/configs/multihead_transformer.toml \
  --epochs 30 \
  --chunks 100000 \
  --valid-chunks 10000 \
  --batch 64 \
  --device cuda
```

Optional warm start from a pretrained basecaller:

```bash
bonito train_mod runs/multihead_r9 \
  --directory /path/to/train_data \
  --config bonito/models/configs/multihead_transformer.toml \
  --pretrained dna_r10.4.1@v5.0 \
  --freeze-conv \
  --freeze-encoder-layers 2
```

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
