# `bonito train_mod` 使用指南（多头 Transformer）

本指南说明如何为 `bonito train_mod` 准备数据、配置模型，以及训练时会生成哪些输出文件。

> 适用场景：Shared-Encoder + Multi-Head 模型（Base Head + Mod Head），
> 使用 `bonito train_mod` 进行训练。

---

## 1. 训练数据准备

`bonito train_mod` 使用与 `bonito train` 相同的基础数据格式，但需要额外的修饰标签文件。

### 1.1 目录结构

```
/path/to/train_data/
├── chunks.npy
├── references.npy
├── reference_lengths.npy
├── mod_targets.npy
└── validation/              # 可选：如果不存在，会从训练集切分
    ├── chunks.npy
    ├── references.npy
    ├── reference_lengths.npy
    └── mod_targets.npy
```

### 1.2 文件说明

| 文件名 | 作用 | 形状/类型说明 |
|---|---|---|
| `chunks.npy` | 原始信号 chunk | `float32`，形状约为 `[N, chunk_length]` |
| `references.npy` | 碱基序列标签（CTC targets） | `int64`，形状 `[N, max_target_len]` |
| `reference_lengths.npy` | 每条序列的实际长度 | `int64`，形状 `[N]` |
| `mod_targets.npy` | 修饰标签 | `int64`，形状 `[N, max_target_len]` |

#### `mod_targets.npy` 要求
- **长度对齐**：`mod_targets[i]` 的长度必须和 `references[i]` 对齐（同一条 read 的碱基级标签）。
- **多标签 or 多类别**：
  - **二分类/多标签（`num_mod_classes=1`）**：值为 `0/1`，表示无修饰 / 有修饰。
  - **多类别（`num_mod_classes>1`）**：值为 `0..K-1` 的类别索引。
- **mask 行为**：`reference_lengths.npy` 控制每条样本有效长度，多余部分会被 mask 掉。

> 说明：如果你的数据不在 npy 格式，而是在自定义数据管线中，
> 也可通过 `dataset.py` 自定义 Loader 输出 `(chunks, targets, lengths, mod_targets)`。

---

## 2. 模型配置

`train_mod` 使用 TOML 配置加载模型，并支持 `model.file` 指向一个独立的模型文件。

示例：`bonito/models/configs/multihead_transformer.toml`

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

[global_norm]
state_len = 4
```

关键字段：
- `model.file`：模型文件路径（相对于 TOML 目录解析）。
- `num_mod_classes`：修饰类别数。
- `mod_task`：`multilabel` 或 `multiclass`（影响 loss 计算）。
- `mod_loss_weight`：修饰 loss 的权重。

---

## 3. 启动训练

基础命令：

```bash
bonito train_mod <training_directory> \
  --directory /path/to/train_data \
  --config /path/to/multihead_transformer.toml \
  --epochs 30 \
  --chunks 100000 \
  --valid-chunks 10000 \
  --batch 64 \
  --device cuda
```

参数说明：
- `training_directory`：输出目录（保存权重、日志、config）。
- `--directory`：训练数据目录（含 `chunks.npy` 等）。
- `--config`：模型与训练配置文件。
- `--epochs`：训练轮数。
- `--chunks`：每个 epoch 训练 chunks 数。
- `--valid-chunks`：验证 chunks 数。
- `--batch`：batch size。
- `--device`：`cuda` / `cpu`。

> 注意：`bonito train_mod` 与 `bonito train` 完全独立，不影响原有训练路径。

---

## 4. 训练输出

训练会在 `<training_directory>` 中生成以下内容：

```
<training_directory>/
├── config.toml               # 合并后的训练配置
├── training.csv              # 每轮训练摘要
├── losses_1.csv              # 每轮训练 loss 记录（可多文件）
├── weights_1.tar             # 每轮模型权重
├── optim_10.tar              # 按 save_optim_every 保存的优化器状态
└── ...
```

### 文件说明
- `config.toml`：包含原始 config + CLI 参数 + 数据集 metadata。
- `training.csv`：每个 epoch 的训练/验证 loss 与 accuracy 汇总。
- `losses_*.csv`：更细粒度的 loss 曲线数据。
- `weights_*.tar`：模型权重。
- `optim_*.tar`：优化器状态（如果开启保存）。

---

## 5. 常见注意事项

1. **mod_targets 必须与 references 对齐**：否则 loss 计算会报 shape 错误。
2. **labels 必须包含 blank**：第一个 label 应为空字符串 `""`。
3. **`num_mod_classes=1`** 时按二分类处理；大于 1 时按多分类处理。
4. **GPU 优先**：多头模型计算量较大，建议使用 `cuda`。

---

如需进一步调试模型或数据管线，可结合 `bonito inspect` 查看模型结构。
