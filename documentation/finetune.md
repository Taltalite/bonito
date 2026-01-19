# `bonito finetune` 使用指南

本指南说明如何使用 `bonito finetune` 进行微调训练，并展示如何在 `config.toml` 中配置冻结层。

---

## 1. 训练流程

`bonito finetune` 的训练逻辑与 `bonito train` 一致：

1. 加载模型配置与模型权重（如指定 `--pretrained`）。
2. 加载训练数据集（与 `bonito train` 相同的数据格式）。
3. 读取 `config.toml` 中的微调配置，冻结指定层。
4. 启动训练并输出权重与日志。

---

## 2. 使用示例

```bash
bonito finetune <training_directory> \
  --directory /path/to/train_data \
  --config /path/to/config.toml \
  --pretrained /path/to/pretrained_dir \
  --epochs 10 \
  --chunks 100000 \
  --valid-chunks 10000 \
  --batch 64 \
  --device cuda
```

---

## 3. `config.toml` 冻结层示例

在原有 `config.toml` 的基础上，新增 `finetune.freeze` 配置即可。示例：

```toml
[finetune.freeze]
# 冻结输入卷积层（如果模型中存在 conv 或 encoder.conv）
conv = true

# 冻结整个 encoder（如果模型中存在 encoder_layers 或 encoder.transformer_encoder）
encoder = true

# 只冻结前 N 个 encoder layers（当 encoder = false 时生效）
encoder_layers = 2

# 额外冻结指定模块路径（点号分隔）
modules = ["decoder", "encoder.transformer_encoder.3"]

# 通过参数名精确冻结
parameter_names = ["decoder.linear.weight", "decoder.linear.bias"]

# 通过通配符批量冻结参数
parameter_patterns = ["encoder_layers.*.self_attn.*", "conv.*"]
```

> 提示：`modules` 使用 `model` 属性路径进行解析，路径不正确会在启动时输出提示信息。
