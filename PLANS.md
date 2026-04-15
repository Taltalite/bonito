# basecaller_mod 主任务规划（阶段一审计结论）

## 阶段一范围

本阶段只做代码审计与方案规划，不改动业务代码。当前结论基于以下现状：

- `bonito train_mod` 已能训练共享 encoder 的 multi-head 模型。
- `validate/` 已能对 `.npy` 数据集和 `pod5` 输入做离线评估/推理分析。
- 仓库内目前**没有** `bonito basecaller_mod` 命令入口。

本文件用于明确：

1. 现有 `basecaller` pipeline 中哪些代码可以直接复用。
2. 为形成 `basecaller_mod` 最小可运行架构，必须补齐哪些能力。
3. 后续完整主任务的实施阶段与顺序。

---

## 一、现有 `basecaller` pipeline 审计

### 1. CLI 与主调度入口

- `bonito/__init__.py`
  - 当前注册的子命令包含 `basecaller`、`train_mod`，**不包含** `basecaller_mod`。
- `bonito/cli/basecaller.py`
  - 这是当前从 `pod5` 到 `{fastq,sam,bam,cram}` 的主入口。
  - 负责：
    - `Reader` 选择与读入；
    - `load_model()` 模型加载；
    - `--reference` 触发的 minimap2 对齐；
    - `Writer` / `CTCWriter` 输出；
    - 运行统计与进度显示。

### 2. `basecaller` 已有的可直接复用链路

以下部分对 `basecaller_mod` 都有高复用价值，且应优先复用而不是重写：

- 读入与预处理
  - `bonito/reader.py`
  - `bonito/pod5.py`
  - 现有 `Reader.get_reads()` 已完成 pod5 遍历、trim、normalisation/standardisation、read meta 组装。
- 模型配置与 checkpoint 加载
  - `bonito/util.py: load_model() / _load_model()`
  - 已支持本地目录模型、官方下载模型、`weights_N.tar`、`standalone_mod_head` 重建。
- 对齐链路
  - `bonito/aligner.py`
  - `basecaller` 生成的结果字典会经 `align_map()` 补上 `mapping`，其余字段可保留。
- 输出链路
  - `bonito/io.py: Writer`
  - `Writer` 已预留 `res.get("mods", [])`，说明输出层已经能接收修饰标签并写入 SAM/BAM/CRAM/FASTQ。

### 3. `basecaller` 当前对 `koi` 的依赖方式

- `bonito/cli/basecaller.py`
  - `--use-koi/--no-use-koi` 已存在。
  - 默认值是 `use_koi=True`。
- `bonito/util.py:_load_model()`
  - 如果 `use_koi=True`：
    - 先按 `model.stride` 修正 `chunksize` 和 `overlap`；
    - 再调用 `model.use_koi(...)`。
- 现有可正常支持 `koi` 的模型：
  - `bonito/crf/model.py: Model.use_koi()`
  - `bonito/transformer/model.py` 通过 `MethodType` 注入 `use_koi()`

### 4. 当前 multi-head 模型与 `basecaller` 的关键不兼容点

#### 4.1 `MultiHeadModel` 没有 `use_koi()`

- 文件：`bonito/transformer/multihead_model.py`
- 当前 `MultiHeadModel` 没有实现 `use_koi()`。
- 这意味着如果直接走 `load_model(..., use_koi=True)`，会在 `_load_model()` 调用 `model.use_koi(...)` 时失败。
- 这也是为什么 `validate/predict_mods_from_pod5.py` 当前把 `--use-koi` 设计成默认关闭，而不是沿用 `basecaller` 的默认开启行为。

#### 4.2 `basecaller` 依赖模型模块导出 `basecall` 符号，但 multi-head 模型没有

- `bonito/cli/basecaller.py` 使用 `load_symbol(args.model_directory, "basecall")`。
- 现有普通 transformer 模型的 `bonito/transformer/basecall.py` 直接复用 `bonito.crf.basecall.basecall`。
- 但 `bonito/transformer/multihead_model.py` 只定义了 `Model`/`MultiHeadModel`，**没有** `basecall`。
- 结论：
  - 不能把现有 `basecaller` 无修改地直接指向 multi-head 训练目录。
  - 后续必须新增 `basecaller_mod` 自己的推理主流程，或给 multi-head 模型模块补一个兼容的 `basecall` 入口。

#### 4.3 当前 pod5 修饰推理脚本与正式 basecaller 输出链路还未打通

- `validate/predict_mods_from_pod5.py` 已经具备：
  - pod5 读入；
  - chunk/batch/stitch；
  - `model.predict_mods()`；
  - basecall TSV / FASTA 与 mod-site TSV 输出。
- 但它没有复用 `Writer`，也没有生成 `MM/ML` SAM tags。
- 因此它更像“验证脚本”，不是正式 CLI pipeline。

#### 4.4 当前 base 序列与 mod 序列并不严格共用同一解码路径

- `validate/predict_mods_from_pod5.py` 里：
  - basecalling 结果来自 `koi.decode.beam_search`；
  - mod 站点预测来自 `model.predict_mods()`，其内部使用的是 `MultiHeadModel.decode_batch()` 的 Viterbi 路径。
- 脚本里甚至保留了两套序列字段：
  - `sequence`
  - `mod_sequence`
- 这说明当前代码默认接受“base 输出序列”和“mod 投影序列”不完全一致。
- 对真正的 `basecaller_mod` 而言，这是核心架构问题：
  - 若要给正式 basecalling 结果写修饰标签，必须定义清楚“mod 位点到底挂在哪一条输出序列上”。

### 5. 文档与实现存在历史落差

- `documentation/SAM.md` 提到 `--modified-bases` 会输出 `MM/ML` tags。
- 但当前 CLI 里并没有这个开关，也没有实际的 mod-tag 生成逻辑。
- 说明仓库历史上已有“修饰标签输出”的接口预期，但当前实现并未闭环。

---

## 二、哪些逻辑应复用 `basecaller`，哪些应复用 `validate/`

### 应优先复用 `basecaller` 的部分

- CLI 参数框架与运行日志格式
- `Reader` / `pod5` 读入与信号预处理
- `load_model()` / checkpoint 重建逻辑
- `align_map()` 对齐
- `Writer` 输出 BAM/CRAM/SAM/FASTQ
- 进度条、吞吐统计、read-group 生成

原因：这些部分已经构成正式 basecalling 命令的稳定外壳，`basecaller_mod` 不应重新造一套。

### 应优先复用 `validate/predict_mods_from_pod5.py` 的部分

- 针对 multi-head 输出的 chunk/batch/stitch 处理思路
- `base_scores + mod_logits_by_base` 的联合拼接逻辑
- `model.predict_mods()` 的调用方式
- RNA 输出方向、polyA trim 等推理侧细节

原因：这些逻辑已经证明 multi-head 模型在 pod5 输入上可以跑通修饰推理，但目前还只是验证脚本形态。

### 不建议直接照搬的部分

- `validate/predict_mods_from_pod5.py` 当前“每条 read 额外再跑一次 beam-search basecalling”的做法
  - 这会造成正式 `basecaller_mod` 中的重复推理开销。
- 脚本当前把 basecalling 与 mod decode 绑定到两条不同路径
  - 这对验证可接受，对正式输出不可长期保留。

---

## 三、形成 `basecaller_mod` 最小可运行架构的必要改动

以下是最小闭环所需改动，按必要性排序。

### 必要改动 1：新增 `basecaller_mod` 命令入口

必须新增：

- `bonito/cli/basecaller_mod.py`
- `bonito/__init__.py` 中注册 `basecaller_mod`

原因：

- 当前没有子命令入口；
- 直接复用 `basecaller.py` 不够，因为 multi-head 模型缺少 `basecall` 符号，且还需要追加 mod 输出逻辑。

### 必要改动 2：建立“单次前向 + 双输出”的正式推理链路

`basecaller_mod` 的核心推理流程必须做到：

1. 对每条 read 只跑一套 chunk/batch/stitch 前向；
2. 同时拿到：
   - `base_scores`
   - `mod_logits_by_base`
3. 在这套 stitched 输出上完成：
   - base 序列解码；
   - mod 位点预测；
   - 后续输出封装。

原因：

- 这是从验证脚本升级为正式 basecaller 的最小性能要求；
- 可以避免当前 `validate/predict_mods_from_pod5.py` 的重复模型执行。

### 必要改动 3：定义 basecall 输出序列与 mod 位点的对齐规则

这是最关键的设计点。最小可运行版本至少要固定一种明确策略。

建议的最小策略：

- 仍复用现有 beam-search basecalling 作为最终输出序列；
- `model.predict_mods()` 先在其内部的 Viterbi 发射序列上给出修饰位点；
- 再把 `mod_sequence` 与最终 `basecall sequence` 做一次等号位点投影（可复用现有 edlib equal-pair 思路）；
- 仅把能稳定映射到最终 basecall 序列的位置写入修饰输出。

原因：

- 这是当前代码基础上改动最小、风险最低的闭环方案；
- 能保证正式输出上的修饰标签挂靠在最终导出的 basecall 序列上；
- 后续如果要做到“beam-search 路径直接驱动 mod 位点”，可以再做第二阶段优化。

### 必要改动 4：把 mod 预测转成正式输出字段

至少需要一条正式输出路径：

- 推荐主路径：写入 `Writer` 可接受的 `res["mods"]`，输出 `MM/ML` tags

可选附加路径：

- sidecar TSV/JSON（便于调试）

原因：

- `Writer` 已经能透传 `mods` 标签；
- 这是最小改动复用现有 BAM/SAM 输出架构的方式。

### 必要改动 5：处理 `koi` 兼容策略

这是 `basecaller_mod` 不能回避的点。

最小可运行版本需要先明确行为：

- 若 `MultiHeadModel` 暂未支持 `use_koi()`：
  - 要么在 `basecaller_mod` 中默认关闭并显式提示；
  - 要么 fail-fast，要求用户传 `--no-use-koi`。

完整目标版本则需要：

- 为 `MultiHeadModel` 增加真正可用的 `use_koi()` 支持；
- 保证启用后：
  - basecalling 仍能走高性能 `koi` 解码；
  - mod 分支时间轴不被破坏；
  - stitched 后的 base/mod 时间维度保持一致。

原因：

- 现有 `basecaller` 默认启用 `koi`，这是性能基线；
- 不把这个问题单独拿出来，后面很容易做成“功能能跑但速度明显退化”的假闭环。

---

## 四、推荐的主任务实施阶段

下面是后续真正进入开发时的推荐顺序。

### 阶段二：做出最小可运行 `basecaller_mod`

目标：

- 直接输入 pod5；
- 直接输出 basecalling 结果；
- 能同时给出修饰预测；
- 先以“功能闭环”为第一优先级。

建议子任务：

1. 新增 `bonito/cli/basecaller_mod.py`
2. 复用 `basecaller` 的 CLI/Reader/load_model/aligner/Writer 框架
3. 抽出或内聚一个 multi-head 专用推理函数
   - 输入：`model + reads`
   - 输出：标准 `result` 字典
   - 字段至少包含：
     - `sequence`
     - `qstring`
     - `moves`
     - `mods`
4. 先把修饰结果稳定写入正式输出，哪怕同时保留 sidecar 调试文件
5. 新增最小 CLI 测试

阶段二完成标准：

- `bonito basecaller_mod <model_dir> <pod5_dir>` 能跑通；
- FASTQ/SAM/BAM 至少一种输出格式可用；
- 输出里能看到修饰结果；
- 不改动 `train_mod` 与 `validate/` 原有行为。

### 阶段三：补齐 `koi` 性能路径

目标：

- 让 `basecaller_mod` 在 multi-head 模型上也能像 `basecaller` 一样走 `koi` 优化路径。

建议子任务：

1. 为 `MultiHeadModel` 设计 `use_koi()`
2. 检查启用 `koi` 后：
   - `base_scores` layout
   - `mod_logits_by_base` layout
   - stitch 前后时间轴
   - beam-search 输入格式
3. 统一 `--use-koi/--no-use-koi` 的默认行为与提示
4. 对比启用/关闭 `koi` 的功能一致性与速度差异

阶段三完成标准：

- `basecaller_mod` 默认可启用 `koi`；
- 关闭/开启 `koi` 的输出语义一致；
- 性能有明确改善。

### 阶段四：统一“base 输出序列”和“mod 投影序列”

目标：

- 消除当前验证脚本里的 `sequence` / `mod_sequence` 双轨语义；
- 让修饰预测直接绑定最终导出的 basecall 结果。

建议方向：

- 先保留“Viterbi mod 序列投影到 beam-search 输出序列”的兼容实现；
- 再评估是否需要更进一步，把 mod 位点直接绑定到 beam-search 路径本身。

阶段四完成标准：

- 正式输出中只有一套清晰的序列语义；
- 修饰位点坐标对最终 basecall 结果是单义的。

### 阶段五：验证、文档与回归测试

建议补齐：

- CLI help / 子命令注册测试
- 小规模 smoke test
- 至少一个针对 mod-tag 输出的单元测试或集成测试
- 文档更新：
  - `README.md`
  - `documentation/SAM.md`
  - 新增 `documentation/basecaller_mod.md`（如需要）

---

## 五、后续开发时的最小文件触达面

为控制改动范围，建议优先限制在以下文件：

- `bonito/__init__.py`
- `bonito/cli/basecaller_mod.py`
- `bonito/transformer/multihead_model.py`
- 可能新增一个推理/标签辅助模块，例如：
  - `bonito/transformer/multihead_basecall.py`
  - 或 `bonito/mod_util.py`
- 必要时少量补充：
  - `bonito/io.py`（仅在确实需要新增 mod tag 编码辅助函数时）
  - `test/test_cli.py`
  - 新增 `test/test_basecaller_mod.py`

不建议在第一轮开发中扩散修改到：

- `train_mod` 训练逻辑
- `validate/` 现有评估脚本
- 通用数据生成脚本

除非发现正式推理闭环确实被这些模块阻塞。

---

## 六、阶段一结论

结论可以概括为三点：

1. `basecaller_mod` 完全可以以现有 `basecaller` 为外壳来构建，Reader、load_model、aligner、Writer 这些正式链路都应直接复用。
2. 当前真正缺的不是“修饰模型推理能力”，而是“把 multi-head 推理结果接到正式 basecaller 输出协议上”的中间层，包括：
   - 单次前向联合输出；
   - base/mod 序列对齐规则；
   - `MM/ML` 标签封装；
   - `koi` 兼容。
3. `koi` 不是附属优化点，而是 `basecaller_mod` 最终要补齐的正式性能路径；但从最小可运行架构出发，可以先做功能闭环，再补 `MultiHeadModel.use_koi()`。

