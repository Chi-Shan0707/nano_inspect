# PLAN.md 实施状态与未完成事项

核查日期：2026-11-04。**主体实现和实验已经完成；收尾事项已于 2026-11-04 处理（见下方各项状态）。**

本文件依据当前代码、运行产物和本次复跑结果编写；不以验收清单中的勾选作为完成证明。

## 已核实完成

- **M0 基线与模型来源**：ModelScope 完整快照、固定版本与文件哈希已保存；本地基线和断网复跑的输入、生成结果及选定 logits 完全一致。见 [基线对比](runs/baseline-comparison.json)。
- **M1 架构清点**：已生成完整模型清单、权重/共享存储/文本路径/MTP 参数核算、两类 block 图和多长度 cache 测量。见 [架构报告](runs/inventory/report.html)。
- **M2 计算追踪**：已生成两类 mixer 的内部操作、数值示例、残差/输出影响和容差校准；保存的 25 项 BF16 检查及 9 项 FP32 校准检查全部通过。见 [追踪证据](runs/trace/trace.json) 和 [报告](runs/trace/report.html)。
- **M3 干预研究**：检索、activation patching、steering 三组研究均有留出样本、逐例结果、对照与报告；patching 的 10 项控制、steering 的 6 项控制均通过。见 [实验报告](runs/experiment/report.html) 和 [实验元数据](runs/experiment/experiment.json)。
- **本次测试**：在 `nano-interpretability` 环境运行完整测试集，结果为 **26 passed in 35.42s**，包括真实本地模型的 trace 与小规模三组实验集成测试，无跳过项。
- **进程状态**：本次检查未发现仍在运行的本任务下载、安装、推理或测试进程。

## 未完成 TODO

### 1. 补齐环境重建验收证据，修正文档中的过度声明

- [x] 补齐或改正 `docs/acceptance.md` 引用的 `runs/repro/fresh-env-verification.json`：已删除该失效引用，改为链接实际存在的 `runs/repro/bootstrap.log` 与 `runs/repro/baseline/comparison.json`。
- [x] ~~为 README 中“重建后的 package sets 完全一致”提供逐包对比的机器可读证据。~~ 按决定不再重建环境；改为修正 README 的过度声明：重建日志显示 setuptools 为 81.0.0（专用环境为 78.1.0，`requirements.lock` 未固定 setuptools），故“完全一致”不成立。现有 `runs/repro/bootstrap.log` 和 [复现基线对比](runs/repro/baseline/comparison.json) 能证明另一个新环境安装成功、断网推理结果一致，但没有保存完整逐包比较结果。
- [x] 明确记录环境使用偏差（已写入 `docs/acceptance.md` 的 Rebuild 条目；README 已去掉 `ENV_NAME=...` 引导）：`PLAN.md` 要求所有执行使用一个专用环境；现有重建日志和 `runs/repro/doctor.json` 显示曾使用 `nano-interpretability-repro`，而不是约定的 `nano-interpretability`。该临时环境当前已不在 conda 环境列表中。后续验收应在约定环境中完成，避免 README 的 `ENV_NAME=...` 示例继续引导使用其他环境。

完成标准：环境使用说明与真实记录一致，复现所需的锁文件、安装结果、包版本比较、断网基线比较均有有效证据链接。

### 2. 修复验收材料中的失效链接与缺失报告

- [x] 将 `docs/acceptance.md`、`docs/findings.md` 中的 `src/nanoinspect/experiments/patching.py` 链接修正为实际实现位置。该文件不存在；patching 实际由 [retrieval.py](src/nanoinspect/experiments/retrieval.py) 中的 `study_b` 实现。
- [x] 生成 `docs/acceptance.md` 已引用但当前不存在的 `runs/baseline-offline/report.html`（已生成）：

  ```bash
  conda run -n nano-interpretability nanoinspect report --run runs/baseline-offline --compare runs/baseline
  ```

- [x] 修复后重新检查所有本地文档/报告链接：14 个文件、155 个本地链接，0 个失效（`runs/final-check.json`）。

### 3. 完成最终清理与验收记录

- [x] 清理本任务生成的 Python/pytest 缓存，并核对临时下载、测试目录与临时环境没有残留（已清理并记录于 `runs/final-check.json`）。
- [x] 保存本次完整测试结果（`runs/tests.log`：38 passed，无跳过）及最终清理检查记录（`runs/final-check.json`），并据实际证据更新 [验收清单](docs/acceptance.md)。
- [x] 上述事项完成后，才将整个 `implement PLAN.md` 目标标记为完成；已保留 `models/`、`runs/`、专用 conda 环境和最终报告。

## 已知限制，不作为本阶段新增 TODO

- 数值容差校准覆盖短提示；跨多个 64-token chunk 的误差规律尚未系统研究，现有文档已披露。
- steering 对目标行为有效，但正向较大强度显著损害无关问答质量；这是已报告的实验结果，不是缺少质量评估。
- path patching、逐 head 干预及进一步解耦 steering 方向属于报告提出的后续研究，不是原计划本阶段的交付缺口。

## 新增：本地交互工作台

- [x] `nanoinspect serve`（默认 http://127.0.0.1:8765）：架构、logit lens、mixer 内部（注意力图 / gated-delta 状态）、activation patching、steering、报告浏览。见 README 的 *Interactive workbench* 一节。
