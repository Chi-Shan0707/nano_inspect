# 阅读架构图谱

下载的 ModelScope 配置标明模型为 `Qwen3_5ForConditionalGeneration`，包含 24 个文本 block 和一个 12 个 block 的 vision tower。在与实际加载的模块和权重清单（inventory）核对之前，这些都只是**配置层面的事实**。checkpoint 中记录的历史 `transformers_version` 并不是当前安装的运行时版本；运行溯源（provenance）会同时记录已安装的包及其源码哈希。

文本路径是：token ID → tied embedding 查表 → 24 个 residual block → final RMSNorm → tied 词表投影。每个 block 的计算为：

```text
x ──────────────────────────────┐
└─ input RMSNorm → token mixer ──+ → y ───────────────────────┐
                                    └─ RMSNorm → gated MLP ─+ → output
```

层排布是三个 gated delta block 之后接一个 full attention block，循环重复。图谱展示每一个 block，并根据权重形状推导各投影的输入/输出宽度。对于按 `[out, in]` 存储的线性层权重，数据从 `in` 流向 `out`；形状的顺序并不是执行顺序。

在配置的 full attention block 中，8 个 query head 每个 256 维，合计 2,048 维，超过了 1,024 维的 residual 宽度。query projection 还同时产生 output gate，因此它的原始输出宽度是 query 总宽度的两倍。2 个 key/value head 在各 query head 组之间重复使用。query/key 的 RMS 归一化在部分旋转位置变换（partial RoPE）之前进行。打分、causal/padding mask、softmax、value 聚合、sigmoid output gate 以及 output projection 都是彼此独立的操作。

gated delta block 先投影出 Q/K/V，做 depthwise causal convolution 和激活，再对 Q/K 归一化，然后更新一个循环的矩阵状态。它的控制量是 `beta = sigmoid(b)` 和 `g = -exp(A_log) * softplus(a + dt_bias)`。每个 token 先用 `exp(g)` 衰减旧状态，计算自身 value 与状态按 key 读出的结果之间的误差，再加上一个由 `beta` 缩放的 key/误差外积。归一化并缩放后的 query 读取更新后的状态；gated RMS 归一化和 output projection 再把结果映射回 residual 宽度。该实现的 prefill 分块（chunk）算法与单 token 循环算法必须在实测的数值容差内一致。

## 参数与 cache 统计

inventory 区分了 checkpoint 张量、加载后的逻辑引用、去重后的实际存储，以及真正执行的文本路径。tied embedding/head 的存储在去重总数中只计一次。vision 参数虽然被加载，但在初始的纯文本 trace 中并不执行。辅助/MTP 张量以及加载时缺失/多余的键都会明确列出；checkpoint 的总参数量并不自动等于模型实际激活的参数量。

full attention 的 key/value 状态随序列长度增长：每层逻辑上的 KV 字节数为 `2 × batch × tokens × KV heads × head dimension × element bytes`。gated delta 的循环矩阵逻辑大小固定，为 `batch × value heads × key dimension × value dimension × state element bytes`；卷积历史再额外增加一个固定缓冲区。这些是公式，而不是对分配器内存的测量。实际的 dtype、存储的历史长度、张量字节数以及各测量长度下去重后的存储字节数，请查看运行结果中的 cache 张量。

## 证据导航

生成 `runs/inventory/report.html` 可查看完整的堆叠结构、展开的 block 结构图、实测的投影表格和参数图。旁边的 `inventory.json` 包含每个模块、参数和观测到的边界。trace 报告展示各个内部操作，并链接到固定版本的源码。大的 activation、大的参数分配或高的 attention 概率都不能证明因果重要性；要做因果结论，请使用受控的干预实验。
