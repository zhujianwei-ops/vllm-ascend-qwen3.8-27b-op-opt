# QuantBatchMatmulV3 实验与验收方案

日期：2026-09-22。配套 [开发设计](01-quantbatchmatmulv3-optimization-design.md)。本文规定后续执行方法，**没有已完成的候选性能结果**。

## 1. 先建立可解释的基线

单算子实验不要求先重采整网，也不依赖已发现缓存问题的 BS16 prefill 总时长。先用已知 dtype/format/shape 建立固定输入测试，再用模型中真实 tensor 验证代表性。

每轮记录：

| 类别 | 必须记录的字段 |
| --- | --- |
| 软件 | Python、torch、torch_npu、CANN、驱动、vLLM/Ascend 版本与 commit，实际 import 路径 |
| 硬件 | SoC、物理卡/芯片与逻辑 NPU 映射、可见设备、活动进程、频率/功耗信息（能取得时） |
| 算子 | M/K/N、所有 tensor dtype、逻辑 shape、storage format、stride、offset、bias/offset 是否为空 |
| 路线 | baseline/pad_m/split_m/ascendc、候选版本、tile 参数、构建选项、workspace |
| 计时 | eager/graph、warmup 次数、repeat 次数、event/wall/kernel 的单位与口径 |
| 数据 | seed、输入范围/分布、真实权重来源与 hash、scale 来源、输入是否已量化 |

Device14/15 是历史 profiler 的设备编号。设置 `ASCEND_RT_VISIBLE_DEVICES=14,15` 后，测试进程可能使用重映射后的 `npu:0/1`，不能未经检查在 Python 中继续写 `npu:14/15`。单算子只需一张空闲 NPU；服务集成保持双卡。

当前设计会话读取 CANN 源码及头文件正常，但调用 `msprof --help` 返回 `Init platform by driver faild!`；此前 `npu-smi` 也返回过 DCMI 初始化失败。这个现象表示**本会话尚未验证 NPU 执行权限**，不代表用户原服务无法运行。执行者在用户正常运行服务的容器/环境内做最小 NPU 验证；若仍失败，先完成 CPU 侧代码、数据提取与构建检查，准确记录驱动/设备挂载障碍，不伪造性能数据。

## 2. Shape 测试集

所有首轮性能 shape 固定 K=5120、N=17408、A/W=INT8、scale=BF16、token_scale=FP32、Y=BF16、W=NZ。

| 分组 | M 集合 | 用途 |
| --- | --- | --- |
| 原始目标 | 15900、15912 | 必须通过，首版启用范围 |
| BS32 扩展 | 15920 | 独立通过后才能加入白名单 |
| 相邻已观测 | 15364、15392 | 同 shape 家族的效率参照 |
| 补齐与分块参照 | 15360、15872、15904、15936、16000、16384 | 判断对齐/分块影响与路线 A/B 的净成本 |
| 拆分尾部 | 540、552、560 | M0=15360 时的实际尾部调用成本 |
| 小 M 回退 | 1、8、16、24、32、40、64、128 | 不得误命中大 M 优化 |
| 其他输入回退 | M=2048/12288；K/N=(8704,5120)、(5120,8192) | 不得误替换其他投影 |

若粗扫描显示 15872 附近有清晰边界，再补 M=15871/15873/15888/15896/15901 等邻近值；没有边界证据时不要无目的扩大扫描。某些 shape 即使 kernel 可以算，首版 dispatcher 仍应保持原路径，直至单独验收。

## 3. 输入构造与实际 Tensor 核对

### 3.1 可控的合成输入

先建立逻辑 `W_nd[5120,17408]` INT8，再按本机官方格式转换为 NZ。若从 checkpoint 的 `[N,K]` 权重开始，严格复现生产加载时的转置/contiguous/格式转换顺序。格式转换计入模型加载记录，不放进每次矩阵乘循环；候选引入的运行时转换则必须计时。

各 M 使用同一最大 A 的前 M 行和相同 W，以降低数据差异影响。weight_scale 在构造后确认为 BF16，token_scale 为 FP32，一次准备后供 baseline/candidate 共用；不误用 `weight_scale_fp32` 代替实际 BF16 scale。

至少三组固定 seed。包含以下输入类型：

- 小整数、均匀随机 INT8、真实 BF16 激活经原 `npu_dynamic_quant` 得到的 INT8/scale。
- 全零、交替正负、包含 -128/127 极值和强抵消的数据。
- 每行和每列都不同的 scale，避免全 1 scale 掩盖广播索引错误。
- 合法的零 scale、不同数量级有限 scale，覆盖 BF16 舍入边界；让 baseline 也产生非有限值的极端输入作为单独语义测试，不混入正常性能验收。

M padding 的尾部 INT8 全零，尾部 token_scale=1.0，显式初始化，不能读取 uninitialized buffer。

### 3.2 真实模型参数

至少取得两个 TP rank 的实际权重和 scale，以及前/中/后主干层与 MTP（若计划覆盖）的量化输入样本。记录 `layer.prefix`、shape、dtype、NZ descriptor 与 bias/offset。这样才能证明白名单覆盖真实 `gate_up_proj`，而非只优化一个合成矩阵。

捕获不放在正式计时中；每个必要调用位置只捕获有限样本。权重不在热路径重复下载到 CPU，也不一次缓存所有大 M 层激活。导出的 tensor、模型分片、构建产物和原始 profiler 放到 `/tmp` 或明确忽略的目录，不纳入 Git；可提交尺寸、hash、生成方法及小型汇总结果。

第一轮不需要加载完整 27B 模型来开始测量，合成基准先验证机制。真实 tensor 验证是进入服务集成前的门槛。

## 4. 性能计时

### 4.1 三种口径分别报告

| 名称 | 计时范围 | 用途 |
| --- | --- | --- |
| `kernel_us` | profiler 中目标 kernel 的设备时长 | 分析 tiling/流水；不能单独验收 wrapper |
| `device_call_us` | 当前 stream 的 timing event 包住完整候选调用 | 包含 pad/copy、所有子 kernel、concat/必要输出处理 |
| `sync_wall_us` | 在输入就绪后，从 Python/C++ 调用到完成同步的墙钟 | 包含正式调用中的分配、host dispatch 和设备执行 |

所有计时排除输入生成、权重 NZ 转换、正确性比较和 profiler 导出。若实现把 padding/分配挪进缓存，也要证明服务中该缓存策略合法，并报告首次及稳态成本。

本机 `torch.npu.Event(enable_timing=True)` 支持设备计时。先保证输入就绪；在当前 stream 上 record 起止事件，在尾事件完成后读取时间。`elapsed_time` 的单位与当前 API 核对后统一转换成 us。不能用未同步的 `time.time()` 只测 Python enqueue，然后声称 kernel 加速。

同步只在 benchmark 测量边界执行，候选算子本身不加入 `synchronize()`。若候选内部使用其他 stream，必须通过 event 让完成事件覆盖全部实际工作；第一版优先单 stream。

### 4.2 建议的最小测量规模

1. 每个 shape/候选至少预热 30 次，首次编译、tiling 初始化和内存分配记录独立保存；仍漂移时延长预热并记录原因。
2. 每个配置至少 5 轮，每轮 20 次完成调用，baseline/candidate 使用交替或随机顺序，固定配对 seed，避免温度/频率随时间变化造成偏差。
3. 每轮报告完整调用总时间/次数，用配对轮次计算延时比；另保留单调用样本的 median/p90/min/max，不将“20 次均值的 P99”当作单调用 P99。
4. 正式性能计时不启用 profiler。对代表性 shape 另采 5–10 次短 trace，用于确认所有新增 kernel 和时间归因。
5. 候选通过后在另一张 TP 卡重复最小测试，不将两卡同时互相干扰的结果混入单卡基线。

只保留必要的输入/输出引用；不要把 100 个约 528 MiB 输出保存在列表中。baseline 与 candidate 输出比较完成后及时释放；循环内不反复 `empty_cache()` 或重新生成权重。

### 4.3 缓存代表性

至少报告两组：

- 固定权重反复调用：用于快速、低噪声定位 kernel 差异。
- 按真实层顺序轮换不同权重/输入：用于判断固定权重缓存带来的乐观偏差，控制驻留内存预算。

不把单权重热缓存最优值直接外推到全模型。双卡集成后的量化输入、通信与其他 stream 可能改变访存环境，这由整网 A/B 检查。

## 5. 正确性验收

### 5.1 主基准与独立交叉检查

主基准是本机未改动的 `torch_npu.npu_quant_matmul`，传入同一 A/W/scale/token_scale。每个目标 shape 与候选至少使用三组 seed，逐元素比较完整 BF16 输出，不能只看 cosine similarity。

本设计首版要求 `torch.equal(candidate, baseline)`，同时检查 shape、dtype、格式、stride、finite 状态和输入未被写坏。纯 padding/M 分块在数学上不改变每行点积，优先争取逐位一致；若 shape 改变使 CANN 自身舍入不同，也视作当前严格门槛未通过。

专用 kernel 若无法逐位一致，先输出 mismatch 分布、最大绝对/相对误差、BF16 ULP 差、具体行列和对应 scale。不要自行扩大 rtol/atol 直到测试通过。后续可以单列精度研究，但未满足已约定阈值的候选不进入默认集成；需要先说明误差来源并重新定义经过业务验证的精度门槛。

作为独立交叉检查，对固定抽样的边界行/列计算 CPU INT64 点积，再按顺序做 FP32 反量化及 BF16 舍入。覆盖首/末行、tile 边界、M0 拆分边界、gate/up 的列边界和末列。CPU 数学参考用于定位索引/累加错误，不能代替全部输出与原算子的对照，因为硬件反量化的细节仍以原实现为准。

INT64 交叉检查按小块或抽样执行，不把完整 15900×5120×17408 的计算放在慢速 Python 循环中。

### 5.2 内存、异步与并发

必须覆盖：

- 两次连续调用使用不同输入，第二次执行后第一次输出仍正确；发现输出复用覆盖立即失败。
- 多次交替目标/非目标 M、两张卡、至少两个 stream，事件明确串接依赖；scratch 不串扰。
- pad 尾行清零、拆分 tail scale 的索引、输出只含真实 M 行。
- 验证输入/权重/scale 未改变，异常 dtype/format 走原路径。
- 峰值 allocated/reserved memory、workspace、稳态运行是否持续增长；固定请求数下不能 OOM 或增加 preemption。
- 新写 kernel 使用当前环境可用的越界检查/调试工具；不可用时明确记录，至少完成边界、并发和长循环的正确性验证。

错误不能通过捕获异常后静默调用 baseline 来“修复”；异步 kernel 错误应让该次实验失败，并停止继续使用有错误的 stream。

## 6. 路线专属验收

| 路线 | 必须额外证明 |
| --- | --- |
| pad_m | 完整 pad/copy/MatMul/output 时间有收益；输出 view 的下游使用合法；额外约 80 MiB 输入缓冲及 padded 输出峰值可接受 |
| split_m | 两次调用+concat 的净收益；直接写 Y 分片版本需证明无隐藏 copy、out descriptor/offset 正确 |
| ascendc | 正确 NZ 访问、无 split-K 舍入改变、tile/workspace 合法、M 尾块不越界、Cube/Vector 同步和多 stream 安全 |

候选成本与 baseline 必须使用相同计时口径。例如不能把 candidate 的预分配 kernel-only 时间与 baseline 的分配加完整 API 时间比较。

## 7. 单算子进入集成的标准

以下是本任务预设验收标准，后续不能把它们写成已观测事实：

1. **正确性**：支持范围内完整输出与原算子逐元素一致，负面条件可靠 fallback，内存/并发测试通过。
2. **性能**：两个主目标 M 的配对完整调用 median 均降低至少 10%，单张卡五轮结果方向一致；`device_call_us` 和 `sync_wall_us` 均报告，后者无明显回退。
3. **噪声**：先用 baseline 对 baseline 的 A/A 测试估计波动。收益置信范围包含 0 或波动过大时，标为未证实并排查环境，不能仅挑最小值验收。
4. **非目标**：维持原实现，dispatch 引入的开销应处于 A/A 噪声内；若系统性超过约 1%，继续压缩热路径判断。
5. **资源**：无泄漏、无 OOM；图和 eager 的输出资源生命周期明确，额外峰值内存已量化。内存预算是否可接受以原服务 0.85 utilization 下的实测余量为准，不预先假定有空闲。

追求 15%–20% 为优化目标；没有达到 18.8% 的历史外推目标不等于实验失败。反之，即使新 kernel 自身快 20%，完整调用只快 3%，也未达到本方案的集成门槛。

## 8. 图模式及集成回归

使用与服务一致的图编译环境单独测试；不能用 `enforce_eager` 改全服务来回避图兼容问题。首版可只在验证过的 eager 大 M 路径启用，其余执行模式在捕获前选择原路径，并记录实际覆盖率。

测试新 op 的 schema、Fake/Meta 输出 `[M,N]`、dtype/device，所有 M 的图专门化/缓存键、重复 replay、模式切换后的重新捕获、输出 alias 与缓冲地址生命周期。Meta 支持不代表 graph 全链路已验证。

诊断命中计数在专门短 run 完成，可用 trace/图节点或少量非热路径日志；不能每次算子打印模块名再测正式吞吐，也不能仅以 Python counter 推断图 replay 的设备调用数。

TP 两 rank 使用同一软件构建和配置，shape 不在白名单时各自回原实现；不能为了强行命中候选修改 scheduler 的 token 合批策略。集成记录实际命中的 shape/模块/次数，按它们重算收益预算。

## 9. 整网 A/B

### 9.1 场景与不变量

| 场景 | 输入/输出 | 并发/请求数 | 用途 |
| --- | --- | --- | --- |
| 业务 Prefill | 8192 / 1 | C16=160 请求；C32=320 请求 | TTFT、输入处理吞吐；先固定缓存条件 |
| 业务长输出 | 8192 / 128 | C16=160 请求；C32=320 请求 | output tok/s、TPOT、ITL、TTFT、MTP 接受率 |
| 短 trace 定位 | 8192 / 7 | C16=16 请求；C32=32 请求 | 核对目标 shape 命中与关键路径变化 |

复用 `02` 的 benchmark 参数与 `03` 的采集方式，A/B 使用完全相同的固定请求数据，TP/W8A8/MTP、max-num-seqs、token budget、图配置和设备保持不变。单个 shape 优化的集成过程中不顺带调通信或缓存参数。

冷缓存实验先预热模型/图，再在无活动请求时重置缓存并核实生效；热缓存作为独立实验。对 `vllm bench` 的 readiness/warmup 行为以当前 CLI 源码为准：本机 `--ready-check-timeout-sec` 默认 0；不要依赖另一个版本的默认值。

可以预先检查候选 shape 是否会被该场景自然触发；未触发时报告“覆盖率为零”，不能为了展示收益临时改变 A/B 中任意一方的调度。

### 9.2 可比性和统计

正式 benchmark 不开 profiler；trace 是独立诊断 run。每个 A/B 场景至少 3 组配对，预计整网效果只有约 0.9%，必要时增加配对轮数，直到能区分噪声或明确无法区分。

保存成功/失败请求数、输入输出 tokens、duration、tok/s、Mean/P50/P90/P99 的 TTFT/TPOT/ITL、MTP acceptance、峰值显存、preemption、两 rank 的目标算子命中与累计时间。

如果候选导致数值变化、MTP 接受率显著变化或真实计算 token 数不同，就不能把所有端到端变化归因于 kernel 提速。先归因，再决定是否作为单算子优化收益。

报告下列两种结果之一：

- 同请求工作量、缓存与接受行为可比，目标算子计时改善，整网配对差异超过 A/A 噪声，可报告实测提升及区间。
- 只有单算子收益可靠，整网差异未超过噪声或覆盖不足，明确报告“整网收益未证实”，保留默认关闭和回退能力。

0.88% 是历史 Stage 的条件预算，不作为必须完成的业务吞吐门槛，也不作为 TPOT/ITL 的预计降幅。

## 10. 输出文件与证据

执行者在 `05-operator-development/quantbatchmatmulv3/results/<run_id>/` 保存小型结果，至少包含：

```text
environment.json           软件、硬件、运行模式和源码版本
cases.json                 shape、模块范围、候选和输入种子
samples.csv                每次/每轮原始计时，字段区分 event、wall、kernel
summary.csv                基线/候选、降时%、波动、内存、验收状态
correctness.json           全量对比、独立抽查、边界及并发结果
decision.md                采用/放弃原因和下一步；失败实验也保存
```

推荐计时字段：`run_id,rank,device,impl,execution_mode,M,K,N,seed,cache_mode,round,repeat,device_call_us,sync_wall_us,kernel_us,workspace_bytes,peak_allocated_bytes`。不可测的字段用 null/空值并解释，不填 0 假装测量完成。

实验原始 profile 使用独立目录，例如：

```text
04-operator-profiling-data-analysis/vllm_profile/
  qmm_baseline_c16_8k_o7_<run_id>/
  qmm_pad_m_c16_8k_o7_<run_id>/
  qmm_ascendc_c32_8k_o7_<run_id>/
```

避免覆盖现有 `bs16_*`、`bs32_*`。Raw profiling 保持 Git 忽略；`07-final-performance/` 保存业务 benchmark 文本/小型结构化数据及 A/B 报告。
