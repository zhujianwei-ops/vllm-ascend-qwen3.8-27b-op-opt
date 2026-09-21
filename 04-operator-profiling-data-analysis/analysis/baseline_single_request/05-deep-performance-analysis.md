# Qwen3.8-27B 双卡 TP 性能深度分析

分析日期：2026-09-20

分析对象：

- Device 14 / rank0：`rank0_17391_20260920075211024_ascend_pt`
- Device 15 / rank1：`rank1_17410_20260920075211024_ascend_pt`
- 运行参数：TP=2、W8A8、MTP speculative tokens=3、`max_num_seqs=32`、`max_num_batched_tokens=16384`、`FULL_DECODE_ONLY`
- 当前实际请求：单请求，prompt 为 `San Francisco is a`，`max_tokens=7`

## 结论先行

当前最值得做的工作不是继续微调普通 MatMul，而是先处理 **AllReduce 的启动/同步等待和通信计算不重叠**。在现有窗口中，Device 14 的通信阶段为 172.067 ms，占 Stage 340.640 ms 的 50.51%；两个 Rank 的 Computing 都约 130.29 ms，但通信与计算重叠只有 0.016 ms/0 ms。物理链路实际传输时间只有约 0.903/0.914 ms（SIO），所以主要矛盾是 552 次 AllReduce 的启动、通知、等待和排序，不是链路已经跑满。

普通 W8A8 `QuantBatchMatmulV3` 目前 880 次、约 42.34/42.62 ms，占底层算子统计 18.7%，但按模型规格和输入形状估算的 INT8 MFU 只有约 0.80%，说明存在明显的调度、访存、量化和小矩阵利用率问题。这个 MFU 不能直接等同于整网 MFU，因为 profiling 的 `op_statistic` 是设备算子统计口径，不是完整 Stage。

## 1. 基线和口径

`step_trace_time.csv` 每张卡只有一条聚合记录，不能代表并发 32 的稳态吞吐，也不能把一次 340 ms 直接解释为每 token decode 时延。当前数据混合了短 prompt、7 token 输出、预热/编译影响和 MTP 相关执行。下文所有“当前收益”首先按这次窗口的 Stage 340.64 ms 做理论上限换算，随后给出并发 32 需要重新测量的原因。

硬件规格按 README：整机 8 卡 4.48 PFLOPS FP16、8.96 POPS INT8、片上/内存带宽 3.2 TB/s。因此按 8 卡折算单卡理论峰值为：

- FP16/BF16：560 TFLOP/s
- INT8：1.12 POP/s
- 带宽：3.2 TB/s

这里把 1 MAC 计为 2 operations。芯片规格表没有给出单卡 HBM 与 SIO 的独立峰值，因此 AllReduce 不使用伪造的 MBU，而使用 profiling 中的实际传输带宽和等待时间。

## 2. 单算子优化空间

### 2.1 `allreduceAicpuKernel`：整网第一优先级

| 指标 | rank0 | rank1 |
|---|---:|---:|
| 调用次数 | 552 | 552 |
| 总算子时间 | 92.070 ms | 94.123 ms |
| 平均单次 | 166.793 us | 170.513 us |
| 最大单次 | 378.671 us | 555.425 us |
| 观察到的时间频率（按 Stage） | 1,620/s | 1,620/s |
| 所有通信阶段 | 172.067 ms | 31.228 ms |
| 通信占 Stage | 50.51% | 9.17% |

`communication_matrix.json` 显示 AllReduce 的 SIO 传输时间约 0.903/0.914 ms、带宽约 24.76/24.46 GB/s，而 `communication.json` 中 Wait/同步时间达到 191.855/39.735 ms。说明当前不是“把链路带宽再提高一点”就能解决，而是大量小通信、AICPU 调度、通知和跨 Stream 依赖造成了等待。

理论收益边界：

- 若只优化 `allreduceAicpuKernel` 自身 92--94 ms，且其它时间完全不变，rank0 Stage 最多下降约 27.0%，理论吞吐上限约提升 1.37 倍。这个上限不包含其外层 HCCL/等待，实际会小于它。
- 若通过 bucket、通信合并、提前发起和计算/通信重叠，消除 rank0 当前 172.067 ms 的非重叠通信，Stage 理论下界约为 168.57 ms，极限约 2.02 倍；工程上建议先按 **减少 20%--40% 通信/等待** 作为第一阶段目标，对应 Stage 节省约 34--69 ms、整网吞吐提升约 **11%--25%**。这是目标区间，不是已验证结果。
- AllReduce 的 MFU/MBU 不适用：它是通信算子，应该报告传输带宽、等待占比、调用粒度和 overlap，而不是套用矩阵算力利用率。

### 2.2 `QuantBatchMatmulV3`：有明显算力和小矩阵空间

统计：880 次，rank0 42.339 ms、rank1 42.624 ms，平均约 48.1/48.4 us，观察频率约 2,584 次/s。根据 kernel 输入/输出形状逐调用计算 `2*M*K*N`，得到两卡相同的约 382.94 GFLOP 工作量。

按单卡 INT8 峰值 1.12 POP/s：

| 指标 | rank0 | rank1 |
|---|---:|---:|
| 估算工作量 | 382.94 GFLOP | 382.94 GFLOP |
| 实测时间 | 42.339 ms | 42.624 ms |
| 有效吞吐 | 9.04 | 8.98 TFLOP/s |
| 理论 INT8 MFU | **0.81%** | **0.80%** |
| 估算字节量（INT8 输入/权重、BF16 输出） | 47.96 GB | 47.96 GB |
| 估算带宽 | 1.133 TB/s | 1.125 TB/s |
| 理论带宽利用率 MBU | **35.4%** | **35.2%** |

这个 MBU 是“每次都从内存读取权重”的保守模型；如果权重被缓存或片上复用，真实 HBM 流量会小于估算值，不能把 35% 当作硬件计数器读数。以该保守模型做 roofline：带宽下限约 15.0 ms，当前 42.3--42.6 ms，单算子理论最多约 **2.8 倍**（约节省 64.6%）；实际可兑现目标建议先按 **20%--40%**，即每卡节省约 8.5--17.0 ms。

对整网 Stage 的映射：

- 只优化该算子，理论整网节省 8.5--17.0 ms / 340.64 ms，即 **2.5%--5.0%**；对应 Stage 吞吐约提升 **2.6%--5.3%**。
- 把 roofline 极限直接当业务收益是不成立的，因为算子统计存在并行/嵌套，且通信仍是主导瓶颈。

建议重点检查：M=4 等 decode 小矩阵、W8A8 动态量化的 per-token scale/offset、权重 NZ 布局、workspace 复用、kernel launch 数量和是否被 MTP/TP 拆成过细的 batch。

### 2.3 `MatMulV2`：有空间，但优先级低于通信和量化 MatMul

统计：412 次，25.741/25.605 ms，平均约 62.5/62.1 us，观察频率约 1,210 次/s。形状解析得到约 58.84 GFLOP；按 560 TFLOP/s FP16 峰值：

- 有效吞吐约 2.29 TFLOP/s
- FP16/BF16 MFU 约 **0.41%**
- 估算 BF16 输入/输出流量约 27.13 GB
- 带宽约 1.05--1.06 TB/s，保守 MBU 约 **33%**

保守 roofline 带宽下限约 8.48 ms，相对当前 25.6--25.7 ms，单算子理论最多约 3.0 倍；但其整网占 Stage 约 7.5%，且存在少数最大约 1.03 ms 的长尾。因此建议把目标定为 **15%--30%** 单算子降时，即整网 Stage 节省约 **1.1%--2.3%**，并先定位长尾是否是同步/特殊 Shape，而不是立即重写通用 MatMul。

### 2.4 Vector/状态类算子

- `Transpose`：853 次，13.54/13.46 ms，约 2,504 次/s，占统计约 5.9%。
- `DynamicQuant`：880 次，3.36/3.34 ms，和 QuantBatchMatmul 调用次数完全一致，约 2,584 次/s。
- `RecurrentGatedDeltaRule`：144 次，3.83/3.96 ms。
- `ConcatD`、`Index`、`Slice` 合计约 9.8 ms，且调用次数高。

单个小 Vector 算子的绝对收益不如通信，但它们是融合的高价值对象，因为可以减少中间张量读写和 launch。对 Transpose/Index/Slice 不建议只做单 kernel 微优化，应从布局保持、视图化、生产者/消费者融合入手。

## 3. 可融合算子和收益估计

以下收益是按 rank0 作为代表、以当前 Stage 340.64 ms 映射的工程估计；rank1 数值接近。

| 候选融合 | 当前组成 | 当前总耗时 | 建议可兑现单算子收益 | 对整网 Stage 的直接收益 |
|---|---|---:|---:|---:|
| `DynamicQuant` + `QuantBatchMatmulV3` | 880 + 880 次 | 45.70 ms | 先按减少 DynamicQuant 的 50%--80%，约 1.7--2.7 ms；若进一步做量化 MatMul 内融合，目标总降 15%--30%，约 6.9--13.7 ms | 2.0%--4.0% Stage；吞吐约 2.0%--4.2% |
| GDN 子流程：`ChunkGatedDeltaRuleFwdH` + `recompute_w_u_fwd_kernel` + `solve_tril_16x16_kernel` + `ChunkFwdO` | 每项 48 次 | 8.63 ms | 融合中间结果和 launch，先按 20%--40%，约 1.7--3.5 ms | 0.5%--1.0% Stage |
| Attention/状态路径的 Transpose + `ConcatD`/`Index`/`Slice` | 853/204/200/1505 次 | 23.36 ms | 仅在布局可保持时，先按 15%--30%，约 3.5--7.0 ms | 1.0%--2.1% Stage |
| 多个小 AllReduce bucket 合并，并与下一层计算重叠 | 552 次 AllReduce | 通信阶段 172.07 ms（rank0） | 先按等待/非重叠减少 20%--40%，约 34--69 ms | 10%--20% Stage；这是最大收益方向 |

融合不能简单把表中时间相加后当作可全部消除：kernel 可能并行，父子算子可能重复计时，且融合后会改变通信顺序和显存峰值。每个候选都必须用同一请求集做 A/B profiling。

## 4. 并发 32 场景的判断

当前脚本虽然设置了 `max_num_seqs=32`，但采集脚本实际只发了 1 个请求。因此不能从现有 CSV 推断并发 32 的 token/s、P99 decode 时延或 TTFT。并发 32 预计会改变：

1. decode MatMul 的 M 从 4 附近变为更大 batch，QuantBatchMatmul 的 MFU/MBU 可能显著上升，单算子排序可能变化。
2. AllReduce 次数未必变化，但每次消息更大，通信带宽占比、等待结构和 overlap 会改变。
3. MTP 接受率会影响实际 forward 次数，必须同时记录 accepted token、draft token 和回退比例。
4. 长短请求混合会改变 prefill/decode 比例；需要分别看 TTFT、ITL/P50/P99 和 aggregate output tok/s。

## 5. 后续工作规划

### 第一阶段：先把通信时序做实

- 采集 rank0/rank1 同一窗口的 trace，标出 552 次 AllReduce 的开始/结束、通知等待和依赖 Stream。
- 将 AllReduce 按消息大小和调用位置分桶，确认是否是大量小 tensor、固定层间同步，还是某些异常长尾。
- 尝试 HCCL bucket/capacity、通信提前发起、分块 reduce、通信 stream 与计算 stream 解耦；先保证数值一致和显存峰值可控。
- 目标：rank0 非重叠通信减少至少 20%，并将 overlap 从接近 0 提升到可观测水平。

### 第二阶段：做 W8A8 decode 专项 kernel

- 以 `M=4`、`M=8/16/32`、`K=5120/8704`、`N=5120/8192/17408` 建立 shape 白名单和基准。
- 把 per-token DynamicQuant 的 scale/offset 计算、量化写回和 QuantBatchMatmul 消费尽量放在同一 kernel 或同一图段。
- 检查 NZ 权重布局是否和 decode tile 对齐，复用 workspace，减少动态分配和 format transform。
- 以单算子降时 20% 为最低验收目标，再看整网收益。

### 第三阶段：状态/布局融合

- 优先做 GDN 四段子流程的中间张量消除；保留一个 unfused fallback 便于回归。
- 对 Transpose/ConcatD/Index/Slice 先做 layout contract，避免跨层来回转置；只有能减少真实读写时才写自定义 Vector kernel。
- 检查 `AddRmsNormBias`、`SwiGlu` 等已有融合算子是否覆盖当前动态 Shape，避免重复开发。

### 第四阶段：图编译和运行时配置

- 当前 speculative 配置使用 `enforce_eager=true`，这会限制部分图级融合/捕获空间；在功能允许时分别测试 eager、decode graph/cudagraph 和固定 batch bucket。
- 为并发 32 建立 batch bucket（例如 1/4/8/16/32）和 prompt/decode 分离的 profiling 集合。
- 编译/预热必须排除在正式基线之外，至少固定多轮 warmup 后再采集 30--100 个稳定 decode step。

## 6. 必须补采的数据

建议至少补四组：

1. 单并发 decode：固定输入长度，输出 128--256 token，预热后采集 50 step。
2. 并发 32 同长度：分别测 prefill、decode 和 MTP accepted token。
3. 并发 32 长短混合：记录 TTFT、ITL/P99 和输出 tok/s。
4. 算子 A/B：只改变一个变量（通信 bucket、DynamicQuant 融合、GDN 融合或图模式），保持模型、TP、W8A8、MTP 不变。

每组都保存：请求级 TTFT/ITL、输入/输出 token 数、accepted/draft token、rank trace、`step_trace_time.csv`、`op_statistic.csv`、`kernel_details.csv`、`communication.json` 和显存峰值。

## 最终判断

当前数据支持“**通信同步是第一瓶颈，W8A8 小矩阵和量化路径是第二瓶颈，布局/状态小算子适合融合**”这一结论。普通 MatMul 不是没有空间，但在整网收益上应排在 AllReduce overlap 和 QuantBatchMatmul/DynamicQuant 融合之后。任何“已无优化空间”的结论目前都只能对某个已经接近硬件 roofline 的具体 kernel 给出；本次数据中没有这样的证据，反而显示主要计算算子的 MFU 远低于芯片理论峰值，通信 overlap 也接近零。
