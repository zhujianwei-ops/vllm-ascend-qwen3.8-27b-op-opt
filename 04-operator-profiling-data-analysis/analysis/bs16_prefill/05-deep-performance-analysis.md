# Qwen3.8-27B BS16 Prefill 性能分析

分析日期：2026-09-22。

## 1. 结论与适用范围

本次两卡的 Stage 为 **4,048.502 / 4,048.576 ms**。rank0 的 Computing 占 84.24%，非重叠通信占 10.45%，Free 占 5.30%，CANN 报告的通信计算重叠为 0。

但当前数据**不能作为无缓存的 BS16、8K prefill 基线**：按主干每层 QKV 的输入 shape 还原，实际执行 token 行数为 **32,768**，只有脚本声明的 `16×8192=131,072` 的 25%。它与 BS32 prefill 的实际 256,000 行相差 7.8125 倍，而 Stage 相差约 7.12 倍；不能得出“BS16 算子比 BS32 快 7 倍”的结论。

已有证据支持优先做 Norm/激活与动态量化融合、具体 shape 的量化 MatMul 优化、通信依赖检查。尚无证据支持“这些算子已没有优化空间”；也不能把 AICPU 或等待累计时间全部当成可回收的整网时延。

## 2. 数据来源与采集审计

| 项目 | 记录 |
| --- | --- |
| rank0 / Device14 | `rank0_10893_20260922015612919_ascend_pt` |
| rank1 / Device15 | `rank1_10912_20260922015612919_ascend_pt` |
| 采集脚本 | `03-model-whole-network-profiling/05-run-bs16-prefill-profiling.sh` |
| 请求配置 | random、16 请求、并发上限 16、输入 8192、输出 1、seed=20260921、num-warmups=0 |
| 部署脚本 | TP=2、W8A8、MTP=3、max-num-seqs=32、max-num-batched-tokens=16384、开启 prefix caching、FULL_DECODE_ONLY |
| 采集元数据 | torch_npu=2.10.0.post4、CANN=9.1.0、Level1、PipeUtilization、record_shapes=false、with_stack=false |
| 本机安装版本 | vLLM=0.23.0+empty、vllm-ascend=0.23.0 |

版本元数据直接来自 rank 文件；部署参数来自当前脚本，没有本轮启动命令快照，不能排除采集时有临时调整。请求并发 16 也不意味着每次模型 forward 都有 16 条活动序列。

两个 rank 的七项主要导出文件完整，`op_statistic` 与 `kernel_details` 的逐类型次数和总时间已交叉校验。Step 列为空，日志中的 “step node list is empty” 表明没有逐 step 标记，本报告使用完整窗口汇总。

输出 1 的请求仍执行 prefill、首 token 的 logits/采样与框架逻辑；**首 token 通常由 prefill 结果产生，不应自动视为额外执行了一次独立 decode forward**。本窗口没有主干 QKV 小 M 分桶，但 LM head 仍有小 M 矩阵乘，不能仅凭小矩阵判断阶段。

### 实际 token 行数

模型配置位于 `/home1/model/Qwen3.8-27B-w8a8/config.json`：主干 64 层，其中 48 层 linear attention、16 层 full attention，hidden_size=5120。每层 linear attention 的 TP 本地 QKV 投影为 `K=5120,N=8192`。

按时间排序，连续每 48 个这类投影构成一个主干执行分组；四组内 M 都一致：

| 执行分组 | QKV 的 M | 同 shape 的主干 QKV 次数 |
| --- | ---: | ---: |
| 1 | 2,048 | 48 |
| 2 | 2,048 | 48 |
| 3 | 16,384 | 48 |
| 4 | 12,288 | 48 |
| 合计 | **32,768 行/层** | 192 |

这里只统计主干 QKV，不把其他层、MTP 投影或同一 token 的多次矩阵乘重复当作输入 token。该值是设备实际处理的 token 行数；仅凭它不能给出准确的业务 cache hit rate。

同 seed、相同输入、开启 prefix caching，使缓存复用成为最需要验证的解释。还需检查采集窗口是否完整、服务端是否有其他请求及缓存命中记录。没有这些记录之前，不把差额 75% 写成已证实的缓存命中率。

## 3. 时间与热点

| 字段 | rank0 (ms) | rank1 (ms) |
| --- | ---: | ---: |
| Stage | 4,048.502 | 4,048.576 |
| Computing | 3,410.628 | 3,413.781 |
| Communication(Not Overlapped) | 423.136 | 416.111 |
| Overlapped | 0.000 | 0.000 |
| Free | 214.738 | 218.683 |
| Preparing | 2.087 | 2.175 |

`Stage≈Computing+非重叠通信+Free`；Preparing 不再累加。两卡窗口长度接近，但这不足以证明每个通信点都没有到达时间差。

| 算子 | 次数/卡 | rank0 累计 (ms) | 占底层累计 | 频率 (次/Stage 秒) |
| --- | ---: | ---: | ---: | ---: |
| allreduceAicpuKernel | 552 | 1,716.705 | 32.62% | 136.35 |
| QuantBatchMatmulV3 | 880 | 1,292.290 | 24.56% | 217.36 |
| FusedInferAttentionScore | 76 | 243.427 | 4.63% | 18.77 |
| recompute_w_u_fwd_kernel | 192 | 212.876 | 4.05% | 47.42 |
| MatMulV3 | 204 | 177.423 | 3.37% | 50.39 |
| merge_16x16_to_64x64_inverse_kernel | 192 | 176.882 | 3.36% | 47.42 |
| AddRmsNormBias | 536 | 170.498 | 3.24% | 132.39 |
| Transpose | 2,916 | 148.331 | 2.82% | 720.27 |
| DynamicQuant | 880 | 127.547 | 2.42% | 217.36 |

底层累计为 5,262.236 / 5,259.130 ms，含跨 stream 的重叠，不能当作端到端时间。尤其 AllReduce 的 AI_CPU 包络 1,716.705 ms，不等于 Stage 中的 423.136 ms 通信阶段，不能据前者宣称整网可直接下降 42%。

## 4. 单算子空间、MFU 与 MBU 的边界

本节沿用 README 参考峰值：单份 INT8=1.12 POP/s、BF16=560 TFLOP/s、带宽=3.2 TB/s。前两项按整机规格除以 8；实际 logical Device 与整机“8 NPU”的映射及带宽范围尚未核实。采集的 `device_14/info.json.14` 只确认 24 个 AI Core、48 个 Vector Core、频率字段 1800 MHz，不能独立确认上述峰值。

因此表中是**算子算力利用率 U_compute（算子口径的 MFU 估计）**，不是模型级 MFU；MBU 是每调用对输入/权重/输出/scale 各访问一次的字节模型，未采集真实 HBM 流量。公式与完整 shape 数据见 [Kernel 报告](kernel-and-task/01-kernel-details-analysis.md)。

| 算子 | 逻辑工作量 (TOP) | rank0 有效 TOP/s | U_compute rank0 / rank1 | 模型 MBU rank0 / rank1 | rank0 条件 Roofline 下界 |
| --- | ---: | ---: | --- | --- | ---: |
| QuantBatchMatmulV3 | 784.261 | 606.877 | 54.19% / 54.23% | 5.60% / 5.60% | 700.233 ms |
| MatMulV3 | 54.632 | 307.919 | 54.99% / 55.17% | 6.16% / 6.18% | 97.557 ms |
| MatMulV2 | 0.854 | 19.955 | 3.56% / 3.59% | 26.79% / 26.99% | 11.472 ms |

INT8 TOP 表示整数 operations，BF16 的相同计数可表达为 FLOPs；均按 1 MAC=2 operations。条件模型内，三类算子的 `当前累计/模型下界` 分别为 1.85、1.82、3.73 倍；分母未确认且忽略缓存/调度，**不能当作实际可实现的加速倍数**。

量化 MatMul 的加权 `cube_utilization` 为 98.57%，`aic_mac_ratio` 为 88.49%。本机 CANN 的 cube_utilization 使用核活跃周期覆盖计算，并非有效 FLOP/峰值；其接近 100% 不能证明算力已经满载。大 M 算子更适合做具体 tiling 和融合实验，缺乏依据直接承诺成倍加速。

MatMulV2 总量只有 42.820 ms：即使其 kernel 累计全部消除且全部落在关键路径，当前窗口时间预算也仅约 1.06%。这说明应按整网预算安排优先级，不能只按低 MFU 选算子。

FusedAttention、GDN、Transpose 等没有完整有效长度、访存计数或运算量模型，本文不编造 MFU/MBU。频率、真实 shape、累计时间足以先筛选实验对象。

## 5. 融合候选与整网预算

已按相同 stream、计算 kernel 的时间顺序和输入输出 shape 核对相邻关系；这比仅凭次数相同更有证据，但 CSV 没有 tensor 身份，最终仍需检查图依赖及额外消费者。

收益换算统一为 `ΔT_model=当前候选累计×假设降幅`，`Stage 降幅=ΔT_model/4048.5015`。假设节省全在关键路径且无回退，实际收益可能更小甚至为零。下面是开发预算，不是实测预测区间；候选重叠时不叠加。

| 候选 | 当前匹配范围 | 当前累计 (ms) | 假设局部降幅 | 可释放累计 (ms) | 条件 Stage 降幅 |
| --- | --- | ---: | --- | --- | --- |
| AddRmsNormBias + DynamicQuant | 520 对 | 229.918 | 15%–30% | 34.488–68.975 | 0.85%–1.70% |
| SwiGlu + DynamicQuant | 268 对 | 154.044 | 15%–30% | 23.107–46.213 | 0.57%–1.14% |
| DynamicQuant + QuantBatchMatmulV3 | 880 对 | 1,419.837 | 10%–20% | 141.984–283.967 | 3.51%–7.01% |
| GDN 四段优化/中间结果消除 | recompute、merge inverse、FwdH、FwdO 各 192 次 | 609.392 | 10%–20% | 60.939–121.878 | 1.51%–3.01% |
| Transpose/Slice 的布局优化 | 2,916 / 2,084 次，全类筛选预算 | 214.264 | 10%–20% | 21.426–42.853 | 0.53%–1.06% |
| 非重叠通信减少 | CANN Stage 口径 | 423.136 | 20%–40% | 84.627–169.254 | 2.09%–4.18% |

量化 MatMul 融合若**仅消除 DynamicQuant**，最多只有 127.547 ms 的 kernel 累计预算，即 Stage 的 3.15%；若消除其 50%–80%，为 63.773–102.037 ms，Stage 的 1.58%–2.52%。表中更大的组合收益必须同时改善 MatMul 或数据访问，不能仅靠减少一次 launch 解释。

优先验证本机已有 `torch_npu.npu_add_rms_norm_dynamic_quant`，保留 residual、Gemma/RMSNorm 权重和 bias 语义，以及当前 per-token 量化 scale。`npu_swiglu_quant` 虽存在，但本机接口文档声明输入尾轴不超过 8192，本模型 SwiGlu 输入尾轴为 **17408**，不能未经验证直接替换；可能需要自定义实现或确认另一个支持此 shape 的接口。

GDN 四个名称只是优化链路预算，不意味着可直接合成一个 kernel。状态依赖、跨 tile 同步、寄存器/片上内存容量、输出消费者可能限制融合，应先选相邻的可行子链。

## 6. 通信与等待

`communication.json` 排除 `Total Op Info` 后有 **580 次 collective：552 AllReduce + 28 AllGather**。rank0 Elapse=423.136 ms、Wait=422.930 ms、Transit=0；Wait 与 Synchronization 重叠，不再相加。

但 AllReduce 的 SIO 传输仍记录 **45,634.355 MB、300.308 ms、151.958 GB/s**；rank1 为 299.372 ms、152.434 GB/s。Transit=0 不能解释为无数据传输，Wait 也不能全部解释为可消除的纯空等。LOCAL、SIO 和 SDMA 不重复相加。

EVENT_WAIT 累计 8,563.198 ms，最大 894.485 ms；NOTIFY_WAIT_SQE 累计 7,012.626 ms。它们跨 stream 重叠，只用于定位时序，不能加到 4,048.502 ms Stage 上。先检查与这些长等待对应的真实计算空洞，再考虑通信发起、图切分或依赖调整。

## 7. 后续工作

1. **先补齐可比基线**：保持 TP/W8A8/MTP，充分预热后固定缓存策略；冷缓存实验在无活动请求时重置缓存或使用独立输入集，记录实际输入/输出 token、缓存命中和启动命令。确认主干 token 行数是否覆盖预期工作量。
2. **做 Norm + DynamicQuant 最小验证**：先使用已有融合接口，覆盖 M=2048/12288/16384，核对 residual、量化 scale、BF16 舍入和最终输出；比较匹配的整个组合耗时。
3. **做长输入量化 MatMul 与 GDN 实验**：固定真实 K/N 和 NZ 权重布局，只改变一个 tiling/融合变量；同步记录显存与 PMU，避免以更大中间张量换来局部加速。
4. **再评估通信和 Free**：按通信序号与层依赖定位可隐藏区间，不跨有数据依赖的层随意合并 AllReduce；使用请求级 TTFT 验收。

当前采集不是稳定吞吐测量。`02` 中 160 请求的 BS16、8K/1 基线 Mean TTFT=14,136.13 ms、Total token throughput=8,943.93 tok/s，可作为业务侧记录，但不能与本次有缓存疑点的 Stage 直接相除。

## 8. 复现与阅读入口

- [BS16/BS32 比较及口径审计](../bs16-comparison.md)
- [算子统计](operator-details/02-rank-op-statistic-analysis.md)、[框架调用](operator-details/01-rank-operator-details-analysis.md)
- [Kernel/shape](kernel-and-task/01-kernel-details-analysis.md)、[任务等待](kernel-and-task/02-task-time-analysis.md)
- [通信统计](communication/01-communication-json-analysis.md)、[链路](communication/02-communication-matrix-analysis.md)、[通信阶段](communication/03-step-trace-time-analysis.md)
- [Stage 分解](step-trace-time/01-rank-step-trace-time-analysis.md)、[计算底稿与输入文件 SHA256](../bs16-profile-metrics.json)

从仓库根目录重新计算数值：

```bash
python3 04-operator-profiling-data-analysis/03-summarize-profile.py \
  --output /tmp/bs16-profile-metrics.json
```

该命令只读原始数据，重新输出统计与校验结果；Markdown 中的判断与预算为本次分析结论，不由该命令自动改写。
