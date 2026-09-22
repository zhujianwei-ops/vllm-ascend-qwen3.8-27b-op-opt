# Qwen3.8-27B BS16 Decode 性能分析

分析日期：2026-09-22。

## 1. 结论

两卡 Stage 为 **16,563.930 / 16,563.903 ms**。rank0 Computing 占 88.95%，非重叠通信占 10.32%，Free 占 0.73%。QuantBatchMatmulV3 累计 5,573.330 ms，占底层 kernel 累计的 35.97%，是本窗口最大的计算热点。

**这个目录是输出 7 token 的完整请求窗口，计算仍由长输入 prefill/混合执行主导。** 量化 MatMul 中 M>128 的部分占其时间的 98.84%；M<=128 仅 64.410 ms。它不能代表固定 BS16 的纯 decode 性能，也不能将本窗口减去另一次 BS16 prefill 窗口后解释为 6 个输出 token 的耗时，因为两次实际输入计算量不同。

最具体的开发入口有三个：`M=15900/15912` 的 FFN 量化 MatMul 效率下降；已核对相邻的 Norm/激活加 DynamicQuant；以及长输入与 speculative decode 混合路径上的布局/ScatterUpdate。小 M decode 还应另建稳态基线，重点检查量化/权重访问、通信与图执行。

## 2. 数据与阶段审计

| 项目 | 内容 |
| --- | --- |
| rank0 / Device14 | `rank0_10893_20260922014740695_ascend_pt` |
| rank1 / Device15 | `rank1_10912_20260922014740695_ascend_pt` |
| 采集脚本 | `03-model-whole-network-profiling/04-run-bs16-decode-profiling.sh` |
| 请求 | random、16 请求、并发上限 16、输入 8192、输出 7、seed=20260921、num-warmups=0 |
| 当前部署脚本 | TP=2、W8A8、MTP=3、max-num-seqs=32、max-num-batched-tokens=16384、prefix caching、FULL_DECODE_ONLY |
| 采集版本元数据 | torch_npu=2.10.0.post4、CANN=9.1.0、Level1、PipeUtilization |
| 本机安装 | vLLM=0.23.0+empty、vllm-ascend=0.23.0 |

主要 CSV/JSON 完整，两卡算子类型与次数一致，op_statistic/kernel_details 的逐类型统计校验通过。Step 列为空，因此没有可直接使用的逐 scheduler step 标签。主机 record_shapes=false，但设备 kernel_details 仍有可用 shape。

模型主干有 48 层 linear attention。按每 48 次 `K=5120,N=8192` 的主干 QKV 投影及其 M 值，识别出 17 个顺序执行分组：

```text
6144, 2048, 15364, 14852, 14860, 14868, 15900, 15392,
15912, 11820, 4140, 40, 40, 32, 24, 16, 8
```

前 11 组为大 M，后 6 组为小 M。主干 token 行数合计 **131,460**；它包含 speculative 验证、可能的 padding 与缓存影响，不等于唯一业务输入 token 数，也不能用差额直接计算 MTP 接受率。后段 M 已降到 40/32/24/16/8，也不能把客户端并发上限直接当作实际 forward batch。

MTP 的草稿投影还出现 M=36/28/20 等形状，应与主干 QKV 区分。完整 M、K、N、次数与耗时在 [shape 数据](kernel-and-task/shape-metrics.csv) 中。

## 3. 双卡时间分解

| 字段 | rank0 (ms) | rank1 (ms) |
| --- | ---: | ---: |
| Stage | 16,563.930 | 16,563.903 |
| Computing | 14,733.274 | 14,747.237 |
| Communication(Not Overlapped) | 1,709.492 | 1,695.778 |
| Communication | 1,709.640 | 1,695.856 |
| Overlapped | 0.148 | 0.078 |
| Free | 121.164 | 120.887 |
| Preparing | 2.158 | 2.263 |

`Stage≈Computing+非重叠通信+Free`，两卡 Stage 差 0.027 ms。rank0 的 CANN 通信阶段仅有约 0.0087% 被归为与计算重叠；仍需通过具体依赖确认哪些通信可以隐藏，不能据此无条件承诺 1.7 s 全部可消除。

## 4. 热点、频率与优化空间

| 算子 | 次数/卡 | rank0 累计 (ms) | rank1 累计 (ms) | rank0 底层占比 | rank0 次/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| QuantBatchMatmulV3 | 3,740 | 5,573.330 | 5,574.051 | 35.97% | 225.79 |
| recompute_w_u_fwd_kernel | 528 | 835.853 | 836.650 | 5.39% | 31.88 |
| Transpose | 11,483 | 760.788 | 765.547 | 4.91% | 693.25 |
| allreduceAicpuKernel | 2,346 | 708.092 | 694.713 | 4.57% | 141.63 |
| MatMulV3 | 561 | 697.963 | 696.037 | 4.50% | 33.87 |
| merge_16x16_to_64x64_inverse_kernel | 528 | 694.882 | 695.919 | 4.48% | 31.88 |
| AddRmsNormBias | 2,278 | 678.161 | 679.813 | 4.38% | 137.53 |
| FusedInferAttentionScore | 323 | 623.707 | 626.112 | 4.03% | 19.50 |
| ScatterUpdate | 864 | 583.626 | 580.562 | 3.77% | 52.16 |
| ChunkGatedDeltaRuleFwdH | 528 | 524.407 | 524.480 | 3.38% | 31.88 |
| DynamicQuant | 3,740 | 513.092 | 515.124 | 3.31% | 225.79 |

底层累计时间为 15,495.185 / 15,494.807 ms；表中百分比以这个累计为分母。频率是每秒 Stage 中的调用密度，并非输出 token/s。Host 侧 `npu::npu_quant_matmul` 只有 2,492 条，设备有 3,740 次，图回放等记录口径使二者不能直接等同。

### 4.1 MFU/MBU 估计

沿用 README 条件分母：INT8=1.12 POP/s、BF16=560 TFLOP/s、带宽=3.2 TB/s。逻辑 Device 与整机“8 NPU”的对应尚未充分核实，不能据此宣称已经达到或远离真实单 Device 硬件极限。采集只确认每 Device 24 AI Core、48 Vector Core、频率字段 1800 MHz。

`F=2MKN`，按每次输入/权重/输出/scale 各访问一次估算 `B_model`。下表的 U_compute 是算子口径 MFU 估计，不是模型级 MFU；MBU_model 也不是 HBM 硬件计数。详细计算和所有 shape 见 [Kernel 报告](kernel-and-task/01-kernel-details-analysis.md)。

| rank0 分桶 | 次数 | 累计 (ms) | 有效 TOP/s | U_compute | MBU_model | 条件模型下界 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QuantBatchMatmulV3 全部 | 3,740 | 5,573.330 | 564.531 | 50.40% | 5.27% | 2,828.529 |
| QuantBatchMatmulV3 M>128 | 2,420 | 5,508.920 | 570.439 | 50.93% | 4.92% | 2,805.804 |
| QuantBatchMatmulV3 M<=128 | 1,320 | 64.410 | 59.246 | 5.29% | 35.28% | 22.725 |
| MatMulV3 | 561 | 697.963 | 313.639 | 56.01% | 5.90% | 390.907 |
| MatMulV2 | 1,190 | 196.195 | 22.381 | 4.00% | 25.87% | 50.758 |

rank1 的量化 MatMul U_compute=50.40%、模型 MBU=5.27%；两卡接近。量化大 M 的加权 MAC ratio=83.93%，小 M 为 19.90%；对应 cube_utilization 为 98.14%/75.20%。CANN 的 cube_utilization 是核活跃周期覆盖，不能当作有效乘加占峰值的比例。

条件模型内，小 M 量化 MatMul 当前耗时是下界的约 2.83 倍，但它只占本窗口 Stage 的 0.39%。若其单算子累计下降 20%–40%，节省仅 12.882–25.764 ms，Stage 条件降幅 **0.078%–0.156%**。它对真正的稳态 decode 可能更重要，这需要从长输入执行中单独测量。

Attention、GDN、Scatter 等缺少精确有效长度、有效访存与完整 FLOP 模型，不给出伪精确 MFU/MBU。低模型利用率本身也不能量化融合收益。

### 4.2 明确的 Shape 效率下降

同一次 BS16 decode 采集、同一 `K=5120,N=17408` 的 FFN 投影：

| M | 次数/卡 | rank0 均值 (us) | rank0 U_compute | 加权 MAC ratio |
| --- | ---: | ---: | ---: | ---: |
| 15,364 | 67 | 4,523.319 | 54.06% | 88.17% |
| 15,392 | 67 | 4,522.269 | 54.17% | 88.26% |
| 15,900 | 67 | 5,766.266 | 43.89% | 74.08% |
| 15,912 | 67 | 5,770.114 | 43.89% | 74.07% |

若 M=15900/15912 能达到 M=15364 的单位工作量耗时，则平均单次分别约为 4,681.123/4,684.656 us，下降约 **18.8%**；134 次合计节省约 **145.430 ms**，映射当前 Stage 约 **0.88%**。rank1 按相同方法得到 144.688 ms、0.87%，方向一致。

这是由同 K/N 邻近 shape 得出的**可验证目标**，尚未证明原因就是 tiling。需要比较 tile、尾块、工作区、内存对齐、频率和执行路径；优先复现这些真实形状，避免泛化为“所有量化 MatMul 都能下降 18.8%”。若试验 padding 到 16384，还要包含 padding、scale 和结果裁剪的成本及正确性。

## 5. 融合与整网收益预算

相同 stream 的顺序和 shape 核对发现：3,740 对 DynamicQuant→QuantBatchMatmul，2,210 对 AddRmsNormBias→DynamicQuant，1,139 对 SwiGlu→DynamicQuant。CSV 未记录 tensor 身份，因此还须验证图的数据依赖和额外消费者。

下表中的百分比是工程假设。`ΔT=候选累计×局部降幅`；仅在全部节省都落在关键路径、其他时间不变时，`ΔT/16563.9295` 才是 Stage 的降幅。吞吐在固定工作量下的换算为 `1/(1-Stage降幅)-1`，不能把延时降幅直接称作吞吐增幅。

| 优化候选 | 匹配范围 | 当前累计 (ms) | 假设局部降幅 | 可释放累计 (ms) | 条件 Stage 降幅 |
| --- | --- | ---: | --- | --- | --- |
| Norm + DynamicQuant | 2,210 对 | 916.591 | 15%–30% | 137.489–274.977 | 0.83%–1.66% |
| SwiGlu + DynamicQuant | 1,139 对 | 629.809 | 15%–30% | 94.471–188.943 | 0.57%–1.14% |
| DynamicQuant + QuantBatchMatmul | 3,740 对 | 6,086.423 | 10%–20% | 608.642–1,217.285 | 3.67%–7.35% |
| GDN 四段优化/中间结果消除 | recompute、merge inverse、FwdH、FwdO 各 528 次 | 2,413.304 | 10%–20% | 241.330–482.661 | 1.46%–2.91% |
| 布局及状态写回 | Transpose+Slice+Scatter 全类筛选预算 | 1,713.251 | 10%–20% | 171.325–342.650 | 1.03%–2.07% |
| 非重叠通信减少 | Stage 分解 | 1,709.492 | 20%–40% | 341.898–683.797 | 2.06%–4.13% |

不能叠加共享 DynamicQuant 的融合方案，也不能把布局全类预算和 GDN 内部相同 kernel 重复相加。GDN 四段具有依赖和片上容量限制，并非已验证可直接合成单 kernel。

仅消除全部 DynamicQuant 的预算上限是 513.092 ms、当前 Stage 的 3.10%。若只消除其中 50%–80%，对应 256.546–410.474 ms、1.55%–2.48%；要达到表中的组合高端目标，还必须改善矩阵乘或整体内存访问。

### 现有接口与开发边界

1. 本机 `npu_add_rms_norm_dynamic_quant` 已有 BF16、residual 输出和动态 scale 接口，优先做语义适配与实测；Qwen/Gemma 的权重偏置、residual 多消费者、scale 精度不能改变。
2. 本机 `npu_swiglu_quant` 文档限定输入尾轴不超过 8192，而本模型 SwiGlu 输入为 17408、输出为 8704。先确认实现支持范围，否则开发覆盖该 shape 的融合算子；仅“接口存在”不代表能直接替换。
3. ScatterUpdate 的大热点形状含 `[15900,1,24,128]` 与约 15872 行更新，说明它涉及长输入与 speculative 分支输出的合并，不能全部归因于纯 decode 的 KV cache 更新。可检查 producer 是否能直接写入目标布局、能否消除临时 merged_out；必须保证索引、状态回退和 MTP 接受/拒绝路径一致。
4. 通信只能合并没有先后数据依赖的片段，不能任意跨网络层 bucket。比较图切分、已有通信融合和分块流水，保留 TP/W8A8/MTP。

## 6. 通信与等待证据

`communication.json` 去掉汇总行后记录 **2,629 次 collective：2,393 AllReduce + 236 AllGather**。这里的操作数与 AI_CPU kernel 的 2,346 次 AllReduce 不同，是事件层级和实现路径不同，不能互换。

| 去重字段 | rank0 (ms) | rank1 (ms) |
| --- | ---: | ---: |
| Elapse | 2,451.225 | 2,437.833 |
| Transit | 52.069 | 51.669 |
| Wait | 2,382.307 | 2,369.578 |
| Synchronization | 2,381.955 | 2,368.859 |
| Idle | 16.849 | 16.586 |

Wait 与 Synchronization 不相加。AllReduce SIO 实际仍有 **183,078.154 MB**、1,228.214/1,217.983 ms，带宽 149.061/150.313 GB/s。因此不能根据上表 Transit 较小宣称通信主要是“没有数据搬运的纯等待”。

去重 Collective Elapse 2,451.225 ms 与 Stage Communication 1,709.640 ms 仍不相等，必须保留各自口径。EVENT_WAIT 累计 35,123.547 ms、最大 799.215 ms，跨 stream 的重叠也使其不能作为端到端时间相加。详见通信与 Task 子报告。

## 7. 与 BS32 和业务基线的关系

BS32 decode 的主干 token 行数为 262,860，Stage=33,033.585 ms；BS16 约为它们的 50.01% 和 50.14%。量化 MatMul 有效吞吐均约 564.5 TOP/s，未显示仅降低并发就使这类 kernel 明显更快。

`02` 中另一组 8K/128 的业务结果：

| 指标 | C16，160 请求 | C32，320 请求 |
| --- | ---: | ---: |
| 输出 token/s | 103.31 | 108.14 |
| Mean TTFT (ms) | 2,957.28 | 5,099.40 |
| Mean TPOT (ms) | 130.94 | 255.89 |
| Mean ITL (ms) | 286.31 | 554.06 |
| P99 ITL (ms) | 1,473.67 | 1,966.41 |
| MTP Acceptance rate | 40.28% | 39.53% |

C32 输出吞吐比 C16 高约 4.68%，C16 Mean TPOT 比 C32 低约 48.83%。这是各自已保存的随机负载结果；样本量、轮次、缓存与调度行为不同，尚不代表重复实验的稳定提升，更不能把 OSL128 的接受率套用到本次 OSL7 profiling。

## 8. 建议执行顺序

1. **补采阶段可区分的稳态 decode**：输出 128/256，预热模型与图后采多步，记录每步 scheduled token、活动请求数、prefill token 数、MTP acceptance。以服务端标记分离纯 decode 与混合 step，客户端“收到首 token”不足以保证其余 15 个请求已完成 prefill。
2. **并行推进可复现的单算子实验**：先测试 M=15900/15912、K=5120、N=17408 的效率下降，以及 Norm+DynamicQuant；分别记录完整组合耗时和数值误差。这里只规划工作，并未修改服务配置或执行算子替换。
3. **针对小 M 扩展实验集**：覆盖本次 M=8/16/20/24/28/32/36/40，并加入目标固定并发对应的主干验证、MTP draft shape。TP=2、W8A8、MTP 保持不变，图模式按真实 shape 验证回退。
4. **对 GDN、布局与通信做关键路径 A/B**：先选实际相邻且无额外消费者的子链；再以相同请求集验证 output tok/s、TPOT、ITL 分位数、TTFT、接受率、显存和正确性。

## 9. 明细与复现

- [BS16/BS32 比较及统计口径](../bs16-comparison.md)
- [框架调用](operator-details/01-rank-operator-details-analysis.md)、[算子统计](operator-details/02-rank-op-statistic-analysis.md)
- [Kernel/shape](kernel-and-task/01-kernel-details-analysis.md)、[Task 等待](kernel-and-task/02-task-time-analysis.md)
- [通信 JSON](communication/01-communication-json-analysis.md)、[通信矩阵](communication/02-communication-matrix-analysis.md)、[通信与 Stage](communication/03-step-trace-time-analysis.md)
- [Stage](step-trace-time/01-rank-step-trace-time-analysis.md)、[计算底稿及输入 SHA256](../bs16-profile-metrics.json)

从仓库根目录运行以下命令可重算底稿及校验；不访问 NPU、不重采数据、不自动改写分析判断。

```bash
python3 04-operator-profiling-data-analysis/03-summarize-profile.py \
  --output /tmp/bs16-profile-metrics.json
```
