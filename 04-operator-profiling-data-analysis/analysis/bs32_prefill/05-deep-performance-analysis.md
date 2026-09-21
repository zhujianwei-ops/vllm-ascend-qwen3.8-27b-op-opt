# Qwen3.8-27B BS32 Prefill 性能分析

分析日期：2026-09-21

## 分析对象

- 场景：随机数据集，32 个请求同时发起，输入长度 8192 token，`max_tokens=1`。
- 部署：TP=2、W8A8、MTP speculative tokens=3、`max_num_seqs=32`。
- Device 14 / rank0：`rank0_10893_20260921034824466_ascend_pt`。
- Device 15 / rank1：`rank1_10912_20260921034824466_ascend_pt`。

本轮以一个并发 32 的请求批次作为采集窗口。`max_tokens=1` 使窗口主要反映 prefill，但仍包含首 token 的解码、调度和通信，不能把数据解释为纯 prefill 算子耗时。

## 结论

两个 Rank 的总 Stage 基本一致，约 28.814 s。计算占约 88.3%，非重叠通信占约 11.2%，通信和计算重叠为 0。此场景的首要热点是 `allreduceAicpuKernel`，约占底层算子时间 35%；其次是 `QuantBatchMatmulV3`，约占 24.6%。通信的物理 SIO 带宽约 152 GB/s，但 `communication.json` 的通信总时间几乎全部标为 Wait/同步，因此优先处理通信启动、依赖和重叠，而不是先做链路带宽调优。

## Stage 时间

| 指标 | rank0 / Device14 | rank1 / Device15 |
| --- | ---: | ---: |
| Computing | 25,425.934 ms | 25,447.338 ms |
| Communication(Not Overlapped) | 3,241.190 ms | 3,215.864 ms |
| Overlapped | 0 ms | 0 ms |
| Communication | 3,241.190 ms | 3,215.864 ms |
| Free | 146.901 ms | 150.970 ms |
| Preparing | 2.199 ms | 2.299 ms |
| Stage | 28,814.026 ms | 28,814.173 ms |

以 Stage 为分母，rank0/rank1 的计算占比为 88.24%/88.31%，通信占比为 11.25%/11.16%，Free 均约 0.51%。两卡 Stage 相差 0.147 ms，负载均衡正常；但 0 ms overlap 表明当前不能在计算期间隐藏通信时间。

## 算子统计

| 算子 | Core | 调用次数 | rank0 总耗时 | rank0 占比 | rank1 总耗时 | rank1 占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | AI_CPU | 2,484 | 14,311.734 ms | 34.97% | 14,282.408 ms | 34.91% |
| `QuantBatchMatmulV3` | MIX_AIC | 3,960 | 10,056.275 ms | 24.57% | 10,053.264 ms | 24.57% |
| `recompute_w_u_fwd_kernel` | MIX_AIC | 864 | 1,645.891 ms | 4.02% | 1,646.707 ms | 4.03% |
| `merge_16x16_to_64x64_inverse_kernel` | MIX_AIC | 864 | 1,368.754 ms | 3.35% | 1,370.572 ms | 3.35% |
| `MatMulV3` | AI_CORE | 918 | 1,358.608 ms | 3.32% | 1,356.122 ms | 3.31% |
| `AddRmsNormBias` | AI_VECTOR_CORE | 2,412 | 1,321.963 ms | 3.23% | 1,325.057 ms | 3.24% |
| `allgatherAicpuKernel` | AI_CPU | 126 | 1,125.324 ms | 2.75% | 1,125.304 ms | 2.75% |
| `FusedInferAttentionScore` | MIX_AIC | 342 | 1,119.719 ms | 2.74% | 1,126.879 ms | 2.75% |
| `ChunkGatedDeltaRuleFwdH` | MIX_AIC | 864 | 1,070.782 ms | 2.62% | 1,072.078 ms | 2.62% |
| `Transpose` | AI_VECTOR_CORE | 13,150 | 1,034.558 ms | 2.53% | 1,037.071 ms | 2.54% |
| `DynamicQuant` | AI_VECTOR_CORE | 3,960 | 984.813 ms | 2.41% | 986.922 ms | 2.41% |

两卡的主算子耗时和调用次数接近，未出现单卡算力负载失衡。底层统计总时间约 40.925/40.917 s；该总和含并行核、通信和辅助任务，不能与 28.814 s 的 Stage 直接相加或相除。

按核心类型，MIX_AIC 约 41.2%，AI_CPU 约 37.7%，AI_VECTOR_CORE 约 17.1%，AI_CORE 约 4.0%。因此 prefill 是量化矩阵计算和通信共同主导的场景，通用 BF16 MatMul 并非首要优化入口。

## Kernel 和任务等待

`kernel_details.csv` 与 `op_statistic.csv` 的热点一致：AllReduce、QuantBatchMatmul、GDN 相关核和 `MatMulV3` 排在前列。量化矩阵乘 3,960 次、平均约 2.539 ms；AllReduce 2,484 次、平均约 5.75 ms，最大约 18.58 ms，存在明显长尾。

Task 统计的等待类时间总和不能作为墙钟时间相加，但可以用于定位依赖强度：

| Task 类型 | rank0 总耗时 | rank1 总耗时 | rank0 最大单次 |
| --- | ---: | ---: | ---: |
| `EVENT_WAIT` | 66,961.760 ms | 66,899.020 ms | 874.689 ms |
| `NOTIFY_WAIT_SQE` | 52,318.758 ms | 52,311.914 ms | 17.107 ms |
| `NOTIFY_WAIT` | 18,210.874 ms | 18,150.985 ms | 19.805 ms |

长等待与 Step 中 0 ms 的通信计算 overlap 一致。下一步应在双 Rank `trace_view.json` 对齐 `EVENT_WAIT`、`NOTIFY_WAIT` 和同一序号的 AllReduce，确定是计算到达不齐、HCCL 启动串行，还是跨 stream 事件链阻塞。

## 通信

`communication.json` 每个 Rank 记录 2,611 个 collective。汇总口径如下：

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| Collective Elapse | 6,482.380 ms | 6,431.729 ms |
| Wait | 6,481.444 ms | 6,430.753 ms |
| Synchronization | 6,481.444 ms | 6,430.753 ms |
| Idle | 0.937 ms | 0.976 ms |

该文件的 Elapse 几乎全是 Wait/同步，且和 Step 的 Communication 统计范围不同，不能直接比较绝对总值。`communication_matrix.json` 显示 AllReduce 汇总 SIO 传输量约 356.5 GB、传输时间约 2.33--2.34 s、带宽约 152--153 GB/s；本地传输约 713.0 GB、带宽约 286 GB/s。两卡带宽基本对称，当前优先级是降低同步等待并形成 overlap。

## 优化方向

1. 以 AllReduce 为第一优先级：按消息大小和层位置统计 2,484 次调用，评估 bucket 合并、提前发起和通信 stream 编排。若能将非重叠通信降低 20%--40%，按 rank0 Stage 估算可释放约 648--1,296 ms，整批 Stage 理论改善约 2.3%--4.5%。
2. 优化 `DynamicQuant + QuantBatchMatmulV3`：二者同为 3,960 次，适合检查 per-token 量化、NZ 权重布局和量化 MatMul 融合。若该组合降低 15%--30%，按总计 11,041 ms 的底层统计口径估算可减约 1.66--3.31 s；实际 Stage 收益须用 A/B trace 测量。
3. 融合 GDN 子流程：`recompute_w_u_fwd_kernel`、`merge_16x16_to_64x64_inverse_kernel`、`ChunkGatedDeltaRuleFwdH`、`ChunkFwdO` 都高频执行，可通过中间张量消除降低读写和 launch。
4. 处理布局碎片：`Transpose` 13,150 次、`Slice` 11,394 次。优先建立跨算子 layout contract，避免为了消费者而反复转置和切片。

## 限制

- 本次只采集一个并发 32 批次，不能反映长时间稳态波动或 P99。
- `max_tokens=1` 仍包含首 token decode；本文将其称为 prefill 主导窗口。
- 父子算子、并行 Task 和通信统计口径不同，文中未将各 CSV 的总时间直接作为端到端耗时。
