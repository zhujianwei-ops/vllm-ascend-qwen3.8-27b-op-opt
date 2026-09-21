# Qwen3.8-27B BS32 Decode 性能分析

分析日期：2026-09-21

## 分析对象

- 场景：随机数据集，32 个请求同时发起，输入长度 8192 token，`max_tokens=7`。
- 部署：TP=2、W8A8、MTP speculative tokens=3、`max_num_seqs=32`。
- Device 14 / rank0：`rank0_10893_20260921034601386_ascend_pt`。
- Device 15 / rank1：`rank1_10912_20260921034601386_ascend_pt`。

此窗口包含每个请求的 prefill、首 token 和后续 decode。相对 `max_tokens=1` 窗口新增的部分主要来自 decode，但它不是把 prefill 完全剥离后的纯 decode trace。

## 结论

两个 Rank 的 Stage 都约 33.034 s，计算约占 89.1%，非重叠通信约占 10.2%，通信重叠仅 0.266/0.136 ms。与 BS32 prefill 主导窗口相比，decode 窗口中的最大热点从 AllReduce 转为 `QuantBatchMatmulV3`，占约 36.5%；AllReduce 仍有 3,726 次调用，但占比降到约 3.2%。这表明在当前并发 32、MTP=3 条件下，decode 的首要算子优化对象是 W8A8 量化矩阵乘、动态量化和布局/状态更新链路。

## Stage 时间

| 指标 | rank0 / Device14 | rank1 / Device15 |
| --- | ---: | ---: |
| Computing | 29,434.442 ms | 29,467.802 ms |
| Communication(Not Overlapped) | 3,401.272 ms | 3,371.012 ms |
| Overlapped | 0.266 ms | 0.136 ms |
| Communication | 3,401.538 ms | 3,371.148 ms |
| Free | 197.871 ms | 194.815 ms |
| Preparing | 2.208 ms | 2.197 ms |
| Stage | 33,033.585 ms | 33,033.629 ms |

rank0/rank1 的 Stage 相差 0.044 ms，双卡计算负载均衡。计算占 89.10%/89.20%，通信占 10.30%/10.20%，但 overlap 不足 0.001%，说明通信仍几乎完全暴露在关键路径上。

## 算子统计

| 算子 | Core | 调用次数 | rank0 总耗时 | rank0 占比 | rank1 总耗时 | rank1 占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| `QuantBatchMatmulV3` | MIX_AIC | 5,940 | 11,144.613 ms | 36.55% | 11,143.445 ms | 36.50% |
| `recompute_w_u_fwd_kernel` | MIX_AIC | 960 | 1,668.180 ms | 5.47% | 1,669.515 ms | 5.47% |
| `Transpose` | AI_VECTOR_CORE | 21,180 | 1,526.848 ms | 5.01% | 1,534.748 ms | 5.03% |
| `MatMulV3` | AI_CORE | 1,020 | 1,392.192 ms | 4.57% | 1,392.657 ms | 4.56% |
| `merge_16x16_to_64x64_inverse_kernel` | MIX_AIC | 960 | 1,387.397 ms | 4.55% | 1,390.172 ms | 4.55% |
| `AddRmsNormBias` | AI_VECTOR_CORE | 3,618 | 1,350.305 ms | 4.43% | 1,353.066 ms | 4.43% |
| `FusedInferAttentionScore` | MIX_AIC | 513 | 1,234.113 ms | 4.05% | 1,239.932 ms | 4.06% |
| `ScatterUpdate` | AI_VECTOR_CORE | 1,728 | 1,196.218 ms | 3.92% | 1,200.465 ms | 3.93% |
| `ChunkGatedDeltaRuleFwdH` | MIX_AIC | 960 | 1,079.880 ms | 3.54% | 1,079.518 ms | 3.54% |
| `DynamicQuant` | AI_VECTOR_CORE | 5,940 | 1,022.432 ms | 3.35% | 1,025.175 ms | 3.36% |
| `allreduceAicpuKernel` | AI_CPU | 3,726 | 983.073 ms | 3.22% | 993.234 ms | 3.25% |

按核心类型，MIX_AIC 约 59.6%，AI_VECTOR_CORE 约 31.0%，AI_CORE 约 5.7%，AI_CPU 约 3.6%。这与 prefill 明显不同：量化计算和 Vector 状态处理是 decode 窗口的主导项，通信在底层算子统计中的直接占比降低，但 Stage 仍有约 3.4 s 通信时间。

## Decode 相对 Prefill 的变化

两组均包含一个 BS32 批次，且 decode 组比 prefill 组多 6 个输出 token。因此调用数和耗时不是纯粹的单 decode step 差分，但变化方向可用于识别 decode 专项热点。

| 算子 | Prefill 组调用次数 | Decode 组调用次数 | 变化 |
| --- | ---: | ---: | ---: |
| `QuantBatchMatmulV3` | 3,960 | 5,940 | +1,980 |
| `DynamicQuant` | 3,960 | 5,940 | +1,980 |
| `Transpose` | 13,150 | 21,180 | +8,030 |
| `ScatterUpdate` | 未进入热点 | 1,728 | decode 特有热点 |
| `allreduceAicpuKernel` | 2,484 | 3,726 | +1,242 |

`QuantBatchMatmulV3` 与 `DynamicQuant` 调用次数一一对应，适合进行前后融合。`Transpose` 和 `ScatterUpdate` 的高频执行说明 KV/状态缓存及布局转换在 decode 中值得专项检查。

## Kernel 和任务等待

量化矩阵乘平均约 1.876 ms，最大约 5.86 ms；`MatMulV3` 平均约 1.365 ms，`FusedInferAttentionScore` 平均约 2.41 ms。两卡数值接近，说明当前不是 Rank 间算力不均衡。

Task 等待统计如下，不能与 Stage 直接相加：

| Task 类型 | rank0 总耗时 | rank1 总耗时 | rank0 最大单次 |
| --- | ---: | ---: | ---: |
| `EVENT_WAIT` | 69,069.450 ms | 69,130.102 ms | 800.003 ms |
| `NOTIFY_WAIT_SQE` | 59,761.696 ms | 59,968.166 ms | 19.136 ms |
| `NOTIFY_WAIT` | 4,494.296 ms | 4,483.721 ms | 35.335 ms |

等待任务依然大量存在，而 Stage overlap 接近 0。应通过双 Rank trace 查看每个 decode 周期是否在 `ScatterUpdate`、DynamicQuant 或 AllReduce 前后插入了跨 stream 事件等待。

## 通信

每个 Rank 的 `communication.json` 中有 4,234 个 collective：

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| Collective Elapse | 10,945.798 ms | 10,898.068 ms |
| Transit | 301.661 ms | 299.218 ms |
| Wait | 10,547.269 ms | 10,502.189 ms |
| Synchronization | 10,545.696 ms | 10,498.385 ms |
| Idle | 96.868 ms | 96.661 ms |

`communication_matrix.json` 的 AllReduce 汇总 SIO 传输量约 366.1 GB、传输时间约 2.43--2.46 s，带宽约 149--150 GB/s；本地传输带宽约 282--284 GB/s。双卡链路吞吐对称，Wait 远大于 Transit，优化应聚焦调用粒度、HCCL 发起时机和计算通信重叠。

## 优化方向

1. 优先开发 `DynamicQuant + QuantBatchMatmulV3` 融合：当前 5,940 次一一配对，二者合计约 12,167 ms 的底层统计时间。目标应是减少量化中间写回、scale/offset 读写和 launch；先以该组合降时 15%--30% 做单算子验收。
2. 处理 decode 状态更新：`Transpose + ScatterUpdate + Slice` 高频执行。先检查是否可以在状态缓存布局中直接写入，消除 copy/transpose，再评估融合 Vector kernel。
3. GDN 子流程融合：`recompute_w_u_fwd_kernel`、`merge_16x16_to_64x64_inverse_kernel`、`ChunkGatedDeltaRuleFwdH` 和 `ChunkFwdO` 是持续热点，适合消除中间结果和融合连续算子。
4. 通信和计算重叠：通信约占 Stage 10%，但 overlap 接近零。即使不减少通信量，只要把 20%--40% 的非重叠通信隐藏在计算中，rank0 每批理论可释放约 680--1,360 ms，对 Stage 的直接改善约 2.1%--4.1%。
5. 检查 MTP 接受率：当前 7 token 窗口包含 speculative decode，实际 forward 数由接受率决定。优化前后需要同时记录 acceptance rate/length，避免把 MTP 行为变化误判为算子收益。

## 限制

- `max_tokens=7` 采集窗口包含 prefill，不是纯 decode 切片；应在后续流式请求的首 token 后启动 profiling，以取得更纯的 decode trace。
- 每个场景只有一个并发 32 批次，不能评价长稳态吞吐或延迟分位数。
- 不同 CSV 的时间范围和并行关系不同，本文没有将它们的总时间相加为端到端时延。
