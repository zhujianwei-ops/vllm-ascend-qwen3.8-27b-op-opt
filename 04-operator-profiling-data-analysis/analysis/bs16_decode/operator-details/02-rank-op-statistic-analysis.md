# BS16 Decode op_statistic 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 7；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922014740695_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922014740695_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_decode/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

这是含长输入 prefill、混合调度与短 M 执行的完整请求窗口；不能作为纯 decode 或固定执行 batch=16 的统计。

## 统计完整性

| 指标 | rank0 | rank1 |
| --- | --- | --- |
| op_statistic 行数 | 85 | 85 |
| kernel_details 行数 | 72206 | 72206 |
| 底层累计时间 (ms) | 15,495.185 | 15,494.807 |

按 OP Type 合并不同 Core 行后，与 kernel_details 的次数、总时间逐项校验通过。累计时间会跨 stream 重叠，百分比的分母是底层累计时间，不能当作 Stage 百分比。

## 核心分布

| Core | rank0 (ms) | rank0 比例 | rank1 (ms) | rank1 比例 |
| --- | --- | --- | --- | --- |
| MIX_AIC | 9,086.476 | 58.64% | 9,092.214 | 58.68% |
| AI_VECTOR_CORE | 4,707.879 | 30.38% | 4,717.991 | 30.45% |
| AI_CORE | 894.158 | 5.77% | 891.316 | 5.75% |
| AI_CPU | 783.758 | 5.06% | 770.352 | 4.97% |
| MIX_AIV | 16.648 | 0.11% | 16.668 | 0.11% |
| DSA_SQE | 6.265 | 0.04% | 6.266 | 0.04% |

## 热点及调用频率

| OP Type | 次数/卡 | rank0 (ms) | rank0 比例 | rank1 (ms) | rank0 次/s |
| --- | --- | --- | --- | --- | --- |
| `QuantBatchMatmulV3` | 3740 | 5,573.330 | 35.97% | 5,574.051 | 225.79 |
| `recompute_w_u_fwd_kernel` | 528 | 835.853 | 5.39% | 836.650 | 31.88 |
| `Transpose` | 11483 | 760.788 | 4.91% | 765.547 | 693.25 |
| `allreduceAicpuKernel` | 2346 | 708.092 | 4.57% | 694.713 | 141.63 |
| `MatMulV3` | 561 | 697.963 | 4.50% | 696.037 | 33.87 |
| `merge_16x16_to_64x64_inverse_kernel` | 528 | 694.882 | 4.48% | 695.919 | 31.88 |
| `AddRmsNormBias` | 2278 | 678.161 | 4.38% | 679.813 | 137.53 |
| `FusedInferAttentionScore` | 323 | 623.707 | 4.03% | 626.112 | 19.50 |
| `ScatterUpdate` | 864 | 583.626 | 3.77% | 580.562 | 52.16 |
| `ChunkGatedDeltaRuleFwdH` | 528 | 524.407 | 3.38% | 524.480 | 31.88 |
| `DynamicQuant` | 3740 | 513.092 | 3.31% | 515.124 | 225.79 |
| `SwiGlu` | 1139 | 403.778 | 2.61% | 403.009 | 68.76 |
| `Slice` | 8149 | 368.837 | 2.38% | 370.326 | 491.97 |
| `ChunkFwdO` | 528 | 358.162 | 2.31% | 358.494 | 31.88 |
| `CausalConv1d` | 1248 | 336.450 | 2.17% | 336.783 | 75.34 |
| `MatMulV2` | 1190 | 196.195 | 1.27% | 195.279 | 71.84 |
| `solve_tril_16x16_kernel` | 528 | 150.611 | 0.97% | 150.620 | 31.88 |
| `_layer_norm_fwd_1pass_kernel_npu` | 816 | 146.058 | 0.94% | 145.755 | 49.26 |
| `chunk_scaled_dot_kkt_fwd_kernel` | 528 | 139.685 | 0.90% | 139.725 | 31.88 |
| `TensorMove` | 1783 | 127.139 | 0.82% | 127.197 | 107.64 |

频率按 `调用数 / 本卡 Stage 秒数` 计算，是采集窗口中的调用密度，不是芯片时钟、输出 token/s 或稳态服务 QPS。热点应进一步按 shape 分桶；大 M 与小 M 的平均数混合后不代表任何一个具体 kernel。
