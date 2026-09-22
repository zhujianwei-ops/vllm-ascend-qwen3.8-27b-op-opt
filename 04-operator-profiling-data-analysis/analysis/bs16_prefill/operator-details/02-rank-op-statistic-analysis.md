# BS16 Prefill op_statistic 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 1；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922015612919_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922015612919_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_prefill/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

本窗口实际主干 token 行数为 32,768，未证明覆盖了无缓存的 16×8192 token 计算；见主报告中的缓存限制。

## 统计完整性

| 指标 | rank0 | rank1 |
| --- | --- | --- |
| op_statistic 行数 | 67 | 67 |
| kernel_details 行数 | 17106 | 17106 |
| 底层累计时间 (ms) | 5,262.236 | 5,259.130 |

按 OP Type 合并不同 Core 行后，与 kernel_details 的次数、总时间逐项校验通过。累计时间会跨 stream 重叠，百分比的分母是底层累计时间，不能当作 Stage 百分比。

## 核心分布

| Core | rank0 (ms) | rank0 比例 | rank1 (ms) | rank1 比例 |
| --- | --- | --- | --- | --- |
| MIX_AIC | 2,264.474 | 43.03% | 2,264.392 | 43.06% |
| AI_CPU | 1,845.451 | 35.07% | 1,839.511 | 34.98% |
| AI_VECTOR_CORE | 929.818 | 17.67% | 933.671 | 17.75% |
| AI_CORE | 220.243 | 4.19% | 219.332 | 4.17% |
| MIX_AIV | 1.829 | 0.03% | 1.810 | 0.03% |
| DSA_SQE | 0.422 | 0.01% | 0.415 | 0.01% |

## 热点及调用频率

| OP Type | 次数/卡 | rank0 (ms) | rank0 比例 | rank1 (ms) | rank0 次/s |
| --- | --- | --- | --- | --- | --- |
| `allreduceAicpuKernel` | 552 | 1,716.705 | 32.62% | 1,710.757 | 136.35 |
| `QuantBatchMatmulV3` | 880 | 1,292.290 | 24.56% | 1,291.131 | 217.36 |
| `FusedInferAttentionScore` | 76 | 243.427 | 4.63% | 244.604 | 18.77 |
| `recompute_w_u_fwd_kernel` | 192 | 212.876 | 4.05% | 212.923 | 47.42 |
| `MatMulV3` | 204 | 177.423 | 3.37% | 176.824 | 50.39 |
| `merge_16x16_to_64x64_inverse_kernel` | 192 | 176.882 | 3.36% | 176.947 | 47.42 |
| `AddRmsNormBias` | 536 | 170.498 | 3.24% | 171.216 | 132.39 |
| `Transpose` | 2916 | 148.331 | 2.82% | 149.083 | 720.27 |
| `allgatherAicpuKernel` | 28 | 128.746 | 2.45% | 128.754 | 6.92 |
| `ChunkGatedDeltaRuleFwdH` | 192 | 128.190 | 2.44% | 128.216 | 47.42 |
| `DynamicQuant` | 880 | 127.547 | 2.42% | 128.148 | 217.36 |
| `SwiGlu` | 268 | 98.369 | 1.87% | 97.967 | 66.20 |
| `ChunkFwdO` | 192 | 91.444 | 1.74% | 91.722 | 47.42 |
| `CausalConv1d` | 192 | 82.983 | 1.58% | 82.499 | 47.42 |
| `Slice` | 2084 | 65.933 | 1.25% | 66.154 | 514.76 |
| `MatMulV2` | 208 | 42.820 | 0.81% | 42.509 | 51.38 |
| `TensorMove` | 636 | 39.761 | 0.76% | 40.230 | 157.10 |
| `solve_tril_16x16_kernel` | 192 | 37.915 | 0.72% | 37.922 | 47.42 |
| `_layer_norm_fwd_1pass_kernel_npu` | 192 | 36.604 | 0.70% | 36.905 | 47.42 |
| `chunk_scaled_dot_kkt_fwd_kernel` | 192 | 36.382 | 0.69% | 36.350 | 47.42 |

频率按 `调用数 / 本卡 Stage 秒数` 计算，是采集窗口中的调用密度，不是芯片时钟、输出 token/s 或稳态服务 QPS。热点应进一步按 shape 分桶；大 M 与小 M 的平均数混合后不代表任何一个具体 kernel。
