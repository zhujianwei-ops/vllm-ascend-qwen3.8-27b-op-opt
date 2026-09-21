# BS32 Prefill op_statistic 分析

## 核心统计

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| 算子统计总时间 | 40,924.745 ms | 40,916.580 ms |
| 统计项数量 | 67 | 67 |
| MIX_AIC 占比 | 41.20% | 41.22% |
| AI_CPU 占比 | 37.72% | 37.66% |
| AI_VECTOR_CORE 占比 | 17.03% | 17.08% |
| AI_CORE 占比 | 4.03% | 4.02% |

算子统计总时间含并行核和辅助任务，不能直接作为 28.814 s Stage 的端到端耗时。

## 热点算子

| OP Type | Core | Count | rank0 总耗时 | rank0 占比 | rank1 总耗时 | rank1 占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | AI_CPU | 2,484 | 14,311.734 ms | 34.97% | 14,282.408 ms | 34.91% |
| `QuantBatchMatmulV3` | MIX_AIC | 3,960 | 10,056.275 ms | 24.57% | 10,053.264 ms | 24.57% |
| `recompute_w_u_fwd_kernel` | MIX_AIC | 864 | 1,645.891 ms | 4.02% | 1,646.707 ms | 4.03% |
| `merge_16x16_to_64x64_inverse_kernel` | MIX_AIC | 864 | 1,368.754 ms | 3.35% | 1,370.572 ms | 3.35% |
| `MatMulV3` | AI_CORE | 918 | 1,358.608 ms | 3.32% | 1,356.122 ms | 3.31% |
| `AddRmsNormBias` | AI_VECTOR_CORE | 2,412 | 1,321.963 ms | 3.23% | 1,325.057 ms | 3.24% |
| `allgatherAicpuKernel` | AI_CPU | 126 | 1,125.324 ms | 2.75% | 1,125.304 ms | 2.75% |
| `FusedInferAttentionScore` | MIX_AIC | 342 | 1,119.719 ms | 2.74% | 1,126.879 ms | 2.75% |

## 结论

两卡的算子时间几乎对称。AllReduce 和 QuantBatchMatmul 合计约占 59.5%，是 prefill 优化的前两项。GDN 子流程、量化、Transpose 和 AddRmsNormBias 是下一层高频热点。
