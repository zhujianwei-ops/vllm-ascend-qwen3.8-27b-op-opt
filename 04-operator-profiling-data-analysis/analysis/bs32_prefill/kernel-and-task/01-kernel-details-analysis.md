# BS32 Prefill Kernel Details 分析

| Kernel | Core | 调用次数 | rank0 总耗时 | rank0 平均 | rank0 最大 |
| --- | --- | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | AI_CPU | 2,484 | 14,311.734 ms | 5.762 ms | 18.561 ms |
| `QuantBatchMatmulV3` | MIX_AIC | 3,960 | 10,056.275 ms | 2.539 ms | 4.833 ms |
| `recompute_w_u_fwd_kernel` | MIX_AIC | 864 | 1,645.891 ms | 1.905 ms | 2.278 ms |
| `merge_16x16_to_64x64_inverse_kernel` | MIX_AIC | 864 | 1,368.754 ms | 1.584 ms | 1.892 ms |
| `MatMulV3` | AI_CORE | 918 | 1,358.608 ms | 1.480 ms | 2.944 ms |
| `AddRmsNormBias` | AI_VECTOR_CORE | 2,412 | 1,321.963 ms | 0.548 ms | 0.648 ms |
| `FusedInferAttentionScore` | MIX_AIC | 342 | 1,119.719 ms | 3.274 ms | 4.311 ms |

rank1 的对应数值与 rank0 相差均在很小范围内。AllReduce 最大时延约 18.58 ms，是本轮最明显的通信长尾；量化 MatMul 的次数和总时间都高，适合做 shape 分组和融合基准。
