# BS32 Decode Kernel Details 分析

| Kernel | Core | 调用次数 | rank0 总耗时 | rank0 平均 | rank0 最大 |
| --- | --- | ---: | ---: | ---: | ---: |
| `QuantBatchMatmulV3` | MIX_AIC | 5,940 | 11,144.613 ms | 1.876 ms | 5.862 ms |
| `recompute_w_u_fwd_kernel` | MIX_AIC | 960 | 1,668.180 ms | 1.738 ms | 2.174 ms |
| `MatMulV3` | AI_CORE | 1,020 | 1,392.192 ms | 1.365 ms | 2.850 ms |
| `merge_16x16_to_64x64_inverse_kernel` | MIX_AIC | 960 | 1,387.397 ms | 1.445 ms | 1.787 ms |
| `AddRmsNormBias` | AI_VECTOR_CORE | 3,618 | 1,350.305 ms | 0.373 ms | 0.619 ms |
| `FusedInferAttentionScore` | MIX_AIC | 513 | 1,234.113 ms | 2.406 ms | 4.484 ms |
| `ScatterUpdate` | AI_VECTOR_CORE | 1,728 | 1,196.218 ms | 0.692 ms | 1.691 ms |
| `DynamicQuant` | AI_VECTOR_CORE | 5,940 | 1,022.432 ms | 0.172 ms | 0.414 ms |
| `allreduceAicpuKernel` | AI_CPU | 3,726 | 983.073 ms | 0.264 ms | 7.019 ms |

量化 MatMul 是绝对热点；AllReduce 的平均较低但最大 7.019 ms，仍应结合 trace 检查长尾。rank1 各对应数值与 rank0 接近。
