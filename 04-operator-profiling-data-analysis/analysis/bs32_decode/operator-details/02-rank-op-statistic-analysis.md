# BS32 Decode op_statistic 分析

## 核心统计

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| 算子统计总时间 | 30,489.105 ms | 30,533.546 ms |
| 统计项数量 | 85 | 85 |
| MIX_AIC 占比 | 59.61% | 59.55% |
| AI_VECTOR_CORE 占比 | 30.93% | 30.97% |
| AI_CORE 占比 | 5.71% | 5.70% |
| AI_CPU 占比 | 3.61% | 3.64% |

## 热点算子

| OP Type | Core | Count | rank0 总耗时 | rank0 占比 | rank1 总耗时 | rank1 占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| `QuantBatchMatmulV3` | MIX_AIC | 5,940 | 11,144.613 ms | 36.55% | 11,143.445 ms | 36.50% |
| `recompute_w_u_fwd_kernel` | MIX_AIC | 960 | 1,668.180 ms | 5.47% | 1,669.515 ms | 5.47% |
| `Transpose` | AI_VECTOR_CORE | 21,180 | 1,526.848 ms | 5.01% | 1,534.748 ms | 5.03% |
| `MatMulV3` | AI_CORE | 1,020 | 1,392.192 ms | 4.57% | 1,392.657 ms | 4.56% |
| `merge_16x16_to_64x64_inverse_kernel` | MIX_AIC | 960 | 1,387.397 ms | 4.55% | 1,390.172 ms | 4.55% |
| `AddRmsNormBias` | AI_VECTOR_CORE | 3,618 | 1,350.305 ms | 4.43% | 1,353.066 ms | 4.43% |
| `FusedInferAttentionScore` | MIX_AIC | 513 | 1,234.113 ms | 4.05% | 1,239.932 ms | 4.06% |
| `ScatterUpdate` | AI_VECTOR_CORE | 1,728 | 1,196.218 ms | 3.92% | 1,200.465 ms | 3.93% |
| `DynamicQuant` | AI_VECTOR_CORE | 5,940 | 1,022.432 ms | 3.35% | 1,025.175 ms | 3.36% |
| `allreduceAicpuKernel` | AI_CPU | 3,726 | 983.073 ms | 3.22% | 993.234 ms | 3.25% |

## 结论

Decode 窗口由 W8A8 矩阵计算和 Vector 状态处理主导。`QuantBatchMatmulV3` 与 `DynamicQuant` 调用次数完全一致，是首要融合对象；Transpose、ScatterUpdate 和 Slice 的高频执行说明状态缓存和布局转换需要专项处理。
