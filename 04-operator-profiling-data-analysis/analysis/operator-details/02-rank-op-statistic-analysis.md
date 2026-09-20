# op_statistic 分析

## 1. 分析对象

本次分析使用以下两个 Rank 的 `op_statistic.csv`：

```text
04-operator-profiling-data-analysis/vllm_profile/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv
04-operator-profiling-data-analysis/vllm_profile/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv
```

每个文件包含 68 个算子统计项。`Ratio(%)` 表示该算子的耗时占当前统计范围总耗时的比例，每个 Rank 的比例之和约为 100%。

## 2. 字段说明

| 字段 | 含义 |
| --- | --- |
| `Device_id` | 执行该统计项的设备编号 |
| `OP Type` | 算子类型或底层操作名称 |
| `Core Type` | 执行该操作的核心类型 |
| `Count` | 调用次数 |
| `Total Time(us)` | 该算子所有调用的总耗时 |
| `Min Time(us)` | 单次调用最小耗时 |
| `Avg Time(us)` | 单次调用平均耗时 |
| `Max Time(us)` | 单次调用最大耗时 |
| `Ratio(%)` | 该算子总耗时占比 |

与 `operator_details.csv` 相比，`op_statistic.csv` 更适合回答：

- 哪类底层算子消耗时间最多？
- 时间主要消耗在 AI CPU、AI Core 还是 Vector Core？
- 算子调用次数是否过多？
- 单次耗时是否存在异常尖峰？

## 3. 核心统计结果

| 指标 | Rank0 | Rank1 |
| --- | ---: | ---: |
| Device | 14 | 15 |
| 算子统计项 | 68 | 68 |
| 统计总耗时 | 226.001 ms | 228.176 ms |
| Ratio 总和 | 100.001% | 100.002% |

两个 Rank 的统计总耗时相差约 `2.175 ms`，相对 Rank0 约 `0.96%`，整体较接近。

## 4. 按核心类型分析

### Rank0

| Core Type | 总耗时 | 占比 |
| --- | ---: | ---: |
| `AI_CPU` | 95.622 ms | 42.310% |
| `MIX_AIC` | 54.158 ms | 23.961% |
| `AI_VECTOR_CORE` | 50.298 ms | 22.259% |
| `AI_CORE` | 25.741 ms | 11.390% |
| `MIX_AIV` | 0.183 ms | 0.081% |

### Rank1

| Core Type | 总耗时 | 占比 |
| --- | ---: | ---: |
| `AI_CPU` | 97.734 ms | 42.832% |
| `MIX_AIC` | 54.705 ms | 23.975% |
| `AI_VECTOR_CORE` | 49.937 ms | 21.887% |
| `AI_CORE` | 25.605 ms | 11.222% |
| `MIX_AIV` | 0.195 ms | 0.086% |

### 核心类型结论

- 两个 Rank 的核心类型分布非常接近。
- `AI_CPU` 占比最高，约为 `42%`，主要由 `allreduceAicpuKernel` 贡献。
- `MIX_AIC` 约占 `24%`，主要来自 `QuantBatchMatmulV3`。
- `AI_CORE` 约占 `11%`，主要来自 `MatMulV2`。
- 当前不是单纯的 AI Core 计算瓶颈，通信相关 AI CPU 操作占据了最大耗时比例。

## 5. 主要算子热点

### Rank0

| OP Type | Core Type | Count | Total | Avg | Max | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | `AI_CPU` | 552 | 92.070 ms | 166.793 us | 378.671 us | 40.738% |
| `QuantBatchMatmulV3` | `MIX_AIC` | 880 | 42.339 ms | 48.112 us | 86.907 us | 18.734% |
| `MatMulV2` | `AI_CORE` | 412 | 25.741 ms | 62.478 us | 1,036.383 us | 11.390% |
| `Transpose` | `AI_VECTOR_CORE` | 853 | 13.543 ms | 15.877 us | 25.862 us | 5.992% |
| `ConcatD` | `AI_VECTOR_CORE` | 204 | 3.922 ms | 19.225 us | 25.302 us | 1.735% |
| `RecurrentGatedDeltaRule` | `AI_VECTOR_CORE` | 144 | 3.833 ms | 26.617 us | 30.883 us | 1.696% |
| `allgatherAicpuKernel` | `AI_CPU` | 28 | 3.552 ms | 126.868 us | 149.132 us | 1.572% |

### Rank1

| OP Type | Core Type | Count | Total | Avg | Max | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | `AI_CPU` | 552 | 94.123 ms | 170.513 us | 555.425 us | 41.250% |
| `QuantBatchMatmulV3` | `MIX_AIC` | 880 | 42.624 ms | 48.436 us | 88.867 us | 18.680% |
| `MatMulV2` | `AI_CORE` | 412 | 25.605 ms | 62.148 us | 1,032.003 us | 11.222% |
| `Transpose` | `AI_VECTOR_CORE` | 853 | 13.456 ms | 15.775 us | 24.922 us | 5.897% |
| `RecurrentGatedDeltaRule` | `AI_VECTOR_CORE` | 144 | 3.957 ms | 27.477 us | 29.742 us | 1.734% |
| `allgatherAicpuKernel` | `AI_CPU` | 28 | 3.611 ms | 128.950 us | 166.593 us | 1.582% |
| `ConcatD` | `AI_VECTOR_CORE` | 204 | 3.552 ms | 17.410 us | 21.402 us | 1.557% |

## 6. 重点结论

### 6.1 最大热点是 `allreduceAicpuKernel`

该算子是两个 Rank 的第一大热点：

```text
Rank0: 92.070 ms，占 40.738%
Rank1: 94.123 ms，占 41.250%
```

调用次数均为 `552`，说明它不是偶发的一次慢调用，而是频繁出现的通信相关操作。它在 AI CPU 上执行，是当前最应该优先分析的算子。

该结果与 `step_trace_time.csv` 和 `operator_details.csv` 的结论一致：当前性能优化重点应优先放在 AllReduce、HCCL 通信和同步时序上，而不是首先优化 MatMul。

### 6.2 `QuantBatchMatmulV3` 是第二大热点

```text
Rank0: 42.339 ms，占 18.734%，880 次
Rank1: 42.624 ms，占 18.680%，880 次
```

两个 Rank 的调用次数完全一致，平均耗时也接近，说明量化 Batch MatMul 的工作分布较稳定，当前没有明显的 Rank 间不均衡。

### 6.3 `MatMulV2` 的总耗时不高，但存在单次耗时尖峰

```text
Rank0 Avg: 62.478 us, Max: 1,036.383 us
Rank1 Avg: 62.148 us, Max: 1,032.003 us
```

平均耗时约 `62 us`，但最大耗时超过 `1 ms`。这说明少数调用可能对应不同输入形状、调度条件或同步影响。需要结合 `kernel_details.csv` 和 `trace_view.json` 确认这些慢调用是否集中在预热、编译或特定请求阶段。

### 6.4 Transpose 和 Vector Core 操作是次要开销

`Transpose` 占约 `5.9%`，调用次数为 `853`；`Index`、`Slice`、`DynamicQuant` 等操作各自占比不高，但调用次数较多。

这些操作暂时不是首要优化目标，除非时间线显示它们造成了明显的算子碎片化或设备空洞。

## 7. 两个 Rank 的差异

| 算子 | Rank0 | Rank1 | 差异判断 |
| --- | ---: | ---: | --- |
| `allreduceAicpuKernel` | 92.070 ms | 94.123 ms | Rank1 高 2.053 ms，基本接近 |
| `QuantBatchMatmulV3` | 42.339 ms | 42.624 ms | 基本一致 |
| `MatMulV2` | 25.741 ms | 25.605 ms | 基本一致 |
| `Transpose` | 13.543 ms | 13.456 ms | 基本一致 |
| `allgatherAicpuKernel` | 3.552 ms | 3.611 ms | 基本一致 |

`op_statistic.csv` 中两个 Rank 的主要算子统计非常接近。这一点与 `operator_details.csv` 中设备侧某些同步事件存在较大差异并不矛盾：前者是按底层 OP 类型聚合，后者包含父子算子、事件等待和设备任务层级，统计范围不同。

## 8. 推荐后续分析顺序

1. 在 `communication.json` 中确认 `allreduceAicpuKernel` 对应的通信操作和数据规模。
2. 在 `communication_matrix.json` 中检查 Rank 间通信分布。
3. 在 `trace_view.json` 中定位 552 次 AllReduce 的时间位置，判断通信是否阻塞计算。
4. 使用 `kernel_details.csv` 查看 AllReduce、量化 MatMul 和 MatMul 对应的底层 Kernel。
5. 对 `MatMulV2` 的最大耗时调用进行单独定位，确认约 1 ms 尖峰是否来自特殊输入或同步。
6. 如果优化通信后仍有瓶颈，再分析 `QuantBatchMatmulV3`、`Transpose` 和 Vector Core 操作。

## 9. 总结

当前 `op_statistic.csv` 给出的优先级为：

```text
1. allreduceAicpuKernel       通信热点，约占 41%
2. QuantBatchMatmulV3         量化矩阵计算，约占 19%
3. MatMulV2                   AI Core 计算，约占 11%
4. Transpose                  Vector Core 操作，约占 6%
```

前四类算子合计占约 `76%` 的统计耗时。结合前面的 Step 和 Operator 分析，当前最值得优先验证的是通信和同步时序，而不是直接对矩阵计算 Kernel 做优化。
