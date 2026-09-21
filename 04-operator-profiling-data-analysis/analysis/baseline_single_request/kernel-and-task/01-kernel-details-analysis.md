# Kernel Details 分析

## 1. 分析对象

```text
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
```

`kernel_details.csv` 提供 Kernel 名称、类型、Accelerator Core、执行时长、等待时长、输入输出形状和硬件执行信息，适合从算子继续下钻到具体底层 Kernel。

## 2. 数据规模

| 指标 | Rank0 | Rank1 |
| --- | ---: | ---: |
| Kernel 记录数 | 11,450 | 11,450 |
| Kernel Duration 总和 | 398.069 ms | 259.405 ms |
| Kernel Wait Time 总和 | 93.463 ms | 179.900 ms |
| 单条记录平均 Duration | 34.766 us | 22.655 us |

这些总和不是一次请求的墙上时钟耗时。Kernel 之间存在并行、嵌套和等待，不能将所有行直接相加作为服务端总耗时。

## 3. 主要 Kernel 热点

### Rank0

| Kernel | 调用次数 | 总耗时 | 平均耗时 | 最大耗时 | Wait 总和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | 552 | 92.070 ms | 166.793 us | 378.671 us | 16.011 ms |
| `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3` | 880 | 42.339 ms | 48.112 us | 86.907 us | 1.113 ms |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 412 | 25.741 ms | 62.478 us | 1,036.383 us | 1.349 ms |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 639 | 9.093 ms | 14.230 us | 25.162 us | 1.419 ms |
| `allgatherAicpuKernel` | 28 | 3.552 ms | 126.868 us | 149.132 us | 3.169 ms |

此外，Rank0 存在多个 `hcom_allReduce_*` 单次 Kernel，单次耗时约 2.9-7.5 ms，需结合 Trace 确认其具体阶段。

### Rank1

| Kernel | 调用次数 | 总耗时 | 平均耗时 | 最大耗时 | Wait 总和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `allreduceAicpuKernel` | 552 | 94.123 ms | 170.513 us | 555.425 us | 0.956 ms |
| `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3` | 880 | 42.624 ms | 48.436 us | 88.867 us | 2.538 ms |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 412 | 25.605 ms | 62.148 us | 1,032.003 us | 6.652 ms |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 639 | 9.244 ms | 14.466 us | 21.862 us | 11.565 ms |
| `allgatherAicpuKernel` | 28 | 3.611 ms | 128.950 us | 166.594 us | 0.002 ms |
| `aclnnRecurrentGatedDeltaRule_RecurrentGatedDeltaRule_RecurrentGatedDeltaRule` | 144 | 3.957 ms | 27.477 us | 29.743 us | 0.009 ms |

## 4. Kernel 层结论

### 4.1 通信 Kernel 是第一大热点

`allreduceAicpuKernel` 是两个 Rank 的最大 Kernel：

```text
Rank0: 92.070 ms，552 次，平均 166.793 us
Rank1: 94.123 ms，552 次，平均 170.513 us
```

这与 `op_statistic.csv` 中该算子占约 41% 的结论一致。当前最重要的底层热点是 AllReduce，而不是 MatMul。

### 4.2 `QuantBatchMatmulV3` 主要是调用次数多

```text
调用次数：880
平均耗时：约 48 us
总耗时：约 42 ms
```

单次耗时不高，但调用次数最多，属于高频小 Kernel。后续可以检查算子融合、小批次计算和中间张量转换。

### 4.3 `MatMulV2` 存在少量长尾调用

`MatMulV2` 调用 412 次，平均耗时约 62 us，但最大耗时超过 1 ms：

```text
Rank0 Max: 1,036.383 us
Rank1 Max: 1,032.003 us
```

需要结合输入 Shape、时间线和等待信息，确认长尾是否由同步或特殊输入造成。

### 4.4 存在大量短 Kernel

11,450 条 Kernel 记录中有大量单次几微秒甚至更低的 Vector Core、事件和辅助操作。这可能带来 Kernel 启动开销、时间线碎片化以及计算通信重叠不足，但是否值得优化需要以 `trace_view.json` 为准。

## 5. 与前面分析的关联

- `allreduceAicpuKernel` 是 `op_statistic.csv` 的第一大热点。
- `HcclAllreduce`、`all_reduce` 和相关通知等待共同构成通信链路。
- 两个 Rank 的主要计算 Kernel 统计接近，差异主要应从通信时序和同步等待解释。

## 6. 后续分析顺序

1. 在 `trace_view.json` 中定位 `allreduceAicpuKernel` 和 `hcom_allReduce_*`。
2. 检查通信是否阻塞计算，是否存在明显时间线空洞。
3. 对 `MatMulV2` 的约 1 ms 长尾调用检查输入 Shape 和前后同步。
4. 对 880 次 `QuantBatchMatmulV3` 调用检查是否存在小 Kernel 过多或融合不足。
5. 结合 `task_time.csv` 判断 Kernel 等待是否来自任务依赖。

