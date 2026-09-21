# Rank Step Trace Time 分析

## 1. 分析对象

本次分析使用以下两个 Rank 的 `step_trace_time.csv`：

```text
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
```

两份 CSV 都只有一条聚合记录，且 `Step` 字段为空。因此，本次结果反映的是一次 profiling 采集窗口的整体阶段耗时，不能用于判断多个 step 之间的稳定性。

## 2. 原始数据摘要

`step_trace_time.csv` 中的时间字段按微秒（us）记录。主要字段如下：

| 字段 | Rank0（Device 14） | Rank1（Device 15） |
| --- | ---: | ---: |
| Computing | 130,288.904 us / 130.289 ms | 130,296.500 us / 130.297 ms |
| Communication(Not Overlapped) | 172,050.947 us / 172.051 ms | 31,228.481 us / 31.228 ms |
| Overlapped | 16.420 us / 0.016 ms | 0 us / 0 ms |
| Communication | 172,067.366 us / 172.067 ms | 31,228.481 us / 31.228 ms |
| Free | 38,299.734 us / 38.300 ms | 179,154.407 us / 179.154 ms |
| Stage | 340,639.500 us / 340.640 ms | 340,679.500 us / 340.680 ms |
| Bubble | 0 us | 0 us |
| Preparing | 1,234 us / 1.234 ms | 1,249 us / 1.249 ms |

## 3. 分析结论

### 3.1 两个 Rank 的总阶段耗时基本一致

| Rank | Device | Stage |
| --- | ---: | ---: |
| Rank0 | 14 | 340.640 ms |
| Rank1 | 15 | 340.680 ms |

两者相差约 `0.040 ms`，相对差异约 `0.0117%`。从整体 Stage 看，两个 Rank 基本同步完成，没有发现明显的总阶段耗时失衡。

### 3.2 Computing 阶段基本一致

```text
Rank0: 130.289 ms
Rank1: 130.297 ms
```

计算耗时差异约 `0.008 ms`，说明当前两个 Rank 的计算工作量基本接近，主要差异不在计算阶段。

### 3.3 Rank0 的通信耗时明显高于 Rank1

```text
Rank0 Communication: 172.067 ms
Rank1 Communication: 31.228 ms
```

Rank0 的通信耗时约为 Rank1 的 `5.51` 倍，说明两个 Rank 的通信和等待分布并不对称。Rank0 更偏向于执行通信或等待通信完成，Rank1 则有更多时间处于空闲状态。

### 3.4 Rank1 的 Free 时间明显更长

```text
Rank0 Free: 38.300 ms
Rank1 Free: 179.154 ms
```

Rank1 比 Rank0 多约 `140.855 ms` 的 Free 时间。这可能表示 Rank1 在等待通信、同步、Host 调度或其他设备侧依赖。

仅凭 `step_trace_time.csv` 不能确定 Free 时间的具体原因，需要结合时间线进一步确认。

### 3.5 当前没有明显 Bubble

两个 Rank 的 `Bubble` 均为 `0`。这表示在该统计口径下没有记录到额外的流水线气泡，但不代表不存在同步等待或设备空闲，仍需结合 `Free` 和 `trace_view.json` 判断。

## 4. 当前可以得出的总体判断

```text
整体 Stage 耗时：两个 Rank 基本一致
计算耗时：两个 Rank 基本一致
通信耗时：Rank0 明显高于 Rank1
空闲耗时：Rank1 明显高于 Rank0
主要问题方向：通信、同步或 Rank 间工作分布，而不是计算耗时差异
```

这说明当前采集窗口中，两个 Rank 最终受到相近的 Stage 边界约束，但内部时间分布不同：Rank0 的通信时间更长，Rank1 的空闲等待时间更长。

## 5. 后续分析顺序

建议按以下顺序继续分析：

1. 查看 `communication.json`，确认通信操作和通信耗时。
2. 查看 `communication_matrix.json`，确认 Rank 或设备之间的通信关系和分布。
3. 使用 MindStudio Insight 打开 `trace_view.json`，确认 Rank1 的 Free 时间是否对应等待 Rank0 通信。
4. 查看 `task_time.csv`，确认是否存在 Task 排队、同步或调度延迟。
5. 再查看 `operator_details.csv` 和 `kernel_details.csv`，定位具体算子和 Kernel。

重点需要在时间线中确认：

- Rank0 的通信是否阻塞了后续计算。
- Rank1 的 Free 区间是否与 Rank0 的通信区间重合。
- 通信和计算是否能够重叠执行。
- 是否存在某个同步点导致两个 Rank 一起等待。

## 6. 数据解读限制

- 当前每个 Rank 只有一条聚合记录，不能判断 step 间稳定性。
- `Stage`、`Computing`、`Communication`、`Free` 等字段可能存在并行或重叠关系，不能简单相加后与 `Stage` 做等式校验。
- `Free` 不一定全部是纯设备空转，可能包含同步等待、调度间隙或统计口径中的未分类时间。
- 不应只根据某一个 Rank 下结论，应结合所有 Rank 的数据和时间线。
