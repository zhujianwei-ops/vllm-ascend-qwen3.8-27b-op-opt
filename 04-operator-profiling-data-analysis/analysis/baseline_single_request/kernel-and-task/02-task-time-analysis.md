# Task Time 分析

## 1. 分析对象

```text
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/task_time.csv
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/task_time.csv
```

`task_time.csv` 提供设备 Task 类型、开始时间、结束时间、执行时长、Stream 和 Task ID，适合判断任务调度、事件等待、通知等待、内存拷贝和设备任务之间的依赖关系。

## 2. 数据规模

| 指标 | Rank0 | Rank1 |
| --- | ---: | ---: |
| Task 记录数 | 48,490 | 48,489 |
| Task Time 总和 | 1,800.603 ms | 1,040.721 ms |
| Task Time 为 0 的记录数 | 2,823 | 3,131 |

Task 时间总和不是一次请求的墙上时钟耗时，因为 Task 可能并行执行或处于等待状态，不能直接累加为服务端总耗时。

## 3. Rank0 Task 类型统计

| Task 类型 | 调用次数 | 总耗时 | 平均耗时 | 最大耗时 |
| --- | ---: | ---: | ---: | ---: |
| `NOTIFY_WAIT_SQE` | 10,740 | 581.425 ms | 54.136 us | 7,508.921 us |
| `EVENT_WAIT` | 1,940 | 577.418 ms | 297.638 us | 23,190.296 us |
| `NOTIFY_WAIT` | 1,752 | 409.182 ms | 233.551 us | 24,127.351 us |
| `AI_CPU` | 580 | 95.622 ms | 164.865 us | 378.670 us |
| `MIX_AIC` | 1,388 | 54.157 ms | 39.018 us | 86.907 us |
| `AI_VECTOR_CORE` | 8,472 | 50.296 ms | 5.937 us | 120.849 us |
| `AI_CORE` | 412 | 25.741 ms | 62.477 us | 1,036.383 us |

## 4. Rank1 Task 类型统计

| Task 类型 | 调用次数 | 总耗时 | 平均耗时 | 最大耗时 |
| --- | ---: | ---: | ---: | ---: |
| `EVENT_WAIT` | 1,939 | 361.217 ms | 186.290 us | 23,103.809 us |
| `NOTIFY_WAIT` | 1,752 | 262.830 ms | 150.017 us | 24,102.249 us |
| `NOTIFY_WAIT_SQE` | 10,740 | 181.987 ms | 16.945 us | 1,034.263 us |
| `AI_CPU` | 580 | 97.734 ms | 168.506 us | 555.424 us |
| `MIX_AIC` | 1,388 | 54.704 ms | 39.412 us | 88.867 us |
| `AI_VECTOR_CORE` | 8,472 | 49.935 ms | 5.894 us | 120.629 us |
| `AI_CORE` | 412 | 25.605 ms | 62.148 us | 1,032.002 us |

## 5. Task 层结论

### 5.1 同步等待是主要 Task 开销

`NOTIFY_WAIT_SQE`、`EVENT_WAIT` 和 `NOTIFY_WAIT` 是两卡最主要的 Task 类型，说明任务依赖和同步等待是当前重要分析方向。

Rank0 的等待类 Task 总耗时更高：

```text
Rank0:
  NOTIFY_WAIT_SQE: 581.425 ms
  EVENT_WAIT:      577.418 ms
  NOTIFY_WAIT:     409.182 ms

Rank1:
  EVENT_WAIT:      361.217 ms
  NOTIFY_WAIT:     262.830 ms
  NOTIFY_WAIT_SQE: 181.987 ms
```

Rank1 的等待类 Task 也很多，但单次等待和总耗时分布不同。

### 5.2 存在较长等待长尾

```text
EVENT_WAIT 最大耗时：约 23.1 ms
NOTIFY_WAIT 最大耗时：约 24.1 ms
NOTIFY_WAIT_SQE 最大耗时：Rank0 约 7.5 ms
```

这说明至少存在少量较长等待区间，可能与通信完成事件、不同 Stream 之间的依赖或 Rank 间通知同步有关。

### 5.3 计算 Task 在两个 Rank 间基本一致

两个 Rank 的计算 Task 统计接近：

```text
AI_CPU:
  Rank0: 95.622 ms
  Rank1: 97.734 ms

MIX_AIC:
  Rank0: 54.157 ms
  Rank1: 54.704 ms

AI_CORE:
  Rank0: 25.741 ms
  Rank1: 25.605 ms
```

因此当前 Rank 差异主要来自等待、通信和同步分布，而不是计算负载不均衡。

## 6. 是否存在任务排队和同步等待

从 Task 类型可以明确看到：

```text
NOTIFY_WAIT_SQE
EVENT_WAIT
NOTIFY_WAIT
EVENT_RECORD
NOTIFY_RECORD
EVENT_RESET
```

其中等待类 Task 的最大耗时超过 23 ms，说明存在需要进一步定位的同步长尾。是否是实际任务排队，不能只看统计总和，还需要使用 `trace_view.json` 对照 Task 的开始时间、结束时间和 Stream 关系。

## 7. 与 Step 和 Kernel 分析的关联

前面的 `step_trace_time.csv` 显示：

```text
Rank0 Communication: 172.067 ms
Rank1 Communication: 31.228 ms
Rank0 Free: 38.300 ms
Rank1 Free: 179.154 ms
```

结合 Task 数据可以推断：

- Rank0 的通信和等待类 Task 更重。
- Rank1 的计算 Task 与 Rank0 接近，但可能在部分通信阶段等待其他 Rank。
- `EVENT_WAIT` 和 `NOTIFY_WAIT` 可能是通信完成和跨 Stream 依赖的具体表现。

这里的“可能”需要通过 `trace_view.json` 进行时序验证，不能仅依据 Task 汇总表确定因果关系。

## 8. 后续分析顺序

1. 在 `trace_view.json` 中定位 `EVENT_WAIT`、`NOTIFY_WAIT` 和 `NOTIFY_WAIT_SQE`。
2. 对照 `allreduceAicpuKernel` 和 `hcom_allReduce_*` 的开始、结束时间。
3. 检查 Rank0 与 Rank1 是否存在同一通信操作开始或完成时间不一致。
4. 检查等待是否集中在某个 Stream、请求阶段或特定通信操作之后。
5. 使用 `communication.json` 和 `communication_matrix.json` 确认通信方向与数据量。

## 9. 总结

```text
第一优先级：EVENT_WAIT、NOTIFY_WAIT、NOTIFY_WAIT_SQE
第二优先级：通信完成事件和跨 Stream 依赖
第三优先级：Rank 间通信完成时间差异
```

当前 `task_time.csv` 支持的主要结论是：算子慢的底层原因更可能包含通信同步和任务等待，而不是单纯的 AI Core 计算不足。最终应使用 `trace_view.json` 完成时序确认。

