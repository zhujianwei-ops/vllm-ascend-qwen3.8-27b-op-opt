# Operator Details 分析

## 1. 分析对象

本次分析使用以下两个 Rank 的 `operator_details.csv`：

```text
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv
04-operator-profiling-data-analysis/vllm_profile/baseline_single_request/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv
```

文件包含约 3.76 万条记录、190 个不同的算子名称。由于同一个算子可能嵌套调用其他算子，`Host Total Duration` 和 `Device Total Duration` 存在父子层级关系，不能把所有行直接相加后当作真实总耗时。

## 2. 字段说明

| 字段 | 含义 | 分析用途 |
| --- | --- | --- |
| `Name` | 算子或运行时事件名称 | 定位热点算子、通信和同步事件 |
| `Input Shapes` | 输入张量形状 | 判断算子是否受输入规模影响 |
| `Call Stack` | Host 调用栈 | 追溯算子来源和调用路径 |
| `Host Self Duration(us)` | Host 侧自身耗时 | 查看算子自身在 Host 上的调度开销 |
| `Host Total Duration(us)` | Host 侧总耗时 | 包含子调用的 Host 耗时，适合查看调用路径热点 |
| `Device Self Duration(us)` | Device 侧自身耗时 | 查看算子自身设备执行时间 |
| `Device Total Duration(us)` | Device 侧总耗时 | 包含设备侧子任务，适合定位设备执行热点 |
| `Device Self Duration With AICore(us)` | AICore 自身耗时 | 判断 AICore 计算耗时 |
| `Device Total Duration With AICore(us)` | 包含子任务的 AICore 耗时 | 进一步分析 AICore 计算热点 |

本文件的时间单位为微秒（us）。

## 3. 数据规模和总览

| 指标 | Rank0 | Rank1 |
| --- | ---: | ---: |
| 记录数 | 37,630 | 37,638 |
| 不同算子名称数 | 190 | 190 |
| Host Self 总和 | 277.204 ms | 289.129 ms |
| Host Total 总和 | 693.310 ms | 727.526 ms |
| Device Self 总和 | 720.723 ms | 364.334 ms |
| Device Total 总和 | 2,499.815 ms | 1,140.609 ms |
| Device Total With AICore 总和 | 62.174 ms | 61.671 ms |

这里的总和用于比较两个 Rank 的记录分布，不代表一次请求的墙上时钟耗时，因为不同记录之间可能存在嵌套或并行执行。

## 4. Host 侧热点

两个 Rank 的 Host 热点基本一致：

| 算子 | Rank0 Host Total | Rank1 Host Total | 调用次数 |
| --- | ---: | ---: | ---: |
| `npu_fx_compiler inference` | 221.427 ms | 232.936 ms | 13 |
| `vllm::qwen_gdn_attention_core` | 109.901 ms | 115.756 ms | 48 |
| `ChunkGatedDeltaRuleFunction` | 72.704 ms | 75.750 ms | 48 |
| `aten::copy_` | 23.753 ms | 25.069 ms | 2,160 |
| `aten::clone` | 17.652 ms | 18.514 ms | 1,089 |
| `Event::synchronize` | 17.641 ms | 16.448 ms | 15 |
| `empty_tensor` | 16.653 ms | 17.177 ms | 6,207 |
| `vllm::all_reduce` | 13.903 ms | 15.169 ms | 168 |

### Host 侧结论

- `npu_fx_compiler inference` 是 Host Total 最大项，但只有 13 次调用，可能包含编译或编译相关运行时开销，需要结合是否为预热阶段判断。
- `vllm::qwen_gdn_attention_core` 和 `ChunkGatedDeltaRuleFunction` 是主要模型路径算子，两个 Rank 的耗时接近。
- `aten::copy_`、`aten::clone`、`empty_tensor` 调用次数较多，说明 Host 侧存在较多数据拷贝、张量复制和临时张量创建活动。
- Host 侧没有出现某个 Rank 明显失衡的情况，Rank1 只比 Rank0 略高。

## 5. Device 侧热点

### Rank0

| 算子 | Device Total | 调用次数 |
| --- | ---: | ---: |
| `wait_event` | 453.790 ms | 404 |
| `npu_fx_compiler inference` | 425.793 ms | 13 |
| `vllm::all_reduce` | 394.001 ms | 168 |
| `HcclAllreduce` | 380.305 ms | 336 |
| `c10d::allreduce_` | 233.346 ms | 168 |
| `Event::wait` | 163.126 ms | 12 |
| `vllm::all_gather` | 93.986 ms | 28 |
| `c10d::_allgather_base_` | 90.334 ms | 28 |

### Rank1

| 算子 | Device Total | 调用次数 |
| --- | ---: | ---: |
| `wait_event` | 237.649 ms | 404 |
| `Event::wait` | 137.869 ms | 12 |
| `npu_fx_compiler inference` | 113.910 ms | 13 |
| `HcclAllreduce` | 104.458 ms | 336 |
| `vllm::all_gather` | 87.389 ms | 28 |
| `c10d::_allgather_base_` | 86.710 ms | 28 |
| `vllm::all_reduce` | 69.202 ms | 168 |
| `c10d::allreduce_` | 60.179 ms | 168 |

### Device 侧结论

1. **Rank0 的通信和同步相关耗时明显更高。**

   Rank0 与 Rank1 的关键差异如下：

   | 项目 | Rank0 | Rank1 | Rank0 / Rank1 |
   | --- | ---: | ---: | ---: |
   | `wait_event` | 453.790 ms | 237.649 ms | 1.91x |
   | `HcclAllreduce` | 380.305 ms | 104.458 ms | 3.64x |
   | `vllm::all_reduce` | 394.001 ms | 69.202 ms | 5.69x |
   | `c10d::allreduce_` | 233.346 ms | 60.179 ms | 3.88x |
   | `Event::wait` | 163.126 ms | 137.869 ms | 1.18x |

2. **计算类 AICore 耗时基本一致。**

   `Device Total Duration With AICore` 总和为：

   ```text
   Rank0: 62.174 ms
   Rank1: 61.671 ms
   ```

   `aten::linear`、`aten::matmul`、`aclnnMatmul` 的 AICore 耗时也几乎一致，约为 18.7 ms。因此当前 Rank 差异主要不是矩阵计算本身，而是通信、事件等待或设备任务依赖。

3. **`wait_event` 和 `Event::wait` 不能直接当作计算算子。**

   它们更适合被视为同步和依赖等待信号。需要在 `trace_view.json` 中观察等待发生在通信前、通信后还是计算前，才能判断具体瓶颈。

4. **`npu_fx_compiler inference` 需要排除预热或编译影响。**

   Rank0 的 Device Total 为 425.793 ms，Rank1 为 113.910 ms，差异较大。如果 profiling 覆盖了首次执行或编译阶段，应先重新进行预热，再在稳定业务请求期间采集，避免把编译开销当作推理热点。

## 6. 与 step_trace_time 的对应关系

`step_trace_time.csv` 显示：

```text
Rank0 Communication: 172.067 ms
Rank1 Communication: 31.228 ms
Rank0 Free: 38.300 ms
Rank1 Free: 179.154 ms
```

`operator_details.csv` 进一步说明了这种差异可能来自：

- Rank0 的 `vllm::all_reduce`、`HcclAllreduce` 和 `c10d::allreduce_` 设备耗时更高。
- Rank0 的 `wait_event` 耗时也更高。
- Rank1 的通信算子耗时较低，但在 Step 统计中出现了更多 Free 时间。

因此，当前性能问题的优先方向是通信和同步时序，而不是优先优化 `matmul` 或 `linear`。

## 7. 建议的后续查看顺序

1. 打开 `trace_view.json`，定位 Rank0 的 `HcclAllreduce`、`all_reduce` 和 `wait_event` 时间区间。
2. 对照 Rank1，确认其 `Free` 区间是否正好对应 Rank0 的通信或同步区间。
3. 查看 `communication.json` 和 `communication_matrix.json`，确认通信操作的大小、方向和 Rank 分布。
4. 查看 `task_time.csv`，确认是否存在任务排队或设备侧依赖。
5. 排除编译和预热影响后，再重新比较 `npu_fx_compiler inference`。
6. 最后再使用 `kernel_details.csv` 深入检查具体底层 Kernel。

## 8. 分析限制

- `operator_details.csv` 是算子统计表，不提供完整的时间先后关系；时序问题必须结合 `trace_view.json`。
- 父子算子的 Total Duration 可能重复计算，不能简单累加所有行。
- 当前文件包含编译、同步、通信和推理活动，不能直接把所有耗时都归因于模型计算。
- 当前 Rank0 和 Rank1 的结果来自同一采集窗口，但仍需结合请求阶段和预热状态确认数据是否代表稳定推理阶段。
