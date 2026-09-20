# step_trace_time.csv 分析

## 分析对象

```text
04-operator-profiling-data-analysis/vllm_profile/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
04-operator-profiling-data-analysis/vllm_profile/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
```

两份 CSV 各只有一条聚合记录，`Step` 字段为空。因此本文件反映一次采集窗口的阶段统计，不能判断多个 Step 之间的稳定性。

## Rank 对比

| 指标 | Rank0 / Device14 | Rank1 / Device15 |
| --- | ---: | ---: |
| Computing | 130.289 ms | 130.297 ms |
| Communication(Not Overlapped) | 172.051 ms | 31.228 ms |
| Overlapped | 0.016 ms | 0 ms |
| Communication | 172.067 ms | 31.228 ms |
| Free | 38.300 ms | 179.154 ms |
| Stage | 340.640 ms | 340.680 ms |
| Bubble | 0 ms | 0 ms |
| Preparing | 1.234 ms | 1.249 ms |

## 阶段占比

以 `Stage` 为分母：

| 指标 | Rank0 | Rank1 |
| --- | ---: | ---: |
| Computing / Stage | 38.25% | 38.25% |
| Communication / Stage | 50.51% | 9.17% |
| Free / Stage | 11.24% | 52.59% |
| Overlapped / Stage | 0.005% | 0% |

Rank0 一半以上的阶段时间用于通信；Rank1 的通信占比不足 10%，但 Free 占比超过一半。

## 主要结论

### 1. 计算耗时基本一致

```text
Rank0: 130.289 ms
Rank1: 130.297 ms
```

两个 Rank 的计算阶段差异约 `0.008 ms`，说明当前主要差异不在计算负载。

### 2. 通信耗时不一致

```text
Rank0: 172.067 ms
Rank1: 31.228 ms
```

Rank0 的通信时间明显更高，说明它是通信和同步侧的主要负载方。

### 3. Rank1 有明显 Free 时间

```text
Rank0: 38.300 ms
Rank1: 179.154 ms
```

Rank1 多出约 `140.855 ms` 的 Free 时间，可能是在等待通信、同步、Host 调度或其他设备依赖。仅凭该 CSV 不能确定具体原因。

### 4. 通信与计算几乎没有重叠

Rank0 的 `Overlapped` 只有 `0.016 ms`，Rank1 为 `0 ms`。这说明采集窗口内通信和计算没有形成有效重叠，或重叠比例非常低。

### 5. 总阶段耗时基本同步

```text
Rank0 Stage: 340.640 ms
Rank1 Stage: 340.680 ms
差异: 0.040 ms，约 0.0117%
```

整体 Stage 基本同步完成，但内部时间分布明显不同：Rank0 更偏通信，Rank1 更偏等待或空闲。

## 对第六步问题的回答

- **通信耗时是否占比较高？** Rank0 是，通信占 Stage 的 50.51%；Rank1 的通信占比为 9.17%。
- **各 Rank 通信耗时是否一致？** 不一致，Rank0 明显高于 Rank1。
- **通信与计算是否无法重叠？** 有明显迹象，两个 Rank 的 Overlapped 几乎为零。
- **是否有单个 Rank 成为整体瓶颈？** Rank0 是通信侧主要负载方，Rank1 体现为较长 Free；但最终 Stage 时间基本一致。

## 分析限制

- 只有一条聚合记录，不能判断 Step 稳定性。
- `communication` 字段与 `communication.json` 的统计口径不同，不能直接相加。
- 具体通信阻塞计算的时序，需要第五步的 `trace_view.json` 分析确认。
