# communication.json 分析

## 分析对象

```text
04-operator-profiling-data-analysis/vllm_profile/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/communication.json
04-operator-profiling-data-analysis/vllm_profile/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/communication.json
```

`communication.json` 记录通信操作的开始时间、总耗时、实际传输耗时、等待时间、同步时间、空闲时间、数据量和带宽。

## 统计结果

每个 Rank 都包含 609 个 collective 通信操作，主要是 AllReduce 和 AllGather。

| 指标 | Rank0 | Rank1 |
| --- | ---: | ---: |
| AllReduce Elapse | 168.712 ms | 30.492 ms |
| AllReduce Wait | 167.823 ms | 22.935 ms |
| AllGather Elapse | 24.210 ms | 17.164 ms |
| AllGather Wait | 24.032 ms | 16.800 ms |
| Total Op Elapse | 192.923 ms | 47.656 ms |
| Total Op Transit | 0.054 ms | 0.055 ms |
| Total Op Wait | 191.855 ms | 39.735 ms |
| Total Op Synchronization | 191.812 ms | 39.690 ms |
| Total Op Idle | 1.013 ms | 7.866 ms |

## 主要结论

### 1. Rank0 的通信等待明显更重

Rank0 的通信总耗时约为 Rank1 的 `4.05` 倍。AllReduce 差异最明显：

```text
Rank0: 168.712 ms
Rank1: 30.492 ms
```

Rank0 的 AllReduce 总耗时约为 Rank1 的 `5.53` 倍，Wait 总和约为 `7.32` 倍。通信差异主要集中在 AllReduce，而不是 AllGather。

### 2. 等待和同步占主导

两卡的实际 Transit 都只有约 `0.055 ms`，但 Wait 达到：

```text
Rank0: 191.855 ms
Rank1: 39.735 ms
```

当前主要开销更像是通信完成前后的等待、同步或依赖，而不是物理链路搬运数据本身。具体等待发生的位置需要结合 `trace_view.json` 确认。

### 3. 与 Step 统计的口径不同

`communication.json` 的 Total Op 与 `step_trace_time.csv` 的 Communication 不同：

| 文件 | Rank0 | Rank1 |
| --- | ---: | ---: |
| `communication.json` Total Op Elapse | 192.923 ms | 47.656 ms |
| `step_trace_time.csv` Communication | 172.067 ms | 31.228 ms |

两者统计范围不同，不能直接相加；但同口径对比都表明 Rank0 的通信和同步等待明显更高。

## 分析限制

- 该文件是通信操作汇总，不能单独确认通信阻塞计算的时序因果。
- 统计值不能与其他文件直接相加。
- 需要使用 `trace_view.json` 对齐通信操作的开始和结束时间。
