# communication_matrix.json 分析

## 分析对象

```text
04-operator-profiling-data-analysis/vllm_profile/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/communication_matrix.json
04-operator-profiling-data-analysis/vllm_profile/rank1_17410_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/communication_matrix.json
```

`communication_matrix.json` 按通信方向和链路记录本地传输、跨卡传输、数据量、传输耗时、带宽及对应的 HCCL 操作名。

## 链路类型

当前主要包含：

- `LOCAL`：本地路径传输。
- `SIO`：Rank0 与 Rank1 之间的跨卡传输。

没有发现 RDMA、HCCS 或 PCIe 作为主要传输类型，跨卡通信主要通过 `SIO` 完成。

## AllReduce 总体通信量

| 指标 | Rank0 视角 | Rank1 视角 |
| --- | ---: | ---: |
| `LOCAL` 传输量 | 44.733 MB | 44.733 MB |
| `SIO` 传输量 | 22.364 MB | 22.364 MB |
| `LOCAL` 传输时间 | 2.125 ms | 2.161 ms |
| `SIO` 传输时间 | 0.903 ms | 0.914 ms |
| `LOCAL` 带宽 | 21.051 GB/s | 20.698 GB/s |
| `SIO` 带宽 | 24.759 GB/s | 24.459 GB/s |

两张卡的总数据量一致，实际传输带宽也接近，未发现明显的物理链路吞吐失衡。

## AllGather 总体通信量

| 指标 | Rank0 视角 | Rank1 视角 |
| --- | ---: | ---: |
| `LOCAL` 传输量 | 12.908 MB | 12.908 MB |
| `SIO` 传输量 | 6.454 MB | 6.454 MB |
| `LOCAL` 传输时间 | 0.099 ms | 0.107 ms |
| `SIO` 传输时间 | 0.054 ms | 0.055 ms |
| `LOCAL` 带宽 | 130.557 GB/s | 120.736 GB/s |
| `SIO` 带宽 | 118.845 GB/s | 117.846 GB/s |

AllGather 的数据量和带宽也基本对称，且规模小于 AllReduce。

## 异常分布

在 `allreduce-middle` 中，Rank1 出现较大的本地传输块：

```text
Rank1: LOCAL 10.487 MB，0.500 ms，20.972 GB/s
Rank0: LOCAL 0.164 MB，约 0.011 ms
```

这说明某些通信聚合阶段在不同 Rank 上承担的本地通信工作不同。但 `allreduce-middle` 是聚合项，不能单独据此断定 Rank1 整体负载更重，需要结合具体 `Op Name` 和时间线定位。

## 结论和限制

- 两个 Rank 的通信数据量总体对称。
- `SIO` 是当前主要跨卡链路。
- 实际链路带宽接近，当前没有直接证据表明物理带宽不足是主要瓶颈。
- 主要问题更可能在通信启动、完成同步和计算重叠，时序需要由 `trace_view.json` 验证。
