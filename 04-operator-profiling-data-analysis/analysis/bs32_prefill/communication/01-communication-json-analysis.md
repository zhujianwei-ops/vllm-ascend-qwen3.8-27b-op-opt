# BS32 Prefill communication.json 分析

每个 Rank 记录 2,611 个 collective。`communication.json` 的 Elapse/Wait 统计与 Step 的 Communication 不是同一口径。

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| Collective Elapse | 6,482.380 ms | 6,431.729 ms |
| Transit | 0 ms | 0 ms |
| Wait | 6,481.444 ms | 6,430.753 ms |
| Synchronization | 6,481.444 ms | 6,430.753 ms |
| Idle | 0.937 ms | 0.976 ms |

该文件中 Wait 几乎等于 Elapse，表明其通信时间记录主要反映调度、同步或依赖等待。物理传输带宽应以 `communication_matrix.json` 为准。
