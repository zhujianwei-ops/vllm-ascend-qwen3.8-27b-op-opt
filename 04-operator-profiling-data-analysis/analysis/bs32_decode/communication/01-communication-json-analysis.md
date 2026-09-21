# BS32 Decode communication.json 分析

每个 Rank 记录 4,234 个 collective。

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| Collective Elapse | 10,945.798 ms | 10,898.068 ms |
| Transit | 301.661 ms | 299.218 ms |
| Wait | 10,547.269 ms | 10,502.189 ms |
| Synchronization | 10,545.696 ms | 10,498.385 ms |
| Idle | 96.868 ms | 96.661 ms |

Wait 占 Elapse 约 96%，而两卡数据接近。通信链路的主要问题是同步和依赖等待，非单卡通信不平衡。
