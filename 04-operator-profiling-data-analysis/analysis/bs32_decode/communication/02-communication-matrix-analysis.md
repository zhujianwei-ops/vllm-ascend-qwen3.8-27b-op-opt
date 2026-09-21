# BS32 Decode communication_matrix.json 分析

AllReduce 汇总传输数据如下：

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| LOCAL 传输量 | 732,147.783 MB | 732,147.783 MB |
| LOCAL 传输时间 | 2,599.500 ms | 2,575.450 ms |
| LOCAL 带宽 | 281.650 GB/s | 284.280 GB/s |
| SIO 传输量 | 366,073.876 MB | 366,073.876 MB |
| SIO 传输时间 | 2,455.033 ms | 2,434.627 ms |
| SIO 带宽 | 149.112 GB/s | 150.361 GB/s |

AllGather 的 SIO 汇总为 4,420.564 MB、28.960/28.965 ms、152.641/152.619 GB/s。两卡带宽对称。小消息 AllReduce 仍只有约 5--7 GB/s，适合聚合；大消息的 SIO 带宽约 149--150 GB/s，当前首先应减少等待和提升 overlap。
