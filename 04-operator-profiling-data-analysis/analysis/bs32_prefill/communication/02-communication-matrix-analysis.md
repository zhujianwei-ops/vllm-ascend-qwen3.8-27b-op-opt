# BS32 Prefill communication_matrix.json 分析

AllReduce 汇总传输数据如下：

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| LOCAL 传输量 | 713,034.854 MB | 713,034.854 MB |
| LOCAL 传输时间 | 2,498.132 ms | 2,489.206 ms |
| LOCAL 带宽 | 285.427 GB/s | 286.451 GB/s |
| SIO 传输量 | 356,538.388 MB | 356,538.388 MB |
| SIO 传输时间 | 2,343.214 ms | 2,333.009 ms |
| SIO 带宽 | 152.158 GB/s | 152.823 GB/s |

AllGather 的 SIO 汇总为 3,987.671 MB、25.474/25.523 ms、156.541/156.239 GB/s。两卡传输量和带宽对称，没有发现链路侧的单卡失衡。小消息 AllReduce 的带宽仅约 5--7 GB/s，适合通过 bucket 合并降低启动成本。
