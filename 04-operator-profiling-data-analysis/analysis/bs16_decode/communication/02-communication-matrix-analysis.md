# BS16 Decode communication_matrix.json 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 7；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922014740695_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922014740695_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_decode/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

这是含长输入 prefill、混合调度与短 M 执行的完整请求窗口；不能作为纯 decode 或固定执行 batch=16 的统计。

## 链路汇总

只读取 `allreduce-total@...` 和 `allgather-total@...`，不将 top/middle/bottom 样例再次累加。LOCAL 为卡内路径，SIO 为本 rank 的设备间链路路径；不把两者相加当作业务有效通信量，也不将两张卡的相同链路视图简单叠加。

| 操作 | 路径 | 每 rank 传输量 (MB) | rank0 时间 (ms) | rank1 时间 (ms) | rank0 GB/s | rank1 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| allreduce | LOCAL | 366,156.327 | 1,300.144 | 1,288.074 | 281.627 | 284.266 |
| allreduce | SIO | 183,078.154 | 1,228.214 | 1,217.983 | 149.061 | 150.313 |
| allgather | LOCAL | 4,438.375 | 17.710 | 17.719 | 250.616 | 250.480 |
| allgather | SIO | 2,219.187 | 14.490 | 14.483 | 153.158 | 153.227 |

带宽按 `ΣTransit Size(MB)/ΣTransit Time(ms)` 计算，数值单位正好为十进制 GB/s，不是逐调用带宽的算术平均。SDMA 与 SIO 条目可能描述同一搬运，不重复计数。

## 判断与实验

SIO 大消息聚合带宽约 150 GB/s，两卡接近。README 的 D2D 784 GB/s 是双向产品规格，与此处逻辑 rank 的单路径有效载荷统计不等价，因此不报告 `150/784` 为链路利用率。

应先分消息大小、等待边界与层依赖评估：小消息尝试减少发起和同步开销；大消息检查流水与实际 transfer。连续网络层通常存在真实数据依赖，不能任意把它们的 AllReduce 合并为一个 bucket。
