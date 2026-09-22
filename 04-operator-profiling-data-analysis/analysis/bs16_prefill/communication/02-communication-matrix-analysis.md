# BS16 Prefill communication_matrix.json 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 1；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922015612919_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922015612919_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_prefill/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

本窗口实际主干 token 行数为 32,768，未证明覆盖了无缓存的 16×8192 token 计算；见主报告中的缓存限制。

## 链路汇总

只读取 `allreduce-total@...` 和 `allgather-total@...`，不将 top/middle/bottom 样例再次累加。LOCAL 为卡内路径，SIO 为本 rank 的设备间链路路径；不把两者相加当作业务有效通信量，也不将两张卡的相同链路视图简单叠加。

| 操作 | 路径 | 每 rank 传输量 (MB) | rank0 时间 (ms) | rank1 时间 (ms) | rank0 GB/s | rank1 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| allreduce | LOCAL | 91,268.715 | 319.210 | 318.093 | 285.921 | 286.925 |
| allreduce | SIO | 45,634.355 | 300.308 | 299.372 | 151.958 | 152.434 |
| allgather | LOCAL | 1,038.418 | 4.062 | 4.083 | 255.643 | 254.308 |
| allgather | SIO | 519.209 | 3.316 | 3.326 | 156.567 | 156.086 |

带宽按 `ΣTransit Size(MB)/ΣTransit Time(ms)` 计算，数值单位正好为十进制 GB/s，不是逐调用带宽的算术平均。SDMA 与 SIO 条目可能描述同一搬运，不重复计数。

## 判断与实验

SIO 大消息聚合带宽约 150 GB/s，两卡接近。README 的 D2D 784 GB/s 是双向产品规格，与此处逻辑 rank 的单路径有效载荷统计不等价，因此不报告 `150/784` 为链路利用率。

应先分消息大小、等待边界与层依赖评估：小消息尝试减少发起和同步开销；大消息检查流水与实际 transfer。连续网络层通常存在真实数据依赖，不能任意把它们的 AllReduce 合并为一个 bucket。
