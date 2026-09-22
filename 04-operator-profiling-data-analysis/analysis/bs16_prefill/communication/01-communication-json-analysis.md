# BS16 Prefill communication.json 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 1；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922015612919_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922015612919_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_prefill/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

本窗口实际主干 token 行数为 32,768，未证明覆盖了无缓存的 16×8192 token 计算；见主报告中的缓存限制。

## 去重规则

仅统计真实操作明细，排除 `Total Op Info`；并与该汇总行逐字段核对。Synchronization 是 Wait 的相关子项，不再次相加。原文件自带的 Wait Time Ratio 分母是 `Wait+Transit`；若另算 `Wait/Elapse`，必须明确区别。

| 指标 | rank0 | rank1 |
| --- | --- | --- |
| Collective 次数 | 580 | 580 |
| Elapse Time(ms) | 423.136 | 416.111 |
| Transit Time(ms) | 0.000 | 0.000 |
| Wait Time(ms) | 422.930 | 415.867 |
| Synchronization Time(ms) | 422.930 | 415.867 |
| Idle Time(ms) | 0.206 | 0.245 |

## 按操作类型

| 类型 | 次数/卡 | rank0 Elapse (ms) | rank1 Elapse (ms) | rank0 Wait (ms) | rank0 Transit (ms) |
| --- | --- | --- | --- | --- | --- |
| allReduce | 552 | 418.775 | 411.751 | 418.573 | 0.000 |
| allGather | 28 | 4.361 | 4.360 | 4.357 | 0.000 |

rank0 `Wait/Elapse` 为 99.95%。但这里的 Transit 不能代表所有设备间搬运：同文件的 SIO/SDMA 带宽条目与 communication_matrix 仍记录了真实传输。不能把 Wait 全部认作“空等且可以删除”，也不能根据 Transit 很小推断链路传输几乎免费。

旧 BS32 文档的 Elapse/Wait 将汇总行再次相加，因而放大约 2 倍，操作数也多算 1 个。复核后的 BS32 prefill 为 2,610 次、rank0 Elapse 3,241.190 ms；BS32 decode 为 4,233 次、rank0 Elapse 5,472.899 ms。此次 BS16 和比较文档均使用去重值，历史文档保持原样。
