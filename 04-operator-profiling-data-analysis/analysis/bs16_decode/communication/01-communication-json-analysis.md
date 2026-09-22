# BS16 Decode communication.json 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 7；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922014740695_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922014740695_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_decode/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

这是含长输入 prefill、混合调度与短 M 执行的完整请求窗口；不能作为纯 decode 或固定执行 batch=16 的统计。

## 去重规则

仅统计真实操作明细，排除 `Total Op Info`；并与该汇总行逐字段核对。Synchronization 是 Wait 的相关子项，不再次相加。原文件自带的 Wait Time Ratio 分母是 `Wait+Transit`；若另算 `Wait/Elapse`，必须明确区别。

| 指标 | rank0 | rank1 |
| --- | --- | --- |
| Collective 次数 | 2629 | 2629 |
| Elapse Time(ms) | 2,451.225 | 2,437.833 |
| Transit Time(ms) | 52.069 | 51.669 |
| Wait Time(ms) | 2,382.307 | 2,369.578 |
| Synchronization Time(ms) | 2,381.955 | 2,368.859 |
| Idle Time(ms) | 16.849 | 16.586 |

## 按操作类型

| 类型 | 次数/卡 | rank0 Elapse (ms) | rank1 Elapse (ms) | rank0 Wait (ms) | rank0 Transit (ms) |
| --- | --- | --- | --- | --- | --- |
| allReduce | 2393 | 2,212.481 | 2,199.156 | 2,161.777 | 37.582 |
| allGather | 236 | 238.744 | 238.677 | 220.530 | 14.487 |

rank0 `Wait/Elapse` 为 97.19%。但这里的 Transit 不能代表所有设备间搬运：同文件的 SIO/SDMA 带宽条目与 communication_matrix 仍记录了真实传输。不能把 Wait 全部认作“空等且可以删除”，也不能根据 Transit 很小推断链路传输几乎免费。

旧 BS32 文档的 Elapse/Wait 将汇总行再次相加，因而放大约 2 倍，操作数也多算 1 个。复核后的 BS32 prefill 为 2,610 次、rank0 Elapse 3,241.190 ms；BS32 decode 为 4,233 次、rank0 Elapse 5,472.899 ms。此次 BS16 和比较文档均使用去重值，历史文档保持原样。
