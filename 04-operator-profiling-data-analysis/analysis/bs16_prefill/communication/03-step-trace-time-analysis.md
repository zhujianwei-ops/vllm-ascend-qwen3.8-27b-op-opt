# BS16 Prefill 通信与 Stage 关联分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 1；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922015612919_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922015612919_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_prefill/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

本窗口实际主干 token 行数为 32,768，未证明覆盖了无缓存的 16×8192 token 计算；见主报告中的缓存限制。

## 三种时间口径

| 视角 | rank0 (ms) | rank1 (ms) | 含义 |
| --- | --- | --- | --- |
| Step Communication | 423.136 | 416.111 | CANN 阶段分类 |
| Step 非重叠通信 | 423.136 | 416.111 | Stage 分解中的通信部分 |
| 去重 Collective Elapse 累加 | 423.136 | 416.111 | 逐操作记录，不保证互斥 |
| AllReduce AI_CPU kernel 累加 | 1,716.705 | 1,710.757 | 任务包络与等待，不是纯传输 |
| AllReduce SIO transfer 累加 | 300.308 | 299.372 | 链路工作量 |

这些行不相加，也不能互相相减得出“纯同步开销”。Prefill 的去重 Elapse 与 Step Communication 近似相等；decode 的两者仍不同，应保留口径并通过具体事件关联定位，不能继续用重复汇总解释全部差异。

双卡总 Stage 接近仅说明窗口完成时间接近；应按相同通信序号对齐到达和完成时刻，检查 producer 完成到通信发起、通信完成到 consumer 启动之间的真实空隙。CANN 报告的 overlap 很低可作为实验线索，但尚不足以确定哪个依赖可被隐藏。

通信优化保留 TP=2、W8A8 和 MTP，优先验证现有通信融合/图配置，再考虑自定义分块流水。整网预算统一以非重叠通信为分母，避免把 AICPU 和 Wait 累计一起算作收益。
