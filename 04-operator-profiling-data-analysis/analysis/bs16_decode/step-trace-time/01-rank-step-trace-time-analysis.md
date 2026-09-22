# BS16 Decode Step Trace Time 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 7；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922014740695_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922014740695_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_decode/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

这是含长输入 prefill、混合调度与短 M 执行的完整请求窗口；不能作为纯 decode 或固定执行 batch=16 的统计。

## 时间分解

| 字段 | rank0 (ms) | rank1 (ms) | rank0 / Stage |
| --- | --- | --- | --- |
| Stage | 16,563.929 | 16,563.903 | 100.0000% |
| Computing | 14,733.274 | 14,747.237 | 88.9479% |
| Communication(Not Overlapped) | 1,709.492 | 1,695.778 | 10.3206% |
| Communication | 1,709.640 | 1,695.856 | 10.3215% |
| Overlapped | 0.148 | 0.078 | 0.0009% |
| Free | 121.164 | 120.887 | 0.7315% |
| Preparing | 2.158 | 2.263 | 0.0130% |

原始 CSV 单位为 us，本报告换算为 ms。两卡 Stage 差为 0.027 ms；整体时间接近不能证明每层到达时间或每次通信均衡。

本次 `Bubble=0`，校验关系为 `Stage ≈ Computing + Communication(Not Overlapped) + Free`，以及 `Communication = Communication(Not Overlapped) + Overlapped`。`Preparing` 是单列观测，不再次加到 Stage；重叠时间也不能重复相加。

## 时间窗口限制

两卡只有一条聚合记录，`Step` 列为空；解析日志提示 step node list 为空。这表示没有逐 step 标记，并非文件缺失，也不能把 Stage 除以输出数当作 TPOT。

rank0 通信覆盖的重叠比例为 0.0087%。这是 CANN 阶段分类的结果；其他 stream 上 AICPU 的任务包络可能与计算同时存在，不应据此断言所有硬件任务绝无并行。

Free 为 121.164 ms。只有结合时间线上的 CPU 提交、采样、请求到达和图执行边界才能归因，不能直接把 Free 全部算作可消除的 CPU 开销。

## 优化预算

若非重叠通信减少 20%–40%，且节省全部落在关键路径、其他执行不变，rank0 可节省 341.898–683.797 ms，Stage 降幅为 2.06%–4.13%。这是条件预算，需 A/B 实测；不能用任务累计等待值替代这个分母。
