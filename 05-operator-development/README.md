# 算子开发与优化

当前任务：优化 Qwen3.8-27B W8A8 FFN 的 `QuantBatchMatmulV3`，优先覆盖 `M=15900/15912, K=5120, N=17408`，通过验证后扩展到 BS32 中的 `M=15920`。

设计日期：2026-09-22。**当前状态为设计完成，尚未实现、编译或实测候选算子。** 下列收益均为历史数据推算的目标，不是优化结果。

| 阅读顺序 | 文档 | 用途 |
| --- | --- | --- |
| 1 | [开发设计](01-quantbatchmatmulv3-optimization-design.md) | 算子语义、证据、三条实现路线、集成与回退 |
| 2 | [实验及验收方案](02-quantbatchmatmulv3-validation-plan.md) | Shape 集合、计时、精度、图模式、整网 A/B |
| 3 | [Agent 执行交接](03-quantbatchmatmulv3-agent-handoff.md) | 按阶段执行的任务、交付文件、进入下一阶段的条件 |

第一步是建立单算子基准，复现同 K/N 下的 M 效率下降。随后比较补齐 M、按 M 拆分的完整调用收益；必要时再实现独立 AscendC kernel。达到验收条件的最简单方案即可作为交付，不要求三条路线都开发到底。

业务约束保持 TP=2、W8A8、MTP=3。该任务优先改善长输入和混合执行中的 FFN，不把预计收益等同于纯 decode 的 TPOT/ITL 收益。

分析依据：

- [BS16 Decode 分析](../04-operator-profiling-data-analysis/analysis/bs16_decode/05-deep-performance-analysis.md)
- [BS16/BS32 对比及统计口径](../04-operator-profiling-data-analysis/analysis/bs16-comparison.md)
- [逐 Shape、逐 Rank 数值](../04-operator-profiling-data-analysis/analysis/bs16_decode/kernel-and-task/shape-metrics.csv)

后续实现代码放入本目录的 `quantbatchmatmulv3/`；vLLM-Ascend 集成补丁放入 `06-operator-integration/`；整网验收放入 `07-final-performance/`。这些实现目录及文件目前只是约定，尚未创建。
