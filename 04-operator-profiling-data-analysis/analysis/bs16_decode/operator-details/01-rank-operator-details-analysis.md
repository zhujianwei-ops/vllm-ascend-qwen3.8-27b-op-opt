# BS16 Decode Operator Details 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 7；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922014740695_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922014740695_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_decode/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

这是含长输入 prefill、混合调度与短 M 执行的完整请求窗口；不能作为纯 decode 或固定执行 batch=16 的统计。

## 框架调用路径

`Device Total`、`Host Total` 包含子调用；同一 kernel 可能向多个父层累计。`wait_event` 又可能关联多个并行 stream，因此下面各行不能相加或与 Stage 直接比较为“占比”。

| Name | rank0/1 次数 | rank0 Device Total (ms) | rank1 Device Total (ms) | rank0 Host Total (ms) | rank0 Host Self (ms) |
| --- | --- | --- | --- | --- | --- |
| `wait_event` | 3462/3462 | 34,817.155 | 34,827.052 | 8.392 | 8.392 |
| `npu_fx_compiler inference` | 62/62 | 32,473.893 | 32,453.495 | 3,247.344 | 503.868 |
| `Event::wait` | 68/68 | 18,491.007 | 18,507.578 | 1.948 | 1.678 |
| `vllm::all_reduce` | 1578/1578 | 18,168.666 | 18,136.683 | 140.130 | 61.605 |
| `c10d::allreduce_` | 1578/1578 | 16,490.388 | 16,470.788 | 72.395 | 49.552 |
| `npu::npu_quant_matmul` | 2492/2492 | 5,511.989 | 5,512.488 | 221.015 | 193.802 |
| `aclnnQuantMatmulWeightNz` | 2492/2492 | 5,511.989 | 5,512.488 | 8.399 | 8.399 |
| `vllm::qwen_gdn_attention_core` | 528/528 | 5,178.240 | 5,181.912 | 1,697.986 | 367.257 |
| `HcclAllreduce` | 3156/3156 | 4,495.769 | 4,446.787 | 6.683 | 4.546 |
| `ChunkGatedDeltaRuleFunction` | 528/528 | 3,339.617 | 3,344.950 | 847.033 | 540.067 |
| `aten::linear` | 1175/1175 | 878.431 | 875.621 | 138.526 | 3.215 |
| `aten::matmul` | 1175/1175 | 878.431 | 875.621 | 126.962 | 115.909 |
| `aclnnMatmul` | 1175/1175 | 878.431 | 875.621 | 4.995 | 4.995 |

## 解释

`vllm::all_reduce -> c10d::allreduce_ -> HcclAllreduce` 是嵌套调用路径，不能把三层时间加起来。以通信阶段分析量化整网预算，以这些父层定位代码入口。

`npu_fx_compiler inference` 是图推理区域名称，仅有这个名称并不能证明本次发生了编译；需要实际编译事件、日志或首次 shape 执行证据。

rank0 `npu::npu_quant_matmul` 有 2,492 条 Host 记录，底层 `QuantBatchMatmulV3` 有 3,740 次。图回放和主机算子记录口径可能不同，算子频率以设备 CSV 为准。Host Total 也不是全部暴露在设备关键路径上的 CPU 时间。

下一步在图回放边界与真实长等待之间建立时间关联，再决定处理 Python 调度、图切分还是跨 stream 依赖。
