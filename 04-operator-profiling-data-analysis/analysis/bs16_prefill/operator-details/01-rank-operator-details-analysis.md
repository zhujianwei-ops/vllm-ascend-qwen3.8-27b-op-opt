# BS16 Prefill Operator Details 分析

分析日期：2026-09-22。脚本条件：16 并发、16 个请求、输入 8192、输出 1；TP=2、W8A8、MTP=3。

- rank0 / Device14：`rank0_10893_20260922015612919_ascend_pt`。
- rank1 / Device15：`rank1_10912_20260922015612919_ascend_pt`。
- 原始文件位于 `vllm_profile/bs16_prefill/<rank目录>/ASCEND_PROFILER_OUTPUT/`。
- 数值与 SHA256 见 [计算底稿](../../bs16-profile-metrics.json)，综合结论见 [主报告](../05-deep-performance-analysis.md)。

本窗口实际主干 token 行数为 32,768，未证明覆盖了无缓存的 16×8192 token 计算；见主报告中的缓存限制。

## 框架调用路径

`Device Total`、`Host Total` 包含子调用；同一 kernel 可能向多个父层累计。`wait_event` 又可能关联多个并行 stream，因此下面各行不能相加或与 Stage 直接比较为“占比”。

| Name | rank0/1 次数 | rank0 Device Total (ms) | rank1 Device Total (ms) | rank0 Host Total (ms) | rank0 Host Self (ms) |
| --- | --- | --- | --- | --- | --- |
| `npu_fx_compiler inference` | 16/16 | 9,240.297 | 9,223.967 | 832.728 | 165.734 |
| `wait_event` | 1176/1176 | 8,563.198 | 8,472.793 | 2.724 | 2.724 |
| `vllm::all_reduce` | 552/552 | 5,908.117 | 5,884.838 | 44.664 | 19.131 |
| `c10d::allreduce_` | 552/552 | 5,488.454 | 5,472.229 | 23.491 | 15.642 |
| `Event::wait` | 16/16 | 4,737.785 | 4,657.679 | 0.449 | 0.384 |
| `HcclAllreduce` | 1104/1104 | 4,270.959 | 4,245.016 | 2.159 | 1.467 |
| `npu::npu_quant_matmul` | 880/880 | 1,292.290 | 1,291.131 | 18.455 | 10.299 |
| `aclnnQuantMatmulWeightNz` | 880/880 | 1,292.290 | 1,291.131 | 1.967 | 1.967 |
| `vllm::qwen_gdn_attention_core` | 192/192 | 1,084.480 | 1,086.204 | 451.039 | 69.690 |
| `ChunkGatedDeltaRuleFunction` | 192/192 | 851.056 | 852.353 | 302.051 | 196.661 |
| `vllm::unified_attention_with_output` | 76/76 | 252.242 | 253.533 | 20.329 | 9.232 |
| `npu::npu_fused_infer_attention_score` | 76/76 | 243.427 | 244.604 | 6.011 | 4.841 |
| `aclnnFusedInferAttentionScoreV3` | 76/76 | 243.427 | 244.604 | 0.305 | 0.305 |

## 解释

`vllm::all_reduce -> c10d::allreduce_ -> HcclAllreduce` 是嵌套调用路径，不能把三层时间加起来。以通信阶段分析量化整网预算，以这些父层定位代码入口。

`npu_fx_compiler inference` 是图推理区域名称，仅有这个名称并不能证明本次发生了编译；需要实际编译事件、日志或首次 shape 执行证据。

rank0 `npu::npu_quant_matmul` 有 880 条 Host 记录，底层 `QuantBatchMatmulV3` 有 880 次。图回放和主机算子记录口径可能不同，算子频率以设备 CSV 为准。Host Total 也不是全部暴露在设备关键路径上的 CPU 时间。

下一步在图回放边界与真实长等待之间建立时间关联，再决定处理 Python 调度、图切分还是跨 stream 依赖。
