# BS32 Prefill Step Trace Time 分析

## 分析对象

本报告对应 32 个并发请求、8192 token 输入、`max_tokens=1` 的 prefill 主导窗口。每个 Rank 只有一条聚合记录，单位为 us。

| 指标 | rank0 / Device14 | rank1 / Device15 |
| --- | ---: | ---: |
| Computing | 25,425.934 ms | 25,447.338 ms |
| Communication(Not Overlapped) | 3,241.190 ms | 3,215.864 ms |
| Overlapped | 0 ms | 0 ms |
| Communication | 3,241.190 ms | 3,215.864 ms |
| Free | 146.901 ms | 150.970 ms |
| Stage | 28,814.026 ms | 28,814.173 ms |
| Preparing | 2.199 ms | 2.299 ms |

## 结论

- 两个 Rank 的 Stage 相差 0.147 ms，TP 负载平衡正常。
- 计算占 Stage 约 88.3%，通信占约 11.2%。
- `Overlapped=0`，3.2 s 通信全部落在关键路径，通信编排存在优化空间。
- `Free` 仅约 0.5%，设备不是以空闲为主，优先处理计算和通信依赖。

`max_tokens=1` 仍会执行首 token 相关工作，因此这是 prefill 主导窗口，不是严格的纯 prefill 切片。
