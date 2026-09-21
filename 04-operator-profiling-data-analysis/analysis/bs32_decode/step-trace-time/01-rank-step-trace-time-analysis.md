# BS32 Decode Step Trace Time 分析

## 分析对象

本报告对应 32 个并发请求、8192 token 输入、`max_tokens=7` 的 decode 主导窗口。窗口仍包含请求 prefill，因此不是首 token 后开始的纯 decode 切片。

| 指标 | rank0 / Device14 | rank1 / Device15 |
| --- | ---: | ---: |
| Computing | 29,434.442 ms | 29,467.802 ms |
| Communication(Not Overlapped) | 3,401.272 ms | 3,371.012 ms |
| Overlapped | 0.266 ms | 0.136 ms |
| Communication | 3,401.538 ms | 3,371.148 ms |
| Free | 197.871 ms | 194.815 ms |
| Stage | 33,033.585 ms | 33,033.629 ms |
| Preparing | 2.208 ms | 2.197 ms |

## 结论

- 两卡 Stage 仅相差 0.044 ms，TP 负载均衡正常。
- 计算占 Stage 约 89.1%，通信占约 10.2%。
- 通信重叠占比不足 0.001%，约 3.4 s 通信直接暴露在关键路径。
- 相对 `max_tokens=1` 窗口，Stage 增加约 4.22 s，其中主要是额外 decode 计算；通信仍未被有效隐藏。
