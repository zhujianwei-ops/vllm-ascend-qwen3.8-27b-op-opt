# BS32 Prefill Task Time 分析

| Task 类型 | rank0 调用次数 | rank0 总耗时 | rank0 平均 | rank0 最大 | rank1 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `EVENT_WAIT` | 5,292 | 66,961.760 ms | 12.653 ms | 874.689 ms | 66,899.020 ms |
| `NOTIFY_WAIT_SQE` | 48,330 | 52,318.758 ms | 1.083 ms | 17.107 ms | 52,311.914 ms |
| `NOTIFY_WAIT` | 7,830 | 18,210.874 ms | 2.326 ms | 19.805 ms | 18,150.985 ms |
| `MIX_AIC` | 9,486 | 16,859.168 ms | 1.777 ms | 4.833 ms | 16,865.716 ms |
| `AI_CPU` | 2,610 | 15,437.058 ms | 5.915 ms | 18.561 ms | 15,407.712 ms |

等待类 Task 时间不能相加为墙钟耗时，但它们大于计算 Task 的累计值且有 874 ms 单次长尾。结合 Step 的 0 ms overlap，需在 trace 中定位 `EVENT_WAIT` 的依赖源并减少跨 stream 同步。
