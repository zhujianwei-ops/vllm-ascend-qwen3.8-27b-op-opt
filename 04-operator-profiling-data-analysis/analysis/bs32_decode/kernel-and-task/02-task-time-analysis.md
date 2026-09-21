# BS32 Decode Task Time 分析

| Task 类型 | rank0 调用次数 | rank0 总耗时 | rank0 平均 | rank0 最大 | rank1 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `EVENT_WAIT` | 9,729 | 69,069.450 ms | 7.099 ms | 800.003 ms | 69,130.102 ms |
| `NOTIFY_WAIT_SQE` | 72,495 | 59,761.696 ms | 0.824 ms | 19.136 ms | 59,968.166 ms |
| `MIX_AIC` | 13,413 | 18,174.193 ms | 1.355 ms | 5.862 ms | 18,183.384 ms |
| `AI_VECTOR_CORE` | 98,991 | 9,430.581 ms | 0.095 ms | 2.033 ms | 9,457.554 ms |
| `NOTIFY_WAIT` | 11,773 | 4,494.296 ms | 0.382 ms | 35.335 ms | 4,483.721 ms |
| `AI_CPU` | 4,019 | 1,100.717 ms | 0.274 ms | 7.019 ms | 1,110.922 ms |

等待类 Task 的累计值显著，且 `EVENT_WAIT` 最大单次约 800 ms。与 Step 的近零 overlap 一致，说明需要分析事件依赖和通信发起顺序。
