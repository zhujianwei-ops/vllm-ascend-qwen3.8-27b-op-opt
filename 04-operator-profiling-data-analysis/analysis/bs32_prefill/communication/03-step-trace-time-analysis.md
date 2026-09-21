# BS32 Prefill 通信与 Stage 关联分析

`step_trace_time.csv` 显示 rank0/rank1 的非重叠通信为 3,241.190/3,215.864 ms，而 `communication.json` 的 Wait 为 6,481.444/6,430.753 ms。两者统计口径不同，不能相加。

可以确认的事实是：两个 Rank 的通信量和链路带宽对称，Stage 中没有通信计算重叠，且任务层存在大量事件等待。因此优先优化通信发起时机、bucket 粒度和 stream 依赖；在 trace 中验证后再调整 HCCL 参数。
