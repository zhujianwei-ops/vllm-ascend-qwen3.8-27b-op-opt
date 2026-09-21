# BS32 Decode 通信与 Stage 关联分析

Stage 中 rank0/rank1 的通信为 3,401.538/3,371.148 ms，通信计算重叠仅 0.266/0.136 ms。`communication.json` 的 Collective Elapse 为 10,945.798/10,898.068 ms，主要由 Wait 组成，两者口径不同，不能直接加总。

通信矩阵显示两个 Rank 的传输量和带宽对称；Task 层显示等待类事件很多。因此 decode 优化应将 AllReduce 作为异步工作排入计算间隙，并先用双 Rank trace 验证具体的 `EVENT_WAIT` 与 HCCL 调用关系。
