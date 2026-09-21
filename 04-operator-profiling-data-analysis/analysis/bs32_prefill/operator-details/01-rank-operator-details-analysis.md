# BS32 Prefill Operator Details 分析

`operator_details.csv` 包含父子调用关系，`Device Total Duration` 会重复包含子任务，不能与其他行或 Stage 直接相加。

## Device Total 热点

| 算子 | rank0 Device Total | rank1 Device Total | 调用次数 |
| --- | ---: | ---: | ---: |
| `npu_fx_compiler inference` | 70,687.420 ms | 70,648.870 ms | 72 |
| `wait_event` | 66,961.760 ms | 66,899.020 ms | 5,292 |
| `vllm::all_reduce` | 45,800.570 ms | 45,734.710 ms | 2,484 |
| `c10d::allreduce_` | 42,587.260 ms | 42,547.190 ms | 2,484 |
| `Event::wait` | 38,328.990 ms | 38,276.960 ms | 72 |
| `HcclAllreduce` | 35,039.630 ms | 34,929.410 ms | 4,968 |
| `npu::npu_quant_matmul` | 10,056.275 ms | 10,053.264 ms | 3,960 |
| `vllm::qwen_gdn_attention_core` | 8,202.056 ms | 8,215.130 ms | 864 |

## Host 热点

`npu_fx_compiler inference` 的 Host Total 为 3,789.540/3,824.872 ms，`vllm::qwen_gdn_attention_core` 为 2,054.914/2,080.247 ms，`aten::copy_` 为 1,434.539/1,423.210 ms。

## 结论

`wait_event`、`Event::wait` 和 AllReduce 形成了最主要的设备侧等待链路。两卡分布接近，问题是共同关键路径而不是 Rank 负载不均。应结合双 Rank trace 检查等待是否围绕 HCCL 发起和完成事件集中出现。
