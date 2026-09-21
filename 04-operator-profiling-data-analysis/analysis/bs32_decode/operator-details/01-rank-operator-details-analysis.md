# BS32 Decode Operator Details 分析

`Device Total Duration` 包含父子算子，以下结果用于定位调用路径，不能作为端到端耗时相加。

| 算子 | rank0 Device Total | rank1 Device Total | 调用次数 |
| --- | ---: | ---: | ---: |
| `wait_event` | 68,741.530 ms | 68,803.310 ms | 6,146 |
| `npu_fx_compiler inference` | 64,885.520 ms | 64,968.210 ms | 101 |
| `Event::wait` | 36,262.800 ms | 36,253.080 ms | 108 |
| `vllm::all_reduce` | 36,057.320 ms | 36,108.780 ms | 2,830 |
| `c10d::allreduce_` | 32,704.500 ms | 32,785.040 ms | 2,830 |
| `npu::npu_quant_matmul` | 11,073.100 ms | 11,072.010 ms | 4,484 |
| `vllm::qwen_gdn_attention_core` | 10,472.980 ms | 10,489.590 ms | 960 |
| `HcclAllreduce` | 8,349.966 ms | 8,310.071 ms | 5,660 |

Host Total 侧，`npu_fx_compiler inference` 为 5,921.812/6,081.975 ms，`vllm::qwen_gdn_attention_core` 为 3,129.774/3,206.206 ms。应在预热充分后复采一次，以排除编译路径对窗口的影响。

两卡 Device Total 基本一致；等待、AllReduce 和量化 MatMul 是同一条关键路径上的主要对象。
