# BS16 Profiling 总览与 BS32 对比

分析日期：2026-09-22。本文使用原始 CSV/JSON 重新计算 BS16、BS32 双 rank 数据，不直接复用历史报告中的汇总数。

## 1. 报告入口

- [BS16 Prefill 主报告](bs16_prefill/05-deep-performance-analysis.md)
- [BS16 Decode 主报告](bs16_decode/05-deep-performance-analysis.md)
- 每组保持原有目录结构：`operator-details/`、`kernel-and-task/`、`communication/`、`step-trace-time/`，各 8 份分项报告加 1 份主报告。
- [数值底稿与原文件 SHA256](bs16-profile-metrics.json)
- [BS16 Prefill 全部矩阵 shape](bs16_prefill/kernel-and-task/shape-metrics.csv)、[BS16 Decode 全部矩阵 shape](bs16_decode/kernel-and-task/shape-metrics.csv)

## 2. 首先限制比较范围

以下取 rank0；rank1 结果和差异见各组报告。`BS` 是请求并发上限，`M` 是 kernel 的 token 行数，两者不同。

| 指标 | BS16 prefill | BS32 prefill | BS16 decode | BS32 decode |
| --- | ---: | ---: | ---: | ---: |
| 脚本请求数 | 16 | 32 | 16 | 32 |
| 脚本输入 token/请求 | 8,192 | 8,192 | 8,192 | 8,192 |
| 脚本输出 token/请求 | 1 | 1 | 7 | 7 |
| 脚本声明总输入 token | 131,072 | 262,144 | 131,072 | 262,144 |
| 主干实际 token 行数/层 | **32,768** | **256,000** | **131,460** | **262,860** |
| 主干 QKV 执行分组数 | 4 | 18 | 17 | 27 |
| Stage (ms) | 4,048.502 | 28,814.026 | 16,563.930 | 33,033.585 |
| Computing (ms) | 3,410.628 | 25,425.934 | 14,733.274 | 29,434.442 |
| 非重叠通信 (ms) | 423.136 | 3,241.190 | 1,709.492 | 3,401.272 |
| Free (ms) | 214.738 | 146.901 | 121.164 | 197.871 |
| QuantBatchMatmul 次数 | 880 | 3,960 | 3,740 | 5,940 |
| QuantBatchMatmul 累计 (ms) | 1,292.290 | 10,056.275 | 5,573.330 | 11,144.613 |
| QuantBatchMatmul 有效 TOP/s | 606.877 | 609.275 | 564.531 | 564.507 |
| DynamicQuant 累计 (ms) | 127.547 | 984.813 | 513.092 | 1,022.432 |

实际 token 行数由每个主干 linear-attention 层的 `K=5120,N=8192` QKV 投影还原，模型有 48 个此类层。检查连续 48 次投影的 M 一致后，只将每组的 M 计一次。这不包含其他投影重复计算，但仍可能包含 padding、speculative 验证和缓存影响，不等于唯一请求 token 数。

可得出的结论：

1. BS16 prefill 的输入计算量约为 BS32 的 **12.8%**，不能把 Stage 从 28.8 s 到 4.05 s 的变化当作降低并发带来的性能收益。两者量化 MatMul 单位工作量吞吐接近。
2. BS16/BS32 decode 的实际计算量与 Stage 均接近 1:2；量化 MatMul 吞吐几乎相同。输入量减少解释了主要总时长变化，尚无独立证据说明单算子因为 BS16 而更高效。
3. BS16 decode 的量化大 M 部分耗时 5,508.920 ms，小 M 部分仅 64.410 ms；当前窗口不能用于宣布纯 decode 的首要瓶颈已经确定。
4. 同 seed、重复输入与开启 prefix caching 是 BS16 prefill 计算量偏低的强疑点，但尚缺缓存命中与请求日志，不能把 75% 差额当作已验证的命中率。

## 3. 本次修正的历史统计口径

### 通信汇总行不能再次累加

`communication.json` 的 `Total Op Info` 是所有真实通信条目的汇总。旧 BS32 报告将它与明细再次相加，使 Elapse 等字段放大约 2 倍，操作数也多 1。以下为重新核对后的值：

| 场景 | 真实 collective 次数/卡 | rank0 Elapse (ms) | rank1 Elapse (ms) |
| --- | ---: | ---: | ---: |
| BS16 prefill | 580 | 423.136 | 416.111 |
| BS32 prefill | 2,610 | 3,241.190 | 3,215.864 |
| BS16 decode | 2,629 | 2,451.225 | 2,437.833 |
| BS32 decode | 4,233 | 5,472.899 | 5,449.034 |

本次未改写历史 BS32 报告；进行比较时以这里及数值底稿为准。旧 communication_matrix 的 `*-total@...` 链路汇总口径可保留。

### 不从 Wait 或 Transit 单列推断全部通信成本

BS16 prefill 中 Elapse≈423.136 ms、Transit=0，但 SIO AllReduce 明确记录 300.308 ms 的传输工作量。该现象说明 time-info 与 bandwidth-info 的统计路径不同，不能把 Wait 解释为全部可移除的“无数据搬运同步”。

Stage、AICPU 包络、通信事件 Elapse、Task 等待和链路 transfer 不是同一口径，既不能加总，也不能简单相减得到纯等待。Prefill 的去重 Elapse 与 Stage Communication 接近；decode 的去重后结果仍不同，需要具体时间线映射。

### 其他容易误读的字段

- **输出 1 与阶段**：首 token 的 logits/采样通常来自 prefill；不能认为必定额外执行了一次独立 decode。输出 7 的窗口则仍包含全部长输入处理。
- **算子调用数**：相等只表示数量一致；本次额外核对了相同 stream 下相邻 kernel 及 shape。最终融合还需要检查 tensor 身份和消费者。
- **负载均衡**：两卡 Stage 一致不证明逐层无偏斜，TP 同步本身可能对齐结束时间。
- **图推理名称**：`npu_fx_compiler inference` 不等于本轮发生了编译；Host/Device Total 的父子嵌套不能当作新增耗时。
- **cube_utilization**：本机 CANN 用核活跃周期覆盖计算此列，不能当作 FLOP/峰值或证明“已无优化空间”。
- **MBU**：当前没有 HBM 流量计数；报告给出的是 shape 字节模型，明确包含假设。

## 4. MFU/MBU 的计算边界

沿用 README 中的条件峰值：整机 FP16 4.48 PFLOPS、INT8 8.96 POPS 除以 8，得到 560 TFLOP/s 与 1.12 POP/s，带宽暂按 3.2 TB/s。采集的 `device_14/info.json.14`、`device_15/info.json.15` 只确认每 Device 的 24 AI Core、48 Vector Core、1800 MHz 频率字段，未确认板卡/芯片/逻辑 Device 的完整映射，也未确认 3.2 TB/s 的适用范围。

因此给出不依赖峰值的有效 TOP/s，同时把利用率标为条件估算：

```text
F_i = 2 * M_i * K_i * N_i
U_compute = sum(F_i) / (P_peak * sum(t_i))
B_i = 输入、NZ权重、输出、scale 各访问一次的字节数
MBU_model = sum(B_i) / (BW_peak * sum(t_i))
T_lower_model = sum(max(F_i / P_peak, B_i / BW_peak))
```

`U_compute` 是算子口径估计，不是模型级 MFU。按当前 profile 计算整模型 MFU，还缺少有效请求 token、prefill/decode 边界、缓存复用、MTP 接受率及完整有效运算量。Byte 模型也未考虑 L2 命中、重复搬运、workspace 和融合中间结果，不是严格硬件下界。

硬件 `aic_frequency=1800 MHz` 与报告中的“每 Stage 秒调用次数”是两种不同频率，后者用于衡量 launch 密度。

## 5. 融合开发优先级

| 优先级 | 工作 | 已有证据 | 首次验收 |
| --- | --- | --- | --- |
| P0 | 建立缓存与阶段可比的基线 | prefill 实际工作量严重不一致，decode 混有大 M | 请求 token、缓存状态、每步活动序列和 scheduled token 可核对 |
| P1 | Norm + DynamicQuant | prefill 520 对、decode 2,210 对相邻且 shape 匹配；本机有融合接口 | residual/scale/权重语义正确，匹配组合耗时实测下降 |
| P1 | FFN 量化 MatMul 的 M=15900/15912 | 同 K/N 的邻近 shape 单位工作量更快；双 rank 复现 | 固定 NZ 权重和 dtype，检查 tiling/尾块；同 capture 参照目标约 18.8% 局部降时 |
| P2 | SwiGlu + DynamicQuant | prefill 268 对、decode 1,139 对 | 先解决输入尾轴 17408 超出当前接口文档上限 8192 的问题 |
| P2 | GDN 与布局/状态合并 | 大 M、混合执行存在 Transpose/Slice/Scatter 热点 | 找到可行相邻子链，保持索引与 MTP 状态回退正确 |
| P2 | 通信依赖与图配置 | CANN 非重叠通信约占 Stage 10%，双 rank SIO 接近 | 同序号事件定位可隐藏区间，只合并无真实先后依赖的通信 |
| P3 | 小 M decode 专项 | M<=128 量化 kernel 约 64.410 ms，MFU 模型低 | 在纯 decode 稳态 trace 中评估优先级和真实 TPOT/ITL 收益 |

表中优先级是当前证据下的建议；正式开发保留 TP=2、W8A8 和 MTP。各主报告已给出局部优化目标和条件整网预算，不能把共享算子的预算相加。

## 6. 业务结果作为单独证据

`02` 中现有 8K/128 基线为 C16 的 160 请求与 C32 的 320 请求：输出吞吐分别 103.31/108.14 tok/s，Mean TPOT 分别 130.94/255.89 ms。C32 吞吐高约 4.68%，C16 Mean TPOT 低约 48.83%。这支持后续认真评估低延迟与吞吐的取舍，但一次实验及不同请求量不能作为稳定结论。

业务 OSL128 的接受率分别为 40.28%/39.53%；它们不属于本次 OSL7 profile，不能借用来计算这次 MTP 的有效工作量。MTP 可能一次返回多个 token，TPOT 与流式 ITL 也不能混用。

结果源文件为：

- `02-model-performance-baseline/results/isl8k_osl128_c16/isl8k_osl128_c16.json`
- `02-model-performance-baseline/results/isl8k_osl128_c32/isl8k_osl128_c32.json`
- `02-model-performance-baseline/results/isl8k_osl1_c16/isl8k_osl1_c16.json`
- `02-model-performance-baseline/results/isl8k_osl1_c32/isl8k_osl1_c32.json`

这些文件扩展名为 `.json`，实际保存的是原始 benchmark 文本，未按 JSON 解析或改写。

## 7. 补采方法与本机代码核对

1. **控制缓存**：预热模型/图后，在无活动请求时重置 prefix cache 并确认实际生效，或使用独立输入集；冷/热缓存分别命名、分别分析。记录启动命令和请求清单。
2. **控制阶段**：OSL1 用于 prefill 主导请求；稳态 decode 改用 OSL128/256，并通过服务端 step 标记记录 `num_prefill_tokens`、活动请求数及 MTP 验证/草稿信息。不能通过两个独立 run 的 Stage 相减获得 decode。
3. **核对 bench 的实际行为**：本机 `/vllm-workspace/vllm/vllm/benchmarks/serve.py` 的函数默认 `ready_check_timeout_sec=600`，但 CLI 的 `--ready-check-timeout-sec` 默认值是 **0**，CLI 调用显式传入它，默认跳过 readiness 请求。当前脚本未设置此参数，不能认定它必定多发一次请求。只有显式开启 readiness 或更换版本时，才需要将这部分移出正式 profile。
4. **重复 A/B**：同一请求清单、固定缓存条件、足够 warmup，每次只改一个算子/图/通信参数；同时记录 tok/s、TPOT、ITL 分位数、TTFT、MTP 接受率、显存和数值正确性。

本次只做离线数据分析，没有修改服务参数、清缓存、重新发请求或执行算子替换。

## 8. 可核查的本地实现依据

以下为 2026-09-22 本机安装代码，提供实现口径依据；vLLM checkout 为 `0fc695f`，vllm-ascend checkout 为 `5cb98caaa`。采集时源码是否有额外未提交变更没有快照证明。

| 内容 | 本地文件与入口 |
| --- | --- |
| 通信汇总 | `/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/_communication_parser.py`：`compute_total_info`、`compute_time_ratio` |
| Stage 与缺失 step 的处理 | 同目录 `_trace_step_time_parser.py`：`create_step_file` |
| cube_utilization 定义 | `/usr/local/Ascend/cann-9.1.0/tools/profiler/profiler_tool/analysis/viewer/runtime_report.py`：`cube_usage` |
| Norm/Swiglu 量化融合接口 | `/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/_op_plugin_docs.py`：`npu_add_rms_norm_dynamic_quant`、`npu_swiglu_quant` |
| GDN 混合分支与输出合并 | `/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py`：prefill/spec decode 分支及 `merged_out.index_copy_` |
| Readiness 与 profiler 启动 | `/vllm-workspace/vllm/vllm/benchmarks/serve.py`：benchmark 函数、CLI 参数注册 |
| 随机请求生成 | `/vllm-workspace/vllm/vllm/benchmarks/datasets/datasets.py`：`RandomDataset` |

## 9. 复现与校验

从仓库根目录运行：

```bash
python3 04-operator-profiling-data-analysis/03-summarize-profile.py \
  --output /tmp/bs16-profile-metrics.json
```

脚本只依赖 Python 标准库，不导入 torch、不使用 NPU。输出包含四个场景的双 rank 统计、shape、条件 Roofline、相邻融合候选、通信去重结果和七个主要原文件的 SHA256。

脚本检查：文件存在且非空、聚合 Step 的恒等关系、算子次数与时长一致、两卡类型/次数一致、矩阵 shape 与权重元素数匹配、48 层 QKV 分组一致、通信明细与汇总行一致。每个场景仍只有一次采集，这些校验保证统计自洽，不保证业务样本代表性或优化收益。
