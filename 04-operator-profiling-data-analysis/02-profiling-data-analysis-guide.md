# Profiling 数据分析说明

## 1. 文档范围

本文说明以下目录中的 profiling 分析结果：

```text
04-operator-profiling-data-analysis/vllm_profile/<run>_ascend_pt/ASCEND_PROFILER_OUTPUT/
```

当前示例目录为：

```text
04-operator-profiling-data-analysis/vllm_profile/rank0_17391_20260920075211024_ascend_pt/ASCEND_PROFILER_OUTPUT/
```

`<run>_ascend_pt` 是一次 profiling 采集结果。多卡场景通常会生成多个 rank 目录，例如 `rank0_*_ascend_pt`、`rank1_*_ascend_pt`，分析时应先分别查看各 rank，再进行横向对比。

## 2. 推荐文件格式

使用 Markdown 保存说明最合适：

- 可以直接放在 Git 仓库中进行版本管理。
- 表格适合快速查看文件用途、优先级和查看方式。
- 可以嵌入命令、路径和分析顺序。
- 后续新增文件时容易扩展，不依赖特定办公软件。

## 3. 文件分类和作用

### 3.1 优先查看的结果文件

| 文件 | 类型 | 作用 | 是否重点查看 |
| --- | --- | --- | --- |
| `step_trace_time.csv` | CSV | 查看 step 或阶段级耗时，反映前向、反向、通信和空闲等阶段的时间分布。 | **是，首先查看** |
| `operator_details.csv` | CSV | 查看算子级耗时、调用次数、输入输出信息和耗时占比，用于定位算子瓶颈。 | **是，重点查看** |
| `kernel_details.csv` | CSV | 查看底层 Kernel 的执行时间、调用次数和设备侧执行情况，用于进一步定位算子内部的 Kernel 瓶颈。 | **是，重点查看** |
| `trace_view.json` | JSON | Chrome Trace 格式的时间线数据，可使用 MindStudio Insight 打开，观察 Host、Device、算子、Kernel 和通信的时序关系。 | **是，重点查看** |
| `op_statistic.csv` | CSV | 算子统计和利用率数据，用于观察算子执行效率、耗时占比和资源利用情况。 | **是** |
| `api_statistic.csv` | CSV | API 调用统计，用于查看 Host API 调用次数、耗时和调用分布。 | 需要时查看 |

### 3.2 通信和补充分析文件

| 文件 | 类型 | 作用 | 是否重点查看 |
| --- | --- | --- | --- |
| `communication.json` | JSON | 通信操作的汇总信息，例如通信操作、耗时或通信量等。用于判断通信是否成为瓶颈。 | 多卡场景建议查看 |
| `communication_matrix.json` | JSON | 通信矩阵数据，用于观察不同设备或 rank 之间的通信关系和分布。 | 多卡场景建议查看 |
| `task_time.csv` | CSV | Task 级时间数据，比算子层更接近设备任务执行，用于辅助分析 Kernel、Task 排队和执行耗时。 | 出现设备空闲或调度异常时查看 |

### 3.3 数据库和状态文件

| 文件 | 类型 | 作用 | 是否直接查看 |
| --- | --- | --- | --- |
| `analysis.db` | SQLite 数据库 | 分析后的综合结果数据库，供分析工具或查询逻辑使用。 | **通常不直接查看** |
| `ascend_pytorch_profiler_0.db` | SQLite 数据库 | Ascend PyTorch Profiler 的原始或中间结构化数据。 | **通常不直接查看** |
| `analyse.done` | 标记文件 | 表示 profiling 分析流程已经完成。 | 不需要分析内容 |

数据库文件不建议直接用文本编辑器查看，也不建议直接修改。优先使用已经生成的 CSV、JSON 和可视化工具；只有在需要二次统计、字段缺失或排查分析流程时，才考虑通过 SQLite 查询数据库。

## 4. 推荐分析顺序

### 第一步：确认分析是否完成

确认存在：

```text
ASCEND_PROFILER_OUTPUT/analyse.done
```

如果没有该文件，先重新执行数据解析脚本：

```bash
python3 04-operator-profiling-data-analysis/01-analyse-profile.py
```

### 第二步：查看整体阶段耗时

先查看 `step_trace_time.csv`，回答以下问题：

- 每个 step 的总耗时是否稳定？
- 哪个阶段耗时最长？
- 是否存在明显的空闲时间？
- 是否存在某个 rank 的耗时明显高于其他 rank？

该文件用于建立整体性能基线，不建议一开始就从海量 Kernel 明细入手。

### 第三步：查看算子级热点

查看 `operator_details.csv` 和 `op_statistic.csv`：

- 按总耗时降序查找热点算子。
- 同时关注调用次数、单次平均耗时和总耗时占比。
- 区分计算算子、数据搬运算子和通信算子。
- 对比不同 rank 是否存在负载不均衡。

这一步用于回答“哪个算子最值得优化”。

### 第四步：下钻到 Kernel 和 Task

针对第三步找到的热点算子，查看：

1. `kernel_details.csv`
2. `task_time.csv`

重点确认：

- 热点算子由哪些 Kernel 构成。
- Kernel 是执行时间长，还是调用次数过多。
- 是否存在大量短 Kernel。
- 是否有任务排队、同步或设备空闲。

这一步用于回答“算子慢的具体底层原因是什么”。

### 第五步：使用时间线确认时序

使用 MindStudio Insight 打开 `trace_view.json`，核对：

- Host API、算子和 Device Kernel 的对应关系。
- Kernel 之间是否存在空洞或不连续执行。
- 通信是否阻塞计算。
- 是否存在同步等待、串行执行或流水线断点。
- profiling 区间内是否混入启动、预热或异常请求。

时间线是验证 CSV 统计结论的主要工具。CSV 适合排名，Trace 适合理解因果和时序。

### 第六步：分析多卡通信

多卡或 Tensor Parallel 场景查看：

- `communication.json`
- `communication_matrix.json`
- `step_trace_time.csv`
- `trace_view.json`

重点判断：

- 通信耗时是否占比较高。
- 各 rank 的通信耗时是否一致。
- 是否存在通信与计算无法重叠。
- 是否有单个 rank 成为整体瓶颈。

### 第七步：必要时查看 API 统计和数据库

当算子、Kernel 和 Trace 无法解释问题时，再查看：

- `api_statistic.csv`：定位 Host API 调用过多或调用耗时异常。
- `analysis.db`：查询分析结果中的补充字段。
- `ascend_pytorch_profiler_0.db`：排查原始 profiling 数据或分析过程中的字段问题。

## 5. 按性能问题选择文件

| 问题 | 首选文件 | 辅助文件 |
| --- | --- | --- |
| 整体 step 变慢 | `step_trace_time.csv` | `trace_view.json` |
| 某个算子耗时高 | `operator_details.csv` | `op_statistic.csv`、`kernel_details.csv` |
| Kernel 执行慢 | `kernel_details.csv` | `task_time.csv`、`trace_view.json` |
| 设备利用率低 | `op_statistic.csv` | `trace_view.json`、`task_time.csv` |
| Host 调度开销高 | `api_statistic.csv` | `trace_view.json` |
| 多卡通信耗时高 | `communication.json` | `communication_matrix.json`、`trace_view.json` |
| Rank 间负载不均衡 | 各 rank 的 `step_trace_time.csv` | 各 rank 的算子、Kernel 和通信文件 |
| 怀疑同步或空闲 | `trace_view.json` | `task_time.csv`、`step_trace_time.csv` |

## 6. 不建议的查看方式

- 不要直接用文本编辑器打开 `*.db` 文件。
- 不要只看 `operator_details.csv` 就下结论，算子级耗时还需要结合时间线和 Kernel 明细确认。
- 不要只看单个 rank；Tensor Parallel 场景需要比较所有 rank。
- 不要把 profiling 期间的预热、模型加载或异常请求混入性能基线。
- 不要直接修改 `ASCEND_PROFILER_OUTPUT` 中的结果文件；如需处理，复制到单独的分析目录。

## 7. 最简执行流程

```text
确认 analyse.done
    -> step_trace_time.csv
    -> operator_details.csv / op_statistic.csv
    -> kernel_details.csv / task_time.csv
    -> trace_view.json
    -> communication.json / communication_matrix.json
    -> api_statistic.csv 和数据库（必要时）
```

对于本项目的 Qwen3.8 Tensor Parallel 服务，建议至少保留并分析以下文件：

```text
step_trace_time.csv
operator_details.csv
kernel_details.csv
op_statistic.csv
trace_view.json
communication.json
communication_matrix.json
```
