# QuantBatchMatmulV3 开发任务交接

日期：2026-09-22。

## 1. 给接手 Agent 的任务

阅读 [开发设计](01-quantbatchmatmulv3-optimization-design.md) 和 [验收方案](02-quantbatchmatmulv3-validation-plan.md)，实现并验证 Qwen3.8-27B W8A8 FFN 量化 MatMul 的定向优化。首版 shape 为 `M=15900/15912,K=5120,N=17408`；M=15920 作为通过验收后增加的目标。

本次交付仅包含设计。**当前没有 baseline microbenchmark、候选 wrapper、AscendC kernel、集成 patch 或优化实测结果。** 下列文件名和开关是待实现约定，不能当作已存在的命令或官方 API。

执行顺序是：复现 → 测 pad/split 完整收益 → 必要时写专用 kernel → 正确性/性能/图验证 → 集成与双卡 A/B。选出满足要求的最简单方案即可，不要求同时交付三套生产实现。

## 2. 环境与入口

| 内容 | 路径/值 |
| --- | --- |
| 本任务仓库 | `/home/zhujianwei/task02/vllm-ascend-qwen3.8-27b-op-opt` |
| 设计与单算子代码 | `05-operator-development/` |
| 集成补丁 | `06-operator-integration/` |
| 整网验收 | `07-final-performance/` |
| vLLM-Ascend 源码 | `/vllm-workspace/vllm-ascend` |
| vLLM 源码 | `/vllm-workspace/vllm` |
| 模型 | `/home1/model/Qwen3.8-27B-w8a8/` |
| 原调用入口 | `vllm_ascend/quantization/methods/w8a8_dynamic.py::AscendW8A8DynamicLinearMethod.apply` |
| CANN | `/usr/local/Ascend/cann-9.1.0` |
| 版本 | vllm-ascend 0.23.0、torch_npu 2.10.0.post4 |
| 固定业务约束 | TP=2、W8A8、MTP=3 |

设计时 vLLM-Ascend commit 为 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。开始前检查当前工作区和源码状态；用户可能已经做过新修改，以实际源码为准并记录差异。

如果 `/vllm-workspace` 是只读，先在当前允许写入的工作区建立独立开发副本/检出，再生成可复现补丁。不要为解决写权限直接改系统 CANN；安装/使用开发构建时必须检查实际 import 路径，避免误测原 wheel 或误替换用户正在运行的服务。

## 3. 分阶段执行表

| 阶段 | 工作与产物 | 进入下一阶段的条件 |
| --- | --- | --- |
| S0 环境 | 记录版本/commit/SoC/设备映射；验证最小 NPU 运算；创建小型结果目录 | 运行环境可测，或明确记录无法测量的外部原因 |
| S1 基线 | 实现原调用 benchmark、shape 扫描、正确性基准和 samples/summary | 目标形状测量稳定、参数契约明确、效率下降可重复或已有完整解释 |
| S2 wrapper | 实现 pad_m、split_m 原型，逐元素对照并统计完整成本 | 至少一条路线达到门槛则选最简路线进入 S4；否则基于证据决定 S3 |
| S3 专用核 | 实现独立 op_host/op_kernel、tiling、绑定和测试 | kernel 与完整调用达标；不满足就停止此路线并记录原因 |
| S4 集成准备 | 窄范围 dispatch、启动开关、fallback、Meta/Fake、eager/graph 测试、模块覆盖统计 | 目标与非目标行为正确、无明显回退、资源生命周期明确 |
| S5 整网 | 固定业务配置做 C16/C32 A/A 与 A/B、独立短 trace | 报告真实收益/噪声/覆盖率，保留一键回退 |
| S6 交付 | 小型结果、复现步骤、源码/patch、结论与未解决项 | 另一执行者可以从记录复现，不依赖未记录的系统修改 |

S0/S1 不依赖再次确认精确根因。设计中“不确定的 tiling/尾块原因”是要通过实验解决的问题；执行者应按已有范围继续。若 NPU 权限或设备挂载缺失，完成独立于 NPU 的源码和测试准备，并清楚说明哪些结果尚未取得。

## 4. 建议文件结构

以下为后续实现约定，现在尚未创建：

```text
05-operator-development/
  quantbatchmatmulv3/
    README.md                     真实可执行命令、依赖与结果索引
    bench_qmm.py                  原调用/候选的统一完整计时
    cases.py                      固定 shape、输入构造、精度检查
    candidates.py                 baseline / pad_m / split_m 原型
    tests/
      test_qmm_correctness.py     目标形状与边界、生命周期、多 stream
      test_qmm_dispatch.py        白名单、fallback、dtype/format 负面条件
    ascendc/                      仅在路线 C 确定需要时创建
      op_host/
      op_kernel/
      CMakeLists.txt
    results/<run_id>/             小型数值、环境与实验结论

06-operator-integration/
  quantbatchmatmulv3/
    README.md                     对应commit、应用/构建/启用/回退
    patches/                      对vLLM-Ascend的最小集成patch
    tests/                        图与真实调用路径测试

07-final-performance/
  quantbatchmatmulv3/<run_id>/
    baseline/                     vllm bench原始结果及配置
    candidate/
    comparison.md                 A/A噪声、A/B收益与覆盖范围
```

保持简单即可，若 repo 已有合适工具则复用。只为实现所需阶段创建文件；不需要预先搭建通用 autotune 平台或大量空目录。

建议未来 benchmark 支持 `--impl baseline|pad_m|split_m|ascendc`、`--m`、`--warmup`、`--repeats`、`--rounds`、`--seed`、`--device`、`--output-dir`。K/N/dtype 可先固定为本任务值，避免设计通用算子框架。参数名是拟定接口，最终实现后要在 README 写真实命令，不留下不可执行模板。

## 5. 各阶段最小交付要求

### S1

- 用同一 W、同一最大 A 的前 M 行比较相邻 M，真实 NZ 布局由官方接口产生。
- warmup 与正式计时分离；至少 5×20 次，A/A 估计噪声；保留原始样本。
- 确认 BF16 weight_scale、FP32 token_scale、bias/offset=None，记录实际 layer 前缀计划。
- 同时报告无 profiler 的 event/wall 时间和独立短 profile 中的 kernel 时间。
- 基线表填实测，不复制历史 5.77 ms 充当微基准结果。

### S2

- pad_m：M 补到 16384，INT8 尾部零、token scale 尾部 1，返回真实 M 行；完整成本计时。
- split_m：先测 M0=15360 和实际尾块，完整计入两次调用、拼接与输出分配。
- 输出与 baseline 全量逐元素一致、原输入未修改、连续两次调用输出不串扰。
- 明确选择一条方案、继续 S3，或结束该方向的实验理由。wrapper 有净收益即可交付，不要求为了“自定义算子”强行重写。

### S3

- 独立名称、源码路径、合法 tiling；使用真实 NZ descriptor，输出无 bias 的 BF16 语义不变。
- 按完整 K 计算每个独占输出 tile；有限 workspace、正确的 producer/consumer 同步；无越界与 split-K 引入的语义变化。
- kernel-only 和完整 API 分别计时；不写完整 INT32 中间矩阵后忽略其带宽成本。
- 复用现有构建模式，记录 SoC、编译器、库版本；没有可运行构建时不得标记路线完成。

### S4/S5

- 拟新增 `QWEN_QMM_IMPL` 默认 baseline。命中模块、M/K/N、dtype、NZ、bias 等全部条件后才调用候选。
- 不支持的 shape/模式回原路径；kernel 错误明确失败，不能静默兜底掩盖。
- Meta/Fake、输出 alias、资源复用、实际图路径验证；切换候选重新启动/预热图。
- 服务保持双卡 TP/W8A8/MTP；C16/C32 测试使用相同请求清单、固定缓存状态。
- 明确记录目标实际命中次数。整网差异未超过噪声时，结论为“单算子有效，整网收益尚未证实”。

## 6. 不能遗漏的已知陷阱

| 陷阱 | 正确做法 |
| --- | --- |
| 将 `bs16_decode` 当成纯 decode | 本次大 M 主要来自 prefill/混合执行，收益按真实阶段解释 |
| 将 18.8% 乘以全部量化 MatMul 时间 | 本期只覆盖少量 shape；BS16 目标累计约 772.937 ms |
| 看 kernel 名相同就否定 tiling 差异 | 内核可用不同 runtime tiling；先取证，根因未定 |
| 从 `[544,320,16,32]` 直接 reshape 权重 | 使用真实 NZ 格式转换/描述符，并核对逻辑 K/N |
| 补齐后漏算 copy/分配/输出处理 | 完整调用端到端计时 |
| 固定一个全局 output/workspace | 保证在途调用、stream 和 graph 生命周期互不冲突 |
| 把小误差当作可忽略 | 首版严格匹配原结果；MTP 接受/拒绝会放大语义变化 |
| 只看一次吞吐变化 | A/A 噪声+配对 A/B，约 0.9% 的目标需要重复测量 |
| 把图 Meta 注册当作图验证完成 | 真实 capture/replay 与动态 shape 回退分别验收 |
| 按 Task ID 唯一关联 | 加 stream、时间戳及必要的 model/context 消歧 |

## 7. 结束时应给用户的结果

明确说明最终采用的路线、支持范围、独立基准的前后耗时、正确性与图测试结果、显存代价、两卡 A/B 结果、启用/关闭方法、源码位置。

若任何方案都未取得净收益，也要交付基线与失败原因，例如“pad 的搬运抵消计算收益”“不同 tiling 的收益无法重复”“候选数值不满足严格一致”“集成覆盖率低”。不要为了完成任务把未达标方案默认启用，也不要用条件预算冒充实测。

原始 profiling、模型参数和大构建产物不提交 Git；用户已有修改保持原样。执行者是否提交/推送以接手时用户授权为准，本轮只要求设计文档。
