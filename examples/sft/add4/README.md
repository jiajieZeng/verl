# Add-4 SFT / OPD（VERL 正式入口）

Qwen3.5-9B 与 Qwen3.6-35B-A3B 的 SFT、OPD 均在本 VERL 仓库维护。冻结教师为 204 的 Qwen3.8-Flash-Next，训练侧只通过 HTTP 请求教师。原 3.5 SFT 入口、dataset、exporter 保持原实现；OPD 只在独立进程入口启用，不改 VERL 通用训练器源码。

## 入口

以下路径均相对于本目录。

| 模型 / 训练方式 | Python 入口 | SFT 基础配置 | OPD 配置 |
| --- | --- | --- | --- |
| 3.5 SFT | `train_add4.py` | `qwen35.sft.example.yaml` | 无 |
| 3.6 SFT | `train_add4_qwen36.py` | `qwen36.sft.example.yaml` | 无 |
| 3.5 全词表 OPD | `train_add4_opd_full_qwen35.py` | `qwen35.opd.monitor.yaml` | `opd.full.qwen35.example.yaml` |
| 3.6 全词表 OPD | `train_add4_opd_full_qwen36.py` | `qwen36.opd.monitor.yaml` | `opd.full.qwen36.example.yaml` |
| 3.5 top-128 OPD（保留） | `train_add4_opd.py` | `qwen35.opd.monitor.yaml` | `opd.example.yaml` |
| 3.6 top-128 OPD（保留） | `train_add4_opd.py` | `qwen36.opd.monitor.yaml` | `opd.qwen36.example.yaml` |

SFT 复用既有 Add Parquet，固定标签监督 facts + EOS。OPD 从同一数据取得 prompt IDs，由当前学生生成响应；教师和学生均使用相同原始 IDs，不再分别应用 chat template。原 Add 非思考协议保持显式关闭 thinking；采样温度 0.6，损失分布温度 1.0。

OPD 正式监控入口：[W&B、指标口径、完整398条验证与启动检查](MONITORING.md)。使用表中的 `*.opd.monitor.yaml` 配置；历史 SFT 配置不变。

## 全词表损失和传输

定义有效词表 V = [0, 248077)。输出头仍有 248320 行，排除末尾 243 行后，师生各自在 V 上重新归一化：

```text
pS = softmax(student_logits[:248077])
pT = softmax(teacher_logits[:248077])
KL(t) = sum over ALL v in V: pS(v) * (log pS(v) - log pT(v))
loss = mean_response(KL(t) - 0.005 * H(pS))
```

全词表版 `teacher_top_k: null`，没有 top-128 截断，也没有 tail bucket。只对学生生成的 facts 和实际 EOS 求损失，不对 prompt 求损失。top-128 版使用全有效词表概率，加入尾部桶后计算 K+1 个桶上的 reverse KL：`sum_K pS*(log pS-log pT) + pS_tail*(log pS_tail-log pT_tail)`，其中尾部质量为有效词表中 K 之外的总概率。集合内不重新归一化；这是合并尾部后的 KL，不等同于全词表 KL。旧的无尾部截断项可能为负并鼓励质量流出 K，现要求 `tail_bucket: true`，旧配置会报错，不静默改变其训练目标。教师尾部按 FP64 计算 `max(-expm1(log_mass), 2^-24)`；解析与 loss 统一允许单项及总质量的 log 值最多为 `1e-5`，因此 FP32 舍入为 0 或容差内微正值不会导致训练退出。下限仅平滑尾部，保留 top-K 分数，不做集合内归一化；下限生效时是近似目标，桶总质量可能略超 1，不能保证损失严格非负，也不能恢复真实极小尾部。超出容差的坏数据仍报错，解析阶段优先拦截；当前流程没有跳过批次机制，未处理的错误会终止训练。

教师原生 `/generate` 请求以 `top_logprobs_num=248320` 获取全部输出行，使用 SGLang 镜像已有的 `return_flat_raw_top_logprobs_b64` 二进制接口。这个接口字段名含 top，但请求宽度等于整个输出头；接收端强制验证完整 ID 排列、字节数、概率质量、位置、采样 token 分数和冻结版本，少一列就失败。未定义行在接收后排除。

每次最多评分 8 个响应位置；每块携带相同 prompt 加此前所有学生响应，不改变上下文。FP32 教师分布逐块写入训练机 NumPy mmap，训练张量只携带每个 token 的样本/行号。输出头每 8 个响应位置计算一次完整 KL，反向重算；不会创建 prompt 长度 × 全词表的监督张量。

两条损失路径均在矩阵乘法前把 hidden/head 转为FP32（FP64输入保留数值检查精度）；log-softmax、logits梯度和跨响应块的head梯度累加保持FP32，只在返回autograd时转回输入/参数dtype。BF16参数仍会接收BF16梯度，FSDP/优化器后续精度不由此处改变。该修复增加一个有效词表head的FP32投影副本和FP32梯度缓冲；完整35B的GPU峰值显存和吞吐仍待验收。

两个 `slot` 轮换存储当前/上一批的完整教师分布；成功评分后原子替换本 slot 的样本文件，失败保留原文件及 partial。每步不可变审计保存 prompt/response IDs、版本、形状及归一化分布 SHA-256；完整概率缓存不是永久历史归档。每次启动/恢复创建新的 attempt，历史审计保留。

## 环境与 204 连接

- 梯度训练：`verl-sft:qwen3.5-9b-cu130-driver535`，挂载本仓库并将其置于 `PYTHONPATH` 最前面。不要继续从 OpenClaw-RL 启动新的训练。
- 教师和学生 rollout：同一镜像 `local/qwen38-add-teacher:sglang0.5.21-cu130-tp8-20261017`，SGLang 0.5.21、BF16、FP32 Mamba 状态、无量化、262144 上下文。
- 训练宿主机 `http://127.0.0.1:18081` → SSH → `47.103.58.204:127.0.0.1:18082`。教师名称 `Qwen3.8-Flash-Next`，冻结版本 `qwen38-frozen`。两模型的四份 OPD 配置均固定该连接。
- `start_opd_teacher_204_tunnel.sh` 只建立这个转发，复用本机已有 SSH 身份。训练容器使用 host 网络；换训练机必须在实际训练宿主机启动转发。
- `start_opd_student.sh` 启动本轮专用学生服务；训练进程和学生服务需以同一绝对路径挂载 `sync_dir`。它会热更新学生，不能指向教师或现有业务服务。
- 新 OPD 入口、连接脚本及客户端清除/绕过六项 HTTP 代理。preflight 核对版本、模型角色、冻结版本、tokenizer 探针和真实评分协议。

2026-10-19本机隧道已修正至204实际服务端口18082，原18080返回401。正式客户端实测`scoring_ready`：13个tokenizer探针通过；全输出头248320列校验后保留248077有效词，15响应位置以8+7分块写入FP32 mmap，形状`[15,248077]`，归一化最大误差1.91e-7；top-128同时通过。评分为固定诊断IDs，并非实际学生rollout；262K长输入评分和完整训练步仍待联调。证据：`/data/zengjiajie/projects/verl/qwen38-add-204-262k-20261018/opd-teacher-check-20261019/`。

独立教师验收（在有 Transformers 的训练容器中）：

```bash
python "$VERL_ROOT/examples/sft/add4/check_opd_teacher.py" \
  --config "$VERL_ROOT/examples/sft/add4/opd.full.qwen35.example.yaml" \
  --tokenizer "$ADD_TOKENIZER_DIR" \
  --report "$ADD_RUN_DIR/teacher-check.json" --timeout 600
```

3.6 换对应 full 配置。省略 tokenizer 只核验元数据；`scoring_ready` 才表示完整评分预检通过。训练入口会强制执行师生预检。

## 新实验运行方法

以下命令在已分配 GPU、挂载源码/模型/数据/本轮目录的训练容器内执行。配置只给准备参数，默认 8 卡 FSDP2、SP=1、CPU offload、BF16 计算、FP32 主权重/Adam、单步保存；历史 SFT run 的参数不改。

```bash
export VERL_ROOT=/data/zengjiajie/projects/verl/verl
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# 必填：QWEN35_MODEL_DIR 或 QWEN36_MODEL_DIR、ADD_TOKENIZER_DIR、ADD_TRAIN_FILES、
# ADD_RUN_ID、ADD_RUN_DIR；后两者须指向本轮专属名称与目录。
python -c 'import verl; print(verl.__file__)'
# 输出应位于上述 VERL_ROOT，而非镜像中旧 /opt/verl。
```

SFT 3.5：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  "$VERL_ROOT/examples/sft/add4/train_add4.py" \
  --config-path "$VERL_ROOT/examples/sft/add4" --config-name qwen35.sft.example
```

SFT 3.6 换 `train_add4_qwen36.py` 和 `qwen36.sft.example`。

OPD 先准备专用学生、在 SGLang 环境运行 `export_opd_logit_processor.py --output <本轮mask.json>`，再将对应 OPD 配置复制到 `ADD_RUN_DIR/opd.yaml`，填 `run_id`、`student_url`、`student_logit_processor_file`。学生 `OPD_RUN_ID` 必须等于配置 `run_id`；默认同步目录为 `/data/zengjiajie/projects/verl/add4-opd/student-weight-sync`。

全词表 OPD 3.5：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  "$VERL_ROOT/examples/sft/add4/train_add4_opd_full_qwen35.py" \
  --opd-config "$ADD_RUN_DIR/opd.yaml" \
  --config-path "$VERL_ROOT/examples/sft/add4" --config-name qwen35.opd.monitor
```

全词表 OPD 3.6 换 `train_add4_opd_full_qwen36.py`、`opd.full.qwen36.example.yaml` 和 `qwen36.opd.monitor`。专用入口拒绝错误学生架构或 top-128 配置。模型路径是本地已有权重，不会自动下载。原 top-128 入口保留，使用表中对应 profile。

每步流程：当前学生全量权重导出到两个 HF slot 之一 → 专用学生 SGLang 热更新并校验版本 → 温度 0.6 生成 → 204 教师评分 → VERL reverse KL 反传/更新 → 原 checkpoint/验证生命周期。验证继续使用固定标签 SFT CE；原独立 SFT 进程不安装 OPD hooks。

## 资源与验证

全词表模式默认评分并发 1、块大小 8，可配置并发至 4；限制只控制瞬时内存/请求数，不截断词表。当前完整概率 mmap 要求单训练节点、共享本地目录，SP=1、packed/no_padding、`pad_to_length=false`。

按当前 FP32 分数 + int32 ID + base64 传输，约 **2.65 MB/响应 token**；8192 token 约 **21.7 GB** 网络数据，尚未计重发前缀与 HTTP 开销。落盘归一化概率约 **0.992 MB/token**，8192 token 约 **8.13 GB/样本**；batch32 最长响应约 **260 GB/slot**，双 slot 约 **520 GB**，另留原子写入临时空间。每步仍需导出约 18 GB / 72 GB 学生权重并重载。全词表实现完成不代表这个通信方案已经达到可用吞吐。

8 卡模板指训练卡；学生 rollout 另占推理卡，204 教师独立部署。262K 是上下文配置上限，真实混合注意力 GPU 容量和吞吐需在指定资源上验收。

迁入时 **106 项 CPU 测试通过**；后续数值与监控验收见[监控说明](MONITORING.md#验收入口)。验证结果见 `/data/zengjiajie/projects/verl/add4-opd/tests/`：SFT 回归、top-128、完整词表梯度对照/gradcheck、248320 行协议、HTTP 分块/原子写入、VERL 变长分批行映射、实际小型 dense/MoE 反传、专家权重导出重载和配置组合。另直接调用部署镜像中的原生编码器验证全宽二进制协议。2026-10-19已完成180B在线全词表预检、15响应token分两块评分并落盘、top-128预检；尚未执行多卡GPU训练、真实学生热更新及长上下文评分吞吐联调。

## 迁入记录

2026-10-18：从 OpenClaw-RL `ab4492ba` 的 `scripts/swe_harbor/sft/` 迁入 Add SFT/OPD 入口及相关测试；此前这些包装入口本来就调用 VERL 引擎。正式维护位置改为本目录，全词表功能只在本仓库新增。旧源码与历史 run 原位保留，避免历史路径失效。原 `train_add4.py`、`add4_sft_dataset.py`、`export_add4_parquet.py` 内容保持一致。新配置使用 `VERL_ROOT` 与本项目输出目录。迁移明细见 `/data/zengjiajie/projects/verl/add4-opd/migration-20261018.json`。

## GPU常驻参数的SFT

144四卡35B训练可使用 `engine.offload_policy=false`，参数和梯度保留CUDA；`optim.optimizer_impl=add4_streaming_adamw`、`optim.optimizer=StreamingAdamW` 保持FP32 AdamW更新，每次只把一个参数的一阶/二阶矩加载到GPU，更新后存回CPU。保持通用引擎 `optimizer_offload=false`，避免引擎把所有状态一次性搬回。该包装器拒绝BF16参数、foreach/fused/capturable/differentiable模式；不会更改参数或状态精度。当前正式run只保存model+extra，不使用优化器状态恢复。

CPU/CUDA多参数组、无梯度参数、学习率调度、AMSGrad等5项测试通过；两卡FSDP2连续3步与原AdamW参数和矩状态逐元素完全一致。完整35B最长样本GPU峰值仍由144实际验收日志判定。测试入口 `tests/add4/test_add4_streaming_adamw.py`、`tests/add4/check_streaming_adamw_fsdp.py`，实现提交 `2ec16018`。

需要少量参数卸载时，Qwen3.6 SFT入口支持显式设置 `ADD4_CPU_OFFLOAD_LAYERS=36,37,38,39`（从0计数）。仅这些decoder层使用原生FSDP2 `CPUOffloadPolicy`，其他层继续常驻GPU；不改变反向预取、精度或训练目标，未设置变量时保持原行为。与 `StreamingAdamW` 配合时，CPU层在CPU更新，CUDA层仍逐参数暂存Adam状态。两卡混合放置的加载、梯度累积、裁剪及3步更新验证入口为 `tests/add4/check_partial_offload_fsdp.py`；使用与训练相同的 `cpu:gloo,cuda:nccl` 后端。

本机 2026-10-22 的四卡 SFT 准备入口见 `/home/zengjiajie/workingdocker/experiments/add4-qwen36-sft-144/local-20261022/README.md`。该批次使用末 8 层卸载，每 20 步及每轮末保存 model/optimizer/extra。显式设置 `ADD4_KEEP_EPOCH_CHECKPOINTS=1` 与 `ADD4_KEEP_LATEST_CHECKPOINTS=2` 时，Qwen3.6 专用入口关闭内置数量轮换，使用 `add4_checkpoint_retention.py` 在完整保存后保留最新 2 个及全部 epoch checkpoint。清理仅作用于本轮有完成标记的普通 checkpoint；未启用变量的历史批次行为不变。CPU 保存策略、优化器序列化恢复测试已通过，真实四卡运行及 35B 完整保存尚未执行。
