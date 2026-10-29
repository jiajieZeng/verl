# Add-7 无 tool results：204 八卡 SFT

学生 Qwen3.5-9B，全参数 SFT；使用本仓库 Add-4 的 Qwen3.5 fused lm_head/FSDP 修正，复用 VERL 的独立样本 packed attention、token 加权 SFT loss、优化器和 checkpoint。仅此入口替换 loader 和增加监控，不修改通用训练器或 Add-4。

实际实验目录：`/data/zengjiajie/projects/codemem/swe-add-pairing-20261026/add7-noresults-sft-8gpu-20261029`。配置 `qwen35.8gpu.yaml`，宿主机入口 `launch_204.sh`。准备脚本不自动启动训练。

| 参数 | 设置 |
| --- | --- |
| GPU / batch | 单机 8×H20，global batch **64** |
| 并行 | FSDP2 分片 8，SP=4，DP=2；每组每步 32 条 |
| 合批 | `use_dynamic_bsz=true`、`no_padding`、`use_remove_padding=true` |
| token 预算 | 每卡 65,536 × SP4 = 每个 packed microbatch 最多 262,144 tokens |
| 上下文 | 单条最多 262,144，超限报错；不截断、不丢样本、不追加前段 memory |
| 计算 | BF16，FP32 参数/Adam 状态，FlashAttention2、fused loss、gradient checkpointing |
| CPU offload | offload_policy / param_offload / optimizer_offload 全关 |
| 优化器 | 沿用 Add-4：AdamW，lr 1e-5，cosine，warmup 5%，最小 LR 比例 0.1，weight decay 0.01，clip 1 |
| epoch | 3；12,008 行对应 188 步/epoch、564 步；末批 40 条完整保留 |
| thinking / loss | thinking=false，仅原始 summary/facts JSON + im_end；prompt、空 think 前缀及末尾换行均不监督 |
| 验证 / 保存 | 独立 val398；每个 epoch 验证并保存模型/优化器/loader状态；不做在线题目评测 |

优化工作量的两层均衡：每个随机 global batch 的成员保持不变，按 `n*(n+24576)` 将样本分到两组 DP，各组样本数相同；组内继续使用 VERL 的动态 packing 和同步 microbatch 数。这个工作量是近似值，实际收益看 GPU 测量，不承诺固定倍数。样本 position 从 0 重置，底层 cu_seqlens 隔离注意力/线性状态；合批不会把不同轨迹变成相互可见的一条上下文。

`prepare_data.py` 一次性编码到本地 NVMe 的 int32 mmap，训练时直接读取；完整回读所有监督 target，保持 JSON 原文，不把教师 reasoning 放入目标。分别接入此批 `delivery/messages.jsonl` 和 `val/messages.jsonl`，要求 train=12008、val=398、passed/failed 已判分、tool_results_removed=true。预检核对源 group 无交集和完整覆盖。只存在6000源输入包或旧12220有工具返回数据时不能启动。

## 204 上的命令

```bash
cd /data/zengjiajie/projects/verl/verl/examples/sft/add7
./launch_204.sh check                 # 仅 CPU，展开并检查 Hydra 配置
./launch_204.sh prepare-val           # 一次性；已有缓存不要重跑
./launch_204.sh prepare-train         # 等完整 delivery 到位；缺失即报错
./launch_204.sh preflight             # 核对两份缓存，写 reports/preflight.json
# 以下两条需要 GPU 空闲，准备工作本身不会执行：
./launch_204.sh smoke                 # 最长样本组成的3个 batch64，独立 checkpoint
./launch_204.sh train                 # 正式3 epoch，从 base 开始，绝不自动复用 smoke
```

脚本拒绝占用中的 GPU，不停止现有 Add 生成或任何用户任务。CPU 模式不申请 GPU。模型只读挂载已有共享权重 `/data/memory-verl-sft/models/base/Qwen3.5-9B` 为 `/model`；项目内不复制权重。镜像 `verl-sft:qwen3.5-9b-cu130-driver535`，源码显式挂载本仓库且优先于镜像旧 `/opt/verl`。每次启动建立独立 `runs/<mode>-<timestamp>/`，记录源码 HEAD、已有修改、入口副本、镜像ID和 override；Hydra 保存实际完整配置。六项代理清除，网络关闭。

GPU smoke 保留正式 564 步的 LR schedule，只提前在第三步停止；覆盖较长样本而不是仅测试短样本。它只用于测显存、packing、step time 和 loss，不用于接续正式训练。默认无 CPU offload、SP4/65K 的组合参照旧 Add-4；八卡实测前仍可能需要调 token 预算或 offload，任何调整应记录并重新 smoke。CPU 单测不能代替真实分布式 GPU 验收。

正式训练 `training.log` 记录每组每步的 microbatch 数、样本数、token 数；`metrics.jsonl` 记录 loss、grad norm、LR、MFU、最慢 rank step 秒数、全机 tokens/s、峰值显存。对比速度时排除加载、首次编译、验证/保存，并观察稳定数步。

恢复使用同一 prepared 数据和配置，并指定独立旧 checkpoint：

```bash
./launch_204.sh train trainer.resume_mode=resume_path \
  trainer.resume_from_path=/absolute/path/to/checkpoints/global_step_188
```

Sampler 无自定义游标状态，由 StatefulDataLoader 保存/恢复已消费位置；trainer 恢复 epoch，再构造相同顺序。不能改变 batch/SP/采样规则后沿用旧 loader checkpoint。

CPU 回归：`python -m unittest discover -s tests/add7 -v`（在该训练镜像中，挂载模型为 `/model`）。覆盖 12,008 行末批、398 val、非零 epoch 恢复、最长样本 smoke、缓存完整监督、错误来源拒绝、packing token 上限和 token 加权 loss/梯度等价。GPU/SP 真正数值回归另见 `tests/special_distributed/test_qwen35_linear_attention_ulysses_sp.py`，本次未占卡运行。
