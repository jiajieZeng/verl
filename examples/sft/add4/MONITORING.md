# OPD 监控与 W&B

适用 Qwen3.5 / Qwen3.6 的全词表与 top-k OPD。独立 SFT 入口和历史 run 不变。正式 OPD 使用 `qwen35.opd.monitor.yaml` 或 `qwen36.opd.monitor.yaml`，再传对应 `--opd-config`。数值、优化器、批量和模型设置与原 SFT 模板一致，有配置对照测试。

## 启动连接

1. 在实际训练宿主机设置已有 `VERL_ROOT`、本轮独立 `ADD_RUN_ID`、`ADD_RUN_DIR`，准备原 OPD profile、模型、数据、专用学生和教师。监控不负责启动模型或替训练分配 GPU。
2. 用 `bash "$VERL_ROOT/examples/sft/add4/with_opd_resources.sh" <原 docker/训练命令及参数>` 包裹训练命令。它先确认本机与204的 GPU 可读，再每30秒原子更新 `$ADD_RUN_DIR/resources.json`，训练退出后结束自己的采集进程。仅训练宿主机需要 SSH 身份，复用 `$HOME/.ssh/dockerserver`，可用 `OPD_TEACHER_SSH_IDENTITY` 指定既有路径。没有修改远端进程。
3. 训练容器挂载整个 `ADD_RUN_DIR` 到相同绝对路径；挂载验证 Parquet；只读挂载现有 `/home/zengjiajie/.config/verl-sft/wandb.env` 到 `/run/secrets/wandb.env`。容器内执行 `set -a; . /run/secrets/wandb.env; set +a` 后启动原 torchrun。凭据不进入配置、manifest、日志和仓库。
4. 3.5 的原训练命令将 `--config-name qwen35.sft.example` 改为 `--config-name qwen35.opd.monitor`；3.6 使用 `qwen36.opd.monitor`。OPD 入口继续清除六个代理环境变量。W&B 强制 online，不能用 offline 冒充上传成功。

每次启动/恢复建立 `opd-<OPD run_id>-<随机 attempt>`，`WANDB_RESUME=never`，防止历史曲线互相覆盖。恢复步数保存到 manifest；横轴使用实际 `opd/step`，W&B 内部 `_step` 是事件序号，因此同一训练步的训练、验证、checkpoint 和失败事件均可保留。两个训练项目为 `codemem-add4-opd-qwen35`、`codemem-add4-opd-qwen36`。

启动时必须读取120秒内且两台主机探针成功的资源快照，否则统一向所有 rank 传播启动失败。运行中快照过期会上报 `resources/snapshot_fresh=0`，不冒充实时数据，也不因一次资源采集失败中断训练。师生身份和模型评分预检仍按原协议独立执行；GPU 可读不代表教师服务就绪。

## 指标口径

| 指标 | 含义 |
| --- | --- |
| `train/loss`, `opd/reverse_kl`, `opd/entropy` | 总目标为 KL − entropy_coef × entropy；分别看三条曲线，不以总 loss 下降代替效果提升 |
| `train/grad_norm`, `train/lr`, `train/grad_clip_triggered` | 裁剪前梯度范数、学习率、是否超过裁剪阈值 |
| `train/update_skipped`, `train/skipped_updates_total`, `errors/nonfinite_grad_total` | 当前 BF16/FSDP 路径中非有限梯度导致的跳过更新；计数按本次 attempt 累积 |
| `opd/student_mass`, `opd/teacher_mass` | top-k 模式为所选 K 的质量；full 模式为全有效词表质量，约1 |
| `opd/student_tail_mass`, `opd/teacher_tail_mass`, `opd/teacher_tail_floor_rate` | 仅 K<V；学生用1−top-k质量，教师为平滑后的尾部；下限触发比例按实际响应 token 加权。学生相减存在 FP32 误差，不用于 loss |
| `generation/response_tokens_{mean,p95,max}` | 实际生成长度，包括 EOS；`generation/response_tokens_total` 是成功批次的响应 token 总数 |
| `generation/eos_rate`, `generation/truncated_rate`, `generation/format_valid_rate` | EOS/长度截断按已返回生成请求计；格式要求 JSON 对象仅有 facts 字段，内容为字符串列表，允许空列表。格式统计不是事实正确率；HTTP失败不进入生成分母 |
| `time/weight_export_s`, `time/student_reload_s`, `weights/export_bytes` | 权重同步总耗时减热更新耗时为导出/元数据开销；导出字节数 |
| `time/rollout_service_s`, `time/teacher_service_s` | 并发请求的累计服务秒数，后者包含解析和缓存写盘，不能直接相加当墙钟时间 |
| `time/generate_score_wall_s`, `time/prepare_s`, `time/train_s` | 生成评分墙钟、准备阶段、训练阶段时间；这些含有嵌套阶段，不相加 |
| `time/step_to_metrics_s`, `perf/response_tokens_per_s` | 当前步从准备开始到训练指标的时间与响应吞吐，包含同步/生成/评分，不含随后验证与checkpoint |
| `time/checkpoint_s`, `time/step_through_checkpoint_s`, `time/previous_step_with_validation_checkpoint_s` | 保存时间、当前步到保存完成、上一步包含验证/保存的完整间隔（后者在下一步上报） |
| `http/{role}_{operation}_*` | 请求数、错误数、收到的JSON字节、累计耗时和延迟均值/P95/max；延迟样本每类每步上限10000 |
| `services/identity_checks_passed`, `errors/*` | 师生身份检查成功次数、阶段错误、致命错误类型；异常文本和原始响应不上 W&B |
| `perf/all_rank_peak_{allocated,reserved}_gb` | 每步重置后的训练卡峰值，跨 rank 取最大；原 VERL 内存/MFU 指标继续保留，MFU不含远程教师和rollout成本 |
| `training_host/gpu*/…`, `teacher_host/gpu*/…` | 每台主机全部 GPU 的利用率、显存、温度；可能包含别人的任务，不能当成本 run 的独占用量 |
| `disk/{sync_dir,audit_dir,full_score_dir}_free_gb`, `cache/written_bytes` | 相应文件系统余量、当步实际评分缓存文件字节数 |
| `val/loss` | 固定标签 teacher-forcing CE，逐样本均值；不是验证集 OPD KL |

响应 JSON 格式检查目前只记录、不筛掉样本；非 EOS 完成仍依协议报错，但失败前先累计 EOS/截断统计。HTTP错误、评分解析错误和阶段失败均保留；失败事件写入本地并尝试上传，不提供“跳过坏批次”机制。SIGKILL、宿主机掉电等不能保证末次事件上传。

每个 attempt 的 `$ADD_RUN_DIR/monitoring/<run_id>/metrics.jsonl`、`manifest.json` 永久保留。W&B 只上传标量与配置，原始 prompt、response IDs、teacher 分数仍在原审计目录。

## 固定验证与下游结果

默认复用 `/data/zengjiajie/projects/codemem/swe-cc-trajectories-4000/add4/sft/validation-188.parquet` 的全部398条，可用 `ADD_VAL_FILES` 指定另一个已准备的固定集合。启动前、每50步和末步验证；模板仍只有1个训练步，正式步数仍需实验参数确定。

398不能整除8。OPD验证采用每个rank相同顺序遍历全部398条、每rank每批1条，DP均值之后每条只占一个样本权重；不丢弃、不人为复制加权某几条，也保证 FSDP collective 次数一致。代价是验证推理重复8份。原独立 SFT 的验证策略不改。CPU测试验证完整覆盖与归约接口，真实多卡性能尚未验收。

每次保存成功后，自动写 `monitoring/<run_id>/eval-queue/step-NNNNNNNN.json`，含 checkpoint 路径、训练run、project和step。仅产生待评测记录，不自动消耗模型API或调度188题任务。评测完成后整理为：

```json
{"checkpoint_step":50,"checkpoint":"/实际目录/checkpoints/global_step_50",
 "datasets":{"CodeMemPro":{"tasks":62,"attempts":248,"successes":120,"solved_tasks":45}}}
```

以上数字只是格式示例。`python log_opd_evaluation.py --request <待评测记录> --results <实际结果JSON> --dry-run` 先校验；去掉 `--dry-run` 上传。接口拒绝 checkpoint 路径/步数不一致、计数不一致和每题不足4次的统计。数据集各自和合并 `avg_at_4/pass_at_4` 均为0–1比例。使用独立 evaluation run，并以训练run为group、checkpoint_step为横轴，避免迟到评测改写训练历史；收据写到结果JSON旁。

## 验收入口

`check_opd_monitoring.py --output <本轮smoke目录>` 在 CPU 上做两次微型 top-k 梯度更新、真实上传 W&B 并通过 API 回读两步和同一步追加事件；不启动正式训练或教师。`--offline` 仅供离线调试，收据会明确未验证上传。

本次验收与局限见 `/data/zengjiajie/projects/verl/add4-opd/tests/monitoring-20261018/README.md`。
