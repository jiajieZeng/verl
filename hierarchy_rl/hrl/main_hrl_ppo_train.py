# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
注意：我们没有将 main 与 ray_trainer 合并，因为 ray_trainer 被其他 main 脚本复用。
这个文件是整个分布式 PPO 训练任务的入口点。
"""
import ray
import hydra
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

# 导入我们之前已经详细分析过的、包含核心训练逻辑的训练器类
from .trainer.ppo.hrl_ray_trainer import HRLRayTrainer
# 导入用于代码生成任务的自定义奖励管理器
from .reward.reward_manager import HRLRewardManager


# 使用 Hydra 装饰器。这使得我们可以通过命令行和 YAML 配置文件来灵活地管理所有超参数。
# config_path 指向存放配置文件的目录，config_name 是默认的主配置文件名。
@hydra.main(config_path='configs', config_name='azr_ppo_trainer', version_base=None)
def main(config):
    """Hydra 将解析 YAML 配置并将其作为一个 `config` 对象传入此 main 函数。"""
    run_ppo(config)


def run_ppo(config) -> None:
    """运行 PPO 训练流程的主函数。"""
    # 检查 Ray 是否已经初始化，防止重复初始化。
    if not ray.is_initialized():
        # 初始化 Ray。这是启动所有分布式功能的第一步。
        ray.init(
            # runtime_env 用于为 Ray 的所有 worker 设置环境变量。
            # 这里设置了一些常用的环境变量，如启用分词器并行、设置 NCCL 调试级别等。
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            # 从配置中读取 Ray 集群可以使用的 CPU 核心数。
            num_cpus=config.ray_init.num_cpus,
        )

    # 【核心架构】：创建 TaskRunner 作为一个远程 Actor。
    # 这意味着 TaskRunner 的实例将在 Ray 集群的一个独立工作进程中运行，而不是在当前这个驱动脚本的进程中。
    # 这样做可以保持驱动脚本的轻量，并将所有重型任务（如模型加载、训练循环）隔离到专用的工作进程中。
    # 如果配置中启用了性能剖析（profiling）
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        # 为 TaskRunner Actor 配置 NVIDIA Nsight 分析工具的运行时环境
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        # 创建一个标准的 TaskRunner Actor
        runner = TaskRunner.remote()
    
    # 远程调用 TaskRunner 实例的 `run` 方法，并将完整的配置对象 `config` 传递给它。
    # `ray.get()` 会阻塞当前进程，直到远程的 `run` 方法执行完成。
    ray.get(runner.run.remote(config))

    # [可选] 如果配置中指定了 timeline 文件路径，则生成 Ray 的时间线追踪文件。
    # 这个文件可以用来分析 Ray 任务的调度和执行情况，对于性能调优非常有用。
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


# 将 TaskRunner 定义为一个 Ray Actor。`num_cpus=1` 确保这个 Actor 独占一个 CPU 核心。
# 注释 "please make sure main_task is not scheduled on head" 建议不要将这个主任务调度在 Ray 的头节点上，以避免头节点过载。
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        """
        这个方法是实际的“总指挥”，它在远程的 Actor 进程中执行。
        它负责所有的初始化工作，并最终启动训练循环。
        """
        # 打印完整的配置信息，`resolve=True` 会解析配置中的所有变量和插值表达式。
        pprint(OmegaConf.to_container(config, resolve=True))
        # 立即解析配置中的所有延迟计算的变量。
        OmegaConf.resolve(config)

        # 如果配置中开启了 debug 模式，则启动远程调试服务器。
        if config.trainer.debug:
            import debugpy
            # 监听指定的 IP 和端口。
            debugpy.listen(("0.0.0.0", config.trainer.debug_port))
            print(f"调试器正在监听端口 {config.trainer.debug_port}")
            # 等待调试器客户端（如 VS Code）连接。
            debugpy.wait_for_client()
            print("调试器已连接！")

        # --- 动态配置调整 ---
        # 自动计算 PPO 的 mini-batch-size，使其等于一个完整 PPO 步骤所需的所有数据。
        # 公式：(每个任务的批次大小) * (问题类型数量) * (如果是 Proposer+Solver 模式则乘以2，否则乘以1)
        config.actor_rollout_ref.actor.ppo_mini_batch_size = config.data.train_batch_size * len(config.azr.problem_types) * (2 if config.azr.train_propose else 1)
        pprint(f"自动设置 ppo_mini_batch_size: {config.actor_rollout_ref.actor.ppo_mini_batch_size}")
        # 自动计算每个训练步骤需要准备的数据量。
        config.azr.data_selection_strategy.data_len = config.data.train_batch_size * config.azr.data_selection_strategy.update_iteration
        pprint(f"自动设置 data_len: {config.azr.data_selection_strategy.data_len}")

        # 动态构建一个唯一的本地目录，用于存放本次运行的日志、模型和生成的数据，便于实验管理。
        config.trainer.default_local_dir = (Path(config.trainer.default_local_dir) / config.data.train_files.split('/')[-1].split('.')[0] / config.actor_rollout_ref.model.path.split('/')[-1] / config.reward_fn.extraction_type).as_posix()

        # 配置验证：如果允许生成复合函数，则必须禁止“拒绝多函数”的规则。
        assert not (not config.azr.reward.generation_reward_config.reject_multiple_functions and config.azr.data_selection_strategy.composite_function_n_min > 0), "如果 reject_multiple_functions 为 False, composite_function_n_min 必须为 0"

        # --- 资源和模型加载 ---
        # 从 HDFS（或任何远程文件系统）下载模型检查点到本地，并返回本地路径。
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # 实例化分词器（Tokenizer）和处理器（Processor，用于多模态模型）。
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # 如果配置了使用预训练分词器，则设置一个简单的聊天模板。
        if config.actor_rollout_ref.model.pretrained_tokenizer:
            tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"
        
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 对 vllm 版本进行验证，确保支持 LoRA。
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("vllm 0.7.3 之前的版本不支持 PPO LoRA")

        # --- 【核心：根据分布式策略选择 Worker】 ---
        # 这是 VERL 框架灵活性的体现。根据配置文件中指定的 `strategy`，动态导入和选择不同的 Worker 实现。
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # 如果使用 FSDP (Fully Sharded Data Parallel)，则导入 FSDP 版本的 Worker。
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # 如果使用 Megatron-LM，则导入 Megatron 版本的 Worker。
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        # --- 【核心：角色与资源映射】 ---
        # 将 PPO 中的抽象角色（如 Actor, Critic）映射到上面选择的具体 Worker 类。
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # 定义可用的 GPU 资源池。这里定义了一个名为 "global_pool" 的资源池。
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 将每个角色映射到指定的资源池。
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # --- 初始化奖励模型和参考策略（如果需要）---
        # 如果启用了奖励模型（RM）
        # ZZZ NOTE： 我们自己写奖励函数，所以这里不运行的，reward_model.enable是False
        if config.reward_model.enable:
            # 根据策略选择并导入对应的 RewardModelWorker
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            # ... (其他策略的 Worker)
            else:
                raise NotImplementedError
            # 将 RewardModel 角色添加到角色-Worker 和角色-资源映射中
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 如果使用了 KL 散度（作为奖励或损失），则需要一个固定的参考策略（Reference Policy）
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # --- 初始化自定义组件 ---
        # 初始化训练阶段的奖励函数管理器
        # ZZZ FIXME: 这里我们的reward manager还没有写，要自己写，参数还没确定
        reward_fn = HRLRewardManager(
            tokenizer=tokenizer,
            # ... (从 config 中读取大量奖励相关的配置参数) ...
            **config.reward_fn,
            **config.azr.reward,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            valid_program_filter=config.azr.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
        )
        # 初始化验证阶段的奖励函数管理器
        val_reward_fn = HRLRewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            split='test',
            # ... (与训练奖励函数类似的配置) ...
             **config.reward_fn,
            **config.azr.reward,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            valid_program_filter=config.azr.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
        )

        # 实例化资源管理器
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 动态生成 WandB 标签，用于更好地组织和筛选实验
        wandb_tags = [
            'codeio', config.azr.pred_data_mix_strategy, 'executor-' + config.azr.executor,
            config.azr.data_selection_strategy.valid_program_filter, config.azr.gen_data_probabilities_strategy,
        ]
        wandb_tags.extend(config.azr.problem_types)
        if config.trainer.wandb_tags is not None:
            config.trainer.wandb_tags = wandb_tags + config.trainer.wandb_tags.split(',')
        else:
            config.trainer.wandb_tags = wandb_tags

        # --- 【最终组合】实例化训练器 ---
        # 将所有准备好的组件（配置、分词器、Worker映射、资源管理器、奖励函数等）注入到最终的训练器实例中。
        # ZZZ FIXME： 这里开始训练，我们的还没写完，还要确定
        trainer = HRLRayTrainer(
            past_epoch_window=config.azr.past_epoch_window,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        # --- 启动训练 ---
        # 初始化所有远程 Ray Worker 进程
        trainer.init_workers()
        # 调用 `fit` 方法，启动核心的 PPO 训练循环
        trainer.fit()


if __name__ == '__main__':
    try:
        # 程序的入口点
        main()
    except KeyboardInterrupt:
        # 优雅地处理用户通过 Ctrl+C 中断程序的情况
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        # 捕获所有其他异常，打印完整的堆栈信息，并以非零状态码退出，表示程序异常终止。
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)