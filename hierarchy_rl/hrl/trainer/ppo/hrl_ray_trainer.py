from typing import Optional, List, Dict, Tuple
import ray
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from omegaconf import OmegaConf
import numpy as np
from collections import defaultdict
import uuid
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage
from verl.trainer.ppo.utils import Role, WorkerType
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    reduce_metrics,
    compute_timing_metrics,
    agg_loss,
)
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProto

from verl.hierarchy_rl.hrl.agents import Planner, Executor
from verl.hierarchy_rl.hrl.utils.logging_utils.stdout import PrettyPrinter
from verl.hierarchy_rl.hrl.trainer.ppo.reason_rl_ray_trainer import ReasonRLRayPPOTrainer


def create_default_dict():
    return defaultdict(int)

def compute_data_metrics(batch, use_critic=True, tokenizer=None):
    """
    这个函数用来计算一些指标，返回值是一个metrics，参考别人的实现是
    metrics = {
        # --- 分数 (Score) 相关指标 ---
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        
        # --- 奖励 (Reward) 相关指标 ---
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        
        # --- 优势 (Advantage) 相关指标 ---
        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
        'critic/advantages/max': torch.max(valid_adv).detach().item(),
        'critic/advantages/min': torch.min(valid_adv).detach().item(),
        
        # --- 回报 (Return) 相关指标 ---
        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
        'critic/returns/max': torch.max(valid_returns).detach().item(),
        'critic/returns/min': torch.min(valid_returns).detach().item(),
        
        # --- 如果使用 Critic，添加其相关指标 ---
        **({
            # Critic 预测的价值 (Value)
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # 价值函数可解释方差 (Explained Variance)，衡量 Critic 预测准确度的重要指标
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # --- 响应长度 (Response Length) 相关指标 ---
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # 响应长度达到最大值的比例，用于监控是否频繁截断
        'response_length/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        "response_length/reflect_ratio": reflect_ratio,
        "response_length/correct_reflect_ratio": correct_ratio,
        **length_metrics, # 合并正确/不正确响应的平均长度
        
        # --- 提示长度 (Prompt Length) 相关指标 ---
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
        # 提示长度达到最大值的比例
        'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    """
    pass

@dataclass
class ApkInfo:
    """一个数据类，用于清晰地存储每个 APK 的相关信息。"""
    name: str               # APK 名字 (例如："Chrome")
    path: str               # APK 绝对路径 (例如："/path/to/chrome.apk")
    package_name: str       # APK 包名 (例如："com.android.chrome")
    android_manifest: str   # APK activity心思AndroidManifest.xml

@ray.remote
class DatasetManager:
    """
    一个 Ray Actor，用于在分布式环境中管理和分发包含详细信息的 APK 数据集。

    它被设计成一个有状态的服务，能够记住上一次分发到了哪个 APK，
    并在下次请求时提供列表中的下一个 APK，实现了数据集的循环遍历。
    """
    def __init__(self, apk_infos: List[ApkInfo]):
        """
        初始化数据集管理器。

        Args:
            apk_infos (List[ApkInfo]): 包含所有待测试 APK 信息对象的列表。
                                       每个对象应包含名字、绝对路径和包名。
        """
        if not apk_infos:
            raise ValueError("【DatasetManager Error】: APK 信息列表 `apk_infos` 不能为空！")

        # 存储所有 APK 的信息对象列表
        self.apk_infos = apk_infos
        # 初始化一个指针，用于追踪当前应该分发哪个 APK
        self.current_idx = 0
        
        PrettyPrinter.status(f"【DatasetManager】: 初始化成功，共加载 {len(self.apk_infos)} 个 APK。")

    def get_next_app(self) -> ApkInfo:
        """
        获取下一个要测试的 APK 信息对象。

        此方法实现了循环队列的逻辑。当遍历完所有 APK 后，会自动从头开始。

        Returns:
            ApkInfo: 列表中的下一个 APK 信息对象。
        """
        # 1. 根据当前索引获取 APK 信息对象
        app_info = self.apk_infos[self.current_idx]
        
        # 2. 更新索引，使其指向下一个 APK
        self.current_idx = (self.current_idx + 1) % len(self.apk_infos)
        
        # 3. 更新日志，打印更有价值的信息
        PrettyPrinter.status(f"【DatasetManager】: 分配应用 -> {app_info.name} (包名: {app_info.package_name})")
        
        # 4. 返回获取到的完整对象
        return app_info

class HRLRayTrainer(ReasonRLRayPPOTrainer):
    """分层强化学习训练器：包含外层规划器循环与内层执行器循环。

    - 外层：生成/选择测试意图（奖励：缺陷、新增覆盖、去重惩罚）
    - 内层：围绕意图执行GUI操作（奖励：路径对齐、缺陷触发、效率与无效操作惩罚）
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager,
        **kwargs,
    ):
        # 1. 调用父类构造函数，完成 VERL 工作组（如 self.actor_rollout_wg）的初始化
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            **kwargs,
        )

        # 2. 从配置中读取 HRL 特有的超参数
        self.planner_episodes_per_update = self.config.hrl.get("planner_episodes_per_update", 5)
        self.max_executor_steps = self.config.hrl.get("max_executor_steps", 20)

        # 3. 暂不在 __init__ 中初始化 Planner/Executor（actor_rollout_wg 尚未创建）
        self.planner: Optional[Planner] = None
        self.executor: Optional[Executor] = None
        
        # 假设 DatasetManager 存在于您的项目中，并按预期工作
        self.dataset_manager = DatasetManager.remote(apk_paths=self.config.hrl.apk_paths)

    def cleanup(self):
        """Clean up other resources"""
        gc.collect()
        torch.cuda.empty_cache()

    def _create_update_dataloader(self, experiences: List[Dict]) -> DataLoader:
        """
        根据收集到的经验列表创建一个用于 PPO 更新的 PyTorch DataLoader。
        """
        dataset = RLDataset(experiences)
        sampler = RandomSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.config.trainer.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True  # 确保所有批次大小一致，简化处理
        )

    def _compute_token_level_rewards(self, reward: float, response_ids: torch.Tensor) -> torch.Tensor:
        """
        将一个标量奖励转换为 token 级别的奖励张量。
        常见的做法是将奖励值赋给回应序列的最后一个非填充 token。
        """
        token_rewards = torch.zeros(len(response_ids), dtype=torch.float32)
        if len(response_ids) > 0:
            token_rewards[-1] = reward
        return token_rewards

    def _compute_outer_reward(self, episode_info: dict) -> float:
        """计算外层循环（Planner）的稀疏奖励"""
        reward = 0.0
        if episode_info.get("defect_found"):
            reward += 100.0  # 发现缺陷给予高额奖励
        reward += 10.0 * float(episode_info.get("new_pages", 0))  # 每覆盖一个新页面给予奖励
        reward -= 5.0 * float(episode_info.get("duplicate_intent", 0))  # 如果意图重复，给予惩罚
        return reward

    def _compute_inner_step_reward(self, step_info: dict) -> float:
        """计算内层循环（Executor）每一步的稠密奖励"""
        reward = 0.0
        if step_info.get("on_path", False):
            reward += 1.0  # 如果操作符合意图路径，给予奖励
        if step_info.get("defect_triggered", False):
            reward += 50.0  # 如果操作触发了缺陷，给予高额奖励
        reward -= 0.1  # 效率惩罚（每多走一步都有轻微惩罚）
        if step_info.get("invalid", False) or not step_info.get("action_ok", True):
            reward -= 1.0  # 无效操作或失败操作给予惩罚
        return reward

    def _planner_rollout(self, current_ui_state, app_context: dict) -> Tuple[Plan, torch.Tensor, torch.Tensor]:
        """使用 self.planner 生成测试意图，并返回 Plan 对象、prompt ID 和 response ID。"""
        proposed_plans: List[Plan] = self.planner.propose_intents(app_context)
        chosen_plan = proposed_plans[0] if proposed_plans else \
            Plan(intent="在当前页面进行探索性点击", reasoning="Planner 未能生成有效意图，执行默认探索策略。", meta={"source": "fallback"})
        
        prompt_text = self.planner._build_prompt(app_context)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.config.data.max_prompt_length).input_ids.squeeze(0)
        
        # Planner 的 "action" 是它生成的意图文本
        intent_ids = self.tokenizer(chosen_plan.intent, return_tensors="pt", truncation=True, max_length=self.config.data.max_response_length).input_ids.squeeze(0)

        logger.info(f"【Planner】: 已选择意图 -> {chosen_plan.intent}")
        return chosen_plan, prompt_ids, intent_ids

    def _executor_rollout(self, ui_state: Tuple[List[Dict], int, int], plan: Plan) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """使用 Actor 模型生成具体操作，返回 action 文本、prompt ID 和 action ID。"""
        perception_infos, _, _ = ui_state
        element_descriptions = [f"{i+1}. [{e.get('type','elem')}] \"{e.get('text','')}\" at {e.get('coordinates', e.get('bbox'))}" for i, e in enumerate(perception_infos)]
        elements_str = "\n".join(element_descriptions) or "屏幕上未识别到任何元素。"
        
        prompt_text = f"**High-level Intent:**\n{plan.intent}\n\n**Current Screen Elements:**\n{elements_str}\n\n**Your Task:**\nBased on the intent and the current screen elements, decide the single next operation to perform.\n\n**Next Operation:**".strip()

        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.config.data.max_prompt_length)
        prompt_ids = inputs.input_ids.squeeze(0)
        executor_batch = DataProto.from_single_dict({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask})

        output_batch = self.actor_rollout_wg.generate_sequences(executor_batch)
        action_ids = output_batch.batch["responses"][0]
        action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()
        
        return action_text, prompt_ids, action_ids

    def _perform_ppo_update(self, experiences: list, agent_type: str):
        """使用收集到的经验数据执行一次完整的 PPO 更新。"""
        if not experiences:
            PrettyPrinter.status(f"【PPO Update】: {agent_type} 经验缓冲区为空，跳过本次更新。")
            return

        dataloader = self._create_update_dataloader(experiences)
        
        for epoch in range(self.config.trainer.ppo_epochs):
            for batch in dataloader:
                # 核心 PPO 计算逻辑
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                
                if self.use_critic:
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

                batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, **self.config.algorithm)

                # 更新网络参数
                if self.use_critic:
                    self.critic_wg.update_critic(batch)
                
                if self.config.trainer.critic_warmup <= self.global_steps:
                    self.actor_rollout_wg.update_actor(batch)
        
        PrettyPrinter.status("UPDATE", f"已完成对 '{agent_type}' 经验的 {self.config.trainer.ppo_epochs} 个周期的参数更新。", "info")
    
    def fit(self):
        """
        HRL 训练主循环，实现了外层 Planner 和内层 Executor 的嵌套逻辑。
        这是整个分层强化学习训练的核心驱动函数。
        """
        # 步骤 0: 初始化底层资源
        # 调用父类的 init_workers 方法，初始化所有 Ray Actor 工作组 (Worker Groups)，
        # 例如用于 Rollout 的 Actor 模型、用于价值评估的 Critic 模型等。
        # 这是执行任何分布式计算前必需的一步。
        super().init_workers()

        # 步骤 1: 初始化 Planner 和 Executor
        # 在 Worker 初始化完成后，此时 self.actor_rollout_wg 等句柄已经就绪，
        # 可以安全地将它们作为依赖注入到 Planner 和 Executor 中。
        PrettyPrinter.section_header("【HRL Trainer】: Initializing Planner and Executor...")
        if self.planner is None:
            # 实例化高层规划器 Planner。
            # **self.config.hrl.planner 会将配置文件中 hrl.planner 下的所有参数
            # (如 system_prompt 等) 作为关键字参数传递给 Planner 的构造函数。
            self.planner = Planner(
                actor_wg=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                **self.config.hrl.planner,
            )
        if self.executor is None:
            # 实例化底层执行器 Executor。
            # **self.config.hrl.executor 会将 hrl.executor 下的所有参数
            # (如 platform, use_ocr 等) 传递给 Executor 的构造函数。
            self.executor = Executor(
                actor_wg=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                saving_folder=self.config.data.save_folder,
                **self.config.hrl.executor,
            )
        
        # 初始化全局步数计数器，并尝试从检查点加载模型权重和训练状态。
        self.global_steps = 0
        self._load_checkpoint()

        # --- 主训练循环开始 ---
        # 每个 "global_step" 代表一次完整的数据收集和模型更新周期。
        while self.global_steps < self.config.trainer.total_training_steps:
            PrettyPrinter.section_header(f"HRL Global Step: {self.global_steps}")
            
            # 为当前全局步骤初始化空的经验缓冲区，分别用于存储两层智能体的经验。
            planner_experiences, executor_experiences = [], []

            # --- 外层循环 (Planner Rollout) ---
            # 目标：收集一批高层决策的经验数据。
            # 循环次数由配置中的 planner_episodes_per_update 决定。
            for episode_idx in range(self.planner_episodes_per_update):
                print(f"\n--- Planner Episode {episode_idx + 1}/{self.planner_episodes_per_update} ---")
                
                # a. 获取新任务并重置环境
                # 从远程的数据集管理器获取下一个要测试的 app 路径。
                app_info = ray.get(self.dataset_manager.get_next_app.remote())
                # 使用该 app 路径重置 Executor 所控制的环境（例如，启动模拟器并安装/打开应用）。
                current_ui_state = self.executor.reset(app_info=app_info, global_step=self.global_steps, episode_idx=episode_idx)
                
                # 获取 Planner 做决策所需的宏观上下文信息（如覆盖率、历史记录等）。
                app_context = self.executor.get_context()
                # b. Planner 生成高层意图 (Action)
                # 这是 Planner 的 "Rollout" 步骤，它根据当前上下文生成一个测试意图。
                intent, planner_prompt_ids, intent_ids = self._planner_rollout(current_ui_state, app_context)
                
                # c. 初始化内层循环的状态
                done, inner_step = False, 0
                episode_info = defaultdict(float)  # 用于聚合内层循环的关键信息，以计算外层奖励。

                # --- 内层循环 (Executor Rollout) ---
                # 目标：执行 Planner 给出的高层意图，并收集底层的操作经验。
                # 循环直到子任务完成 (done=True) 或达到最大步数限制。
                while not done and inner_step < self.max_executor_steps:

                    # 给出步数信息，便于日志记录和调试。
                    self.executor.set_step_num(global_steps, episode_idx, inner_step)
                    # a. Executor 生成具体操作 (Action)
                    # 这是 Executor 的 "Rollout" 步骤，它根据当前 UI 状态和高层意图生成一个具体的 GUI 操作。
                    action_text, executor_prompt_ids, action_ids = self._executor_rollout(current_ui_state, intent)
                    
                    # b. 在环境中执行动作
                    # 由于 Executor 的 step 方法是异步的 (async def)，在同步的 fit 循环中需要用 asyncio.run 来调用。
                    step_result = asyncio.run(self.executor.step(action_text))
                    # 从返回的 StepResult 对象中解包出新状态、完成标志和辅助信息。
                    next_ui_state, done, step_info = step_result.next_state, step_result.done, step_result.info
                    
                    # c. 计算并存储 Executor 的经验
                    # 计算这一步操作获得的稠密奖励 (dense reward)。
                    inner_reward = self._compute_inner_step_reward(step_info)
                    # 将标量奖励转换为 PPO 算法所需的 token 级奖励向量。
                    token_level_rewards = self._compute_token_level_rewards(inner_reward, action_ids)
                    
                    # 将这次的经验 (prompt, response, reward) 存入 Executor 的经验缓冲区。
                    executor_experiences.append({
                        "prompts": executor_prompt_ids,
                        "responses": action_ids,
                        "token_level_rewards": token_level_rewards,
                    })

                    # d. 更新状态
                    current_ui_state = next_ui_state
                    inner_step += 1
                    # 聚合整个 episode 的信息，用于后续计算 Planner 的奖励。
                    episode_info["new_pages"] += step_info.get("new_pages", 0)
                    if step_info.get("defect_triggered"):
                        episode_info["defect_found"] = True

                # --- 外层循环的收尾 ---
                # a. 计算 Planner 的奖励
                # 在内层循环结束后，根据整个子任务的执行结果（存储在 episode_info 中），计算 Planner 的稀疏奖励 (sparse reward)。
                outer_reward = self._compute_outer_reward(episode_info)
                print(f"【Planner】: Episode 结束. 获得外层奖励: {outer_reward}")
                
                # b. 存储 Planner 的经验
                # 同样将标量奖励转换为 token 级奖励向量。
                token_level_rewards = self._compute_token_level_rewards(outer_reward, intent_ids)
                
                # 将这次高层决策的经验 (prompt, response, reward) 存入 Planner 的经验缓冲区。
                planner_experiences.append({
                    "prompts": planner_prompt_ids,
                    "responses": intent_ids,
                    "token_level_rewards": token_level_rewards,
                })
            
            # --- 步骤 2: 执行 PPO 更新 ---
            # 在收集了足够多的两层经验后，开始更新模型参数。
            # 注意：Planner 和 Executor 使用的是同一个 Actor 模型，但通过不同的经验数据进行训练，
            # 从而让模型学会在不同情境下（即面对不同类型的 Prompt 时）扮演不同的角色。
            
            # a. 使用 Executor 的经验更新模型，教会模型如何具体地执行操作。
            self._perform_ppo_update(executor_experiences, "Executor")
            # b. 使用 Planner 的经验更新模型，教会模型如何进行高层的战略规划。
            self._perform_ppo_update(planner_experiences, "Planner")

            # --- 步骤 3: 清理和迭代 ---
            # 全局步数加一。
            self.global_steps += 1
            # 根据配置的频率，保存模型检查点。
            if self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()
            
            # 调用清理函数，释放 GPU 显存等资源，为下一个全局步骤做准备。
            self.cleanup()

    def _validate(self):
        """
        PPO 的验证循环。
        此函数在验证数据集上评估当前模型的性能。
        它与训练循环的主要区别在于：只进行前向传播和评估，不计算梯度也不更新模型参数。
        """
        # --- 1. 初始化用于收集结果的列表 ---
        reward_tensor_lst, data_source_lst = [], []
        sample_inputs, sample_outputs, sample_scores = [], [], []
        all_eval_metrics = defaultdict(list)

        # --- 2. 遍历验证数据加载器 ---
        # self.val_dataloader 通常会一次性加载整个验证集
        # TODO: HRL的验证逻辑可能更复杂，需要模拟Planner和Executor的交互，
        # 而不是简单地遍历静态数据集。此部分需要根据您的验证需求进行定制。
        PrettyPrinter.status("VALIDATE", "验证循环开始...", "info")
        # for batch in self.val_dataloader:
        #     ... (执行验证逻辑) ...
        PrettyPrinter.status("VALIDATE", "验证循环结束。", "info")
