#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc    : HRL 框架中的高层规划器 (Planner)，负责生成测试意图。
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT_TEMPLATE
from verl.modeling.ray_worker_group import RayWorkerGroup
from verl.proto import DataProto
from verl.hierarchy_rl.hrl.utils.logging_utils import PrettyPrinter

@dataclass
class Plan:
    """
    一个数据类，用于表示一个高层测试计划或意图。
    """
    intent: str  # 测试意图的文本描述
    reasoning: str  # LLM 生成此意图时的思考过程或理由
    meta: Dict[str, Any] = field(default_factory=dict)  # 附加的元信息（如来源、评分等）


# ==============================================================================
# 2. Planner 类实现
# ==============================================================================


class Planner:
    """
    Planner 类是 HRL 框架中的高层决策者。
    它负责分析应用的宏观状态，并提出一系列高价值的测试意图（Plan），
    然后由底层的 Executor 去尝试执行。
    """
    def __init__(self, actor_wg: RayWorkerGroup, tokenizer: Any, system_prompt: str = "") -> None:
        """
        初始化 Planner。

        Args:
            actor_wg (RayWorkerGroup): 指向 Actor 模型分布式工作组的句柄。这是被注入的依赖。
            tokenizer (Any): 用于文本编码和解码的分词器。
            system_prompt (str): 自定义的系统提示。如果为空，则使用默认提示。
        """
        # --- 核心依赖注入 (这部分是正确的) ---
        self.actor_wg = actor_wg
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or PLANNER_SYSTEM_PROMPT

        # --- 状态变量初始化 (修正部分) ---
        # 对于需要列表的属性，直接在 __init__ 中赋值为空列表 []
        # 这是安全且正确的做法，每个 Planner 实例都会有自己独立的列表。
        self.thought: str = ""                         # 当前思考内容
        self.thought_history: List[str] = []           # 历史思考记录列表
        self.summary_history: List[str] = []           # 历史操作总结列表
        self.task_list: str = ""                       # 任务列表
        self.completed_requirements: str = ""          # 已完成的需求
        self.memory: List[str] = []                    # 重要内容记忆列表
        self.error_flag: bool = False                  # 错误标志
        self.perception_infos: List[Dict] = []         # 当前感知信息列表
        self.width: int = 0                            # 屏幕宽度
        self.height: int = 0                           # 屏幕高度

    def reset(self):
        self.thought = ""
        self.thought_history = []
        self.summary_history = []
        self.task_list = ""
        self.completed_requirements = ""
        self.memory = []
        self.error_flag = False
        self.perception_infos = []
        self.width = 0
        self.height = 0

    def _build_prompt(self, app_context: Dict[str, Any]) -> str:
        """根据当前的应用上下文，构建用于请求 LLM 生成意图的详细 Prompt。"""
        main_objective = app_context.get("main_objective", "Thoroughly test this application's functionality and stability.")
        app_desc = app_context.get("app_desc", "（no detail description）")
        coverage = "\n- ".join(app_context.get("coverage", ["None"])) or "None"
        found_defects = "\n- ".join(app_context.get("found_defects", ["None"])) or "None"
        plan_history = "\n- ".join(app_context.get("plan_history", ["None"])) or "None"

        prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
            main_objective=main_objective,
            app_desc=app_desc,
            coverage=coverage,
            found_defects=found_defects,
            plan_history=plan_history
        )
        return PLANNER_SYSTEM_PROMPT + "\n" + prompt

    def _parse_response(self, response_text: str) -> List[Plan]:
        """解析 LLM 返回的 JSON 格式的字符串，并将其转换为 Plan 对象列表。"""
        plans = []
        try:
            cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned_text)
            for item in data.get("plans", []):
                if "intent" in item and "reasoning" in item:
                    plans.append(
                        Plan(
                            intent=item["intent"],
                            reasoning=item["reasoning"],
                            meta={"source": "llm-generated"}
                        )
                    )
        except (json.JSONDecodeError, AttributeError) as e:
            PrettyPrinter.status(f"Failed to parse LLM response into JSON: {e}\nResponse was:\n{response_text}")
            return [Plan(intent="Continue exploring functional areas of the application that have not yet been covered", reasoning="Failed to parse the LLM response; executing the default exploration strategy.", meta={"source": "fallback"})]
        return plans

    def propose_intents(self, app_context: Dict[str, Any]) -> List[Plan]:
        """
        使用注入的 Actor 工作组生成一系列测试意图。

        Args:
            app_context (Dict[str, Any]): 描述应用当前测试状态的上下文信息。

        Returns:
            List[Plan]: 一个由 Actor 生成的、包含多个 Plan 对象的列表。
        """
        # 1. 根据上下文构建 Prompt
        prompt = self._build_prompt(app_context)
        PrettyPrinter.status(f"Generated Planner Prompt:\n{prompt}")

        # 2. 使用 VERL 框架的标准流程调用 Actor 模型
        inputs = self.tokenizer(prompt, return_tensors="pt")
        batch_dict = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}
        planner_batch = DataProto.from_single_dict(batch_dict)
        
        output_batch = self.actor_wg.generate_sequences(planner_batch)
        
        response_ids = output_batch.batch["responses"][0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        PrettyPrinter.status(f"Received Planner Response:\n{response_text}")

        # 3. 解析响应并返回 Plan 列表
        return self._parse_response(response_text)