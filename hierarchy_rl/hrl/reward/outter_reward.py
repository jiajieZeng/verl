def _compute_outer_reward(episode_info: dict) -> float:
    """计算外层循环（Planner）的稀疏奖励"""
    # 外层奖励：缺陷发现 + 新页面覆盖 - 重复惩罚
    # TODO: 还要想
    reward = 0.0
    if episode_info.get("defect_found"):
        reward += 100.0  # 发现缺陷给予高额奖励
    reward += 10.0 * float(episode_info.get("new_pages", 0)) # 每覆盖一个新页面给予奖励
    reward -= 5.0 * float(episode_info.get("duplicate_intent", 0)) # 如果意图重复，给予惩罚
    return reward