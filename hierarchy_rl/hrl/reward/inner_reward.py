

def _compute_inner_step_reward(step_info: dict) -> float:
    """计算内层循环（Executor）每一步的稠密奖励"""
    # 内层步进奖励：路径对齐 + 缺陷触发 - 步数惩罚 - 无效操作
    # TODO：还要想
    reward = 0.0
    if step_info.get("on_path", False):
        reward += 1.0  # 如果操作符合意图路径，给予奖励
    if step_info.get("defect_triggered", False):
        reward += 50.0 # 如果操作触发了缺陷，给予高额奖励
    reward -= 0.1  # 效率惩罚（每多走一步都有轻微惩罚）
    if step_info.get("invalid", False):
        reward -= 1.0 # 无效操作给予惩罚
    return reward