from __future__ import annotations

from typing import Any

from .base import RewardModel


class TimeCompletionReward(RewardModel):
    """Default reward: time cost, task bonuses, deadlines, rendezvous shaping."""

    name = "time_completion"

    def prepare_task_claim(self, env: Any, drone_id: int, task_id: int, side: int) -> None:
        cfg = env.cfg
        ntasks = max(1, int(env.scenario.ntasks))
        weight = 1.0
        if 0 <= int(task_id) < env.scenario.ntasks:
            weight = max(0.2, float(env.scenario.task_reward_weight[task_id]))
        env._pending_task_reward[drone_id] = (
            float(cfg.reward.task_done_bonus) * weight / float(ntasks)
        )

    def task_deadline_adjustment(self, env: Any, task_id: int, completion_t: float) -> float:
        if not (0 <= int(task_id) < env.scenario.ntasks):
            return 0.0
        ddl = float(env.scenario.task_deadline_h[task_id])
        if ddl <= 0.0:
            return 0.0
        weight = max(0.2, float(env.scenario.task_reward_weight[task_id]))
        priority = max(1.0, float(env.scenario.task_priority[task_id]))
        gap = float(ddl) - float(completion_t)
        if gap >= 0.0:
            return float(env.cfg.reward.early_finish_coef) * weight * min(
                gap / max(env.time_ref_h, 1e-6), 1.0
            )
        return -float(env.cfg.reward.deadline_miss_coef) * weight * priority * min(
            abs(gap) / max(env.time_ref_h, 1e-6), 1.0
        )

    def compute(self, env: Any, dt_h: float) -> float:
        reward = -float(env.cfg.reward.time_coef) * float(env._norm_global_time(dt_h))
        reward += float(env.step_events.get("task_reward", 0.0))
        reward += float(env.step_events.get("rendezvous_reward", 0.0))
        return float(reward)
