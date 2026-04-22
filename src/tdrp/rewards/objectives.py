from __future__ import annotations

from typing import Any
import math

from .time_completion import TimeCompletionReward


class CompletionReward(TimeCompletionReward):
    """Reward task completion without time or deadline shaping."""

    name = "completion"

    def task_deadline_adjustment(self, env: Any, task_id: int, completion_t: float) -> float:
        return 0.0

    def compute(self, env: Any, dt_h: float) -> float:
        return float(env.step_events.get("task_reward", 0.0))


class DeadlineReward(TimeCompletionReward):
    """Reward completion and deadline performance without rendezvous shaping."""

    name = "deadline"

    def compute(self, env: Any, dt_h: float) -> float:
        reward = -float(env.cfg.reward.time_coef) * float(env._norm_global_time(dt_h))
        reward += float(env.step_events.get("task_reward", 0.0))
        return float(reward)


class DecayingValueReward(TimeCompletionReward):
    """Task reward decays with waiting time using task-specific urgency."""

    name = "decay_value"

    def prepare_task_claim(self, env: Any, drone_id: int, task_id: int, side: int) -> None:
        if hasattr(env, "_pending_task_reward"):
            env._pending_task_reward[drone_id] = 0.0

    def task_deadline_adjustment(self, env: Any, task_id: int, completion_t: float) -> float:
        if not (0 <= int(task_id) < env.scenario.ntasks):
            return 0.0
        ntasks = max(1, int(env.scenario.ntasks))
        weight = max(0.0, float(env.scenario.task_reward_weight[task_id]))
        urgency = max(0.0, float(env.scenario.task_urgency[task_id]))
        spawn_t = max(0.0, float(env.scenario.task_spawn_time[task_id]))
        wait_h = max(0.0, float(completion_t) - spawn_t)
        decay = math.exp(-urgency * wait_h)
        decay = max(float(env.cfg.reward.min_task_value_frac), float(decay))
        return float(env.cfg.reward.task_done_bonus) * weight * decay / float(ntasks)

    def compute(self, env: Any, dt_h: float) -> float:
        reward = -float(env.cfg.reward.time_coef) * float(env._norm_global_time(dt_h))
        reward += float(env.step_events.get("task_reward", 0.0))
        return float(reward)


class EnergyAwareReward(TimeCompletionReward):
    """Default reward with an additional normalized energy penalty."""

    name = "energy_aware"

    def compute(self, env: Any, dt_h: float) -> float:
        reward = super().compute(env, dt_h)
        energy_used = float(env.step_events.get("energy_used", 0.0))
        fleet_capacity = max(float(env.cfg.max_battery) * float(env.cfg.ndrones), 1e-9)
        reward -= float(env.cfg.reward.energy_coef) * energy_used / fleet_capacity
        return float(reward)


class RiskSensitiveReward(TimeCompletionReward):
    """Default reward with risk exposure and failure penalties."""

    name = "risk_sensitive"

    def compute(self, env: Any, dt_h: float) -> float:
        reward = super().compute(env, dt_h)
        risk_exposure = float(env.step_events.get("risk_exposure", 0.0))
        failures = float(env.step_events.get("payload_failure_count", 0.0))
        failures += float(env.step_events.get("battery_depleted_count", 0.0))
        reward -= float(env.cfg.reward.risk_coef) * risk_exposure
        reward -= float(env.cfg.reward.failure_penalty) * failures
        return float(reward)
