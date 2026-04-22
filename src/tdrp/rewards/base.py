from __future__ import annotations

from typing import Any


class RewardModel:
    """Hook interface for problem-specific reward functions."""

    name = "base"

    def on_reset(self, env: Any) -> None:
        return None

    def prepare_task_claim(self, env: Any, drone_id: int, task_id: int, side: int) -> None:
        return None

    def task_deadline_adjustment(self, env: Any, task_id: int, completion_t: float) -> float:
        return 0.0

    def clear_pending_task_reward(self, env: Any, drone_id: int) -> None:
        if hasattr(env, "_pending_task_reward"):
            env._pending_task_reward[drone_id] = 0.0

    def compute(self, env: Any, dt_h: float) -> float:
        return 0.0


def build_reward_model(cfg: Any) -> RewardModel:
    model = getattr(cfg, "reward_model", None)
    if model is not None:
        return model

    name = str(getattr(cfg, "reward_model_name", "time_completion")).lower()
    from .objectives import (
        CompletionReward,
        DecayingValueReward,
        DeadlineReward,
        EnergyAwareReward,
        RiskSensitiveReward,
    )
    from .time_completion import TimeCompletionReward

    models = {
        "time_completion": TimeCompletionReward,
        "completion": CompletionReward,
        "decay_value": DecayingValueReward,
        "decaying_value": DecayingValueReward,
        "deadline": DeadlineReward,
        "energy_aware": EnergyAwareReward,
        "risk_sensitive": RiskSensitiveReward,
    }
    if name not in models:
        raise ValueError(
            f"Unknown reward_model_name={name!r}. Available: {sorted(models)}"
        )
    return models[name]()
