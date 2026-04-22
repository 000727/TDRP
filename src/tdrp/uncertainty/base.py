from __future__ import annotations

from typing import Any, Optional

import numpy as np


class UncertaintyModule:
    """Hook interface for environment perturbations.

    Modules are intentionally small and stateful. The environment owns the main
    state machine; modules observe or adjust specific transition quantities.
    """

    name = "base"

    def bind(self, env: Any) -> "UncertaintyModule":
        self.env = env
        return self

    def on_reset(self, env: Any) -> None:
        return None

    def on_time_advance(self, env: Any, dt_h: float) -> None:
        return None

    def flight_time(
        self,
        env: Any,
        from_xy: np.ndarray,
        to_xy: np.ndarray,
        base_speed_kmph: float,
        current_h: float,
    ) -> float:
        return float(current_h)

    def hover_power(self, env: Any, xy: np.ndarray, current_power_per_h: float) -> float:
        return float(current_power_per_h)

    def flight_power(
        self,
        env: Any,
        from_xy: np.ndarray,
        to_xy: np.ndarray,
        base_power_per_h: float,
        current_power_per_h: float,
    ) -> float:
        return float(current_power_per_h)

    def on_takeoff(self, env: Any, drone_id: int) -> dict[str, float]:
        return {}

    def on_sortie_start(self, env: Any, drone_id: int) -> None:
        return None

    def payload_failed(self, env: Any, drone_id: int) -> Optional[bool]:
        return None

    def on_new_task_spawn(self, env: Any, task_id: int) -> Optional[bool]:
        return None


def build_uncertainty_modules(cfg: Any) -> list[UncertaintyModule]:
    """Return configured uncertainty modules.

    Passing cfg.uncertainty_modules gives full control to the caller. Otherwise
    the legacy wind, dynamic-task, battery, and payload modules are assembled.
    """

    modules = list(getattr(cfg, "uncertainty_modules", []) or [])
    if modules:
        return [m if isinstance(m, UncertaintyModule) else m for m in modules]

    from .battery import BatteryAgingShockModule
    from .dynamic_tasks import DynamicTaskSpawnModule
    from .payload import PayloadReliabilityModule
    from .wind import DynamicWindModule

    return [
        DynamicWindModule(),
        DynamicTaskSpawnModule(),
        BatteryAgingShockModule(),
        PayloadReliabilityModule(),
    ]
