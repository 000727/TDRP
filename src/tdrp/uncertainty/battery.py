from __future__ import annotations

from typing import Any

import numpy as np

from tdrp.envs.constants import DS_ON_GROUND, DS_ON_TRUCK

from .base import UncertaintyModule


class BatteryAgingShockModule(UncertaintyModule):
    """Battery shock model with optional age-dependent shock probability."""

    name = "battery_aging_shock"

    def on_reset(self, env: Any) -> None:
        env._drone_battery_used_h = np.zeros(env.cfg.ndrones, dtype=np.float64)

    def on_time_advance(self, env: Any, dt_h: float) -> None:
        if dt_h <= env.cfg.eps_time:
            return
        if not hasattr(env, "_drone_battery_used_h"):
            env._drone_battery_used_h = np.zeros(env.cfg.ndrones, dtype=np.float64)
        for d in range(env.cfg.ndrones):
            if int(env.drone_status[d]) not in (DS_ON_TRUCK, DS_ON_GROUND):
                env._drone_battery_used_h[d] += float(dt_h)

    def on_takeoff(self, env: Any, drone_id: int) -> dict[str, float]:
        unc = env.cfg.uncertainty
        if not (unc.enabled and unc.battery_enabled and unc.battery_intensity > 0):
            return {"battery_shock": 0.0}

        used_h = 0.0
        if hasattr(env, "_drone_battery_used_h"):
            used_h = float(env._drone_battery_used_h[drone_id])
        age_ref = max(float(unc.battery_shock_age_ref_h), 1e-9)
        prob = float(unc.battery_shock_base_prob) + float(
            unc.battery_shock_age_prob_coef
        ) * used_h / age_ref
        prob = float(np.clip(prob * float(unc.battery_intensity), 0.0, 1.0))
        if env._unc_rng.random() > prob:
            return {"battery_shock": 0.0}

        shock = float(abs(env._unc_rng.normal(0.0, float(unc.battery_shock_std))))
        env.drone_batt[drone_id] = max(0.0, float(env.drone_batt[drone_id]) - shock)
        if hasattr(env, "_invalidate_mask"):
            env._invalidate_mask()
        return {"battery_shock": shock}
