from __future__ import annotations

from typing import Any

import numpy as np

from tdrp.envs.constants import N_WIND_KERNELS
from tdrp.envs.wind import WindField

from .base import UncertaintyModule


class DynamicWindModule(UncertaintyModule):
    """Spatial and temporal wind perturbation for drone flight and hover power."""

    name = "dynamic_wind"

    def _wind_components(
        self, env: Any, from_xy: np.ndarray, to_xy: np.ndarray
    ) -> tuple[float, float]:
        from_xy = np.asarray(from_xy, dtype=np.float64)
        to_xy = np.asarray(to_xy, dtype=np.float64)
        vec = to_xy - from_xy
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            return 0.0, 0.0
        direction = vec / dist
        wind = env._wind_at(0.5 * (from_xy + to_xy))
        wind_along = float(np.dot(wind, direction))
        wind_cross_vec = wind - wind_along * direction
        wind_cross = float(np.linalg.norm(wind_cross_vec))
        return wind_along, wind_cross

    def on_reset(self, env: Any) -> None:
        unc = env.cfg.uncertainty
        if unc.enabled and unc.wind_enabled and unc.wind_intensity > 0:
            sigma = max(0.01, float(unc.wind_sigma_frac) * float(env.scenario.map_range))
            env._wind_field = WindField(
                map_range=env.scenario.map_range,
                n_kernels=N_WIND_KERNELS,
                sigma=sigma,
                ou_theta=float(unc.wind_ou_theta),
                ou_sigma=float(unc.wind_ou_sigma) * float(unc.wind_intensity),
                max_speed=float(unc.wind_max_speed),
                rng=env._unc_rng,
            )
        else:
            env._wind_field = None

    def on_time_advance(self, env: Any, dt_h: float) -> None:
        if env._wind_field is not None and dt_h >= env.cfg.eps_time:
            env._wind_field.advance(float(dt_h))

    def flight_time(
        self,
        env: Any,
        from_xy: np.ndarray,
        to_xy: np.ndarray,
        base_speed_kmph: float,
        current_h: float,
    ) -> float:
        unc = env.cfg.uncertainty
        if env._wind_field is None or not (
            unc.enabled and unc.wind_enabled and unc.wind_intensity > 0
        ):
            return float(current_h)
        from_xy = np.asarray(from_xy, dtype=np.float64)
        to_xy = np.asarray(to_xy, dtype=np.float64)
        vec = to_xy - from_xy
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            return 0.0
        wind_along, wind_cross = self._wind_components(env, from_xy, to_xy)
        cross_speed_loss = float(unc.wind_crosswind_speed_k) * wind_cross
        eff_speed = max(
            float(base_speed_kmph) + wind_along - cross_speed_loss,
            float(base_speed_kmph) * float(unc.wind_speed_floor),
        )
        return float(dist / eff_speed)

    def hover_power(self, env: Any, xy: np.ndarray, current_power_per_h: float) -> float:
        unc = env.cfg.uncertainty
        if env._wind_field is None or not (
            unc.enabled and unc.wind_enabled and unc.wind_intensity > 0
        ):
            return float(current_power_per_h)
        wind_speed = float(np.linalg.norm(env._wind_at(xy)))
        return float(current_power_per_h) * (
            1.0
            + float(unc.wind_hover_k)
            * wind_speed
            / max(float(unc.wind_max_speed), 1e-9)
        )

    def flight_power(
        self,
        env: Any,
        from_xy: np.ndarray,
        to_xy: np.ndarray,
        base_power_per_h: float,
        current_power_per_h: float,
    ) -> float:
        unc = env.cfg.uncertainty
        if env._wind_field is None or not (
            unc.enabled and unc.wind_enabled and unc.wind_intensity > 0
        ):
            return float(current_power_per_h)
        wind_along, wind_cross = self._wind_components(env, from_xy, to_xy)
        wind_max = max(float(unc.wind_max_speed), 1e-9)
        headwind = max(0.0, -float(wind_along))
        factor = (
            1.0
            + float(unc.wind_headwind_power_k) * headwind / wind_max
            + float(unc.wind_crosswind_power_k) * wind_cross / wind_max
        )
        return float(current_power_per_h) * float(max(1.0, factor))
