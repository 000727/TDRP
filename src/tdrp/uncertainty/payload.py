from __future__ import annotations

from typing import Any, Optional

import numpy as np

from tdrp.envs.constants import DS_EXECUTING

from .base import UncertaintyModule


class PayloadReliabilityModule(UncertaintyModule):
    """Payload reliability model with lifetime and optional work-age probability."""

    name = "payload_reliability"

    def on_reset(self, env: Any) -> None:
        unc = env.cfg.uncertainty
        env._drone_payload_remain[:] = float(unc.payload_lifetime_h)
        env._drone_sortie_start_t[:] = 0.0
        env._drone_payload_failed[:] = False
        env._drone_payload_work_h = np.zeros(env.cfg.ndrones, dtype=np.float64)

    def on_time_advance(self, env: Any, dt_h: float) -> None:
        if dt_h <= env.cfg.eps_time:
            return
        if not hasattr(env, "_drone_payload_work_h"):
            env._drone_payload_work_h = np.zeros(env.cfg.ndrones, dtype=np.float64)
        for d in range(env.cfg.ndrones):
            if int(env.drone_status[d]) == DS_EXECUTING:
                env._drone_payload_work_h[d] += float(dt_h)

    def on_sortie_start(self, env: Any, drone_id: int) -> None:
        unc = env.cfg.uncertainty
        base = float(unc.payload_lifetime_h)
        if unc.enabled and unc.payload_enabled and unc.payload_intensity > 0:
            base = max(
                1e-6,
                base
                + float(
                    env._unc_rng.normal(
                        0.0, float(unc.payload_deg_std) * float(unc.payload_intensity)
                    )
                ),
            )
        env._drone_payload_remain[drone_id] = base
        env._drone_sortie_start_t[drone_id] = float(env.t)
        env._drone_payload_failed[drone_id] = False
        if hasattr(env, "_drone_payload_work_h"):
            env._drone_payload_work_h[drone_id] = 0.0

    def payload_failed(self, env: Any, drone_id: int) -> Optional[bool]:
        unc = env.cfg.uncertainty
        if not (unc.enabled and unc.payload_enabled and unc.payload_intensity > 0):
            return False

        elapsed = float(env.t) - float(env._drone_sortie_start_t[drone_id])
        if elapsed >= float(env._drone_payload_remain[drone_id]):
            return True

        if not bool(unc.payload_probabilistic_failure):
            return False

        work_h = 0.0
        if hasattr(env, "_drone_payload_work_h"):
            work_h = float(env._drone_payload_work_h[drone_id])
        ref_h = max(float(unc.payload_failure_work_ref_h), 1e-9)
        prob = float(unc.payload_failure_base_prob) + float(
            unc.payload_failure_work_coef
        ) * work_h / ref_h
        prob = float(np.clip(prob * float(unc.payload_intensity), 0.0, 1.0))
        return bool(env._unc_rng.random() < prob)
