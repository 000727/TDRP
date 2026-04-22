from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .configs import EnvConfig, Scenario
from .constants import (
    MAX_LINE_PTS, MAX_TASKS, SCALE_CONFIGS,
    TASK_LINE, TASK_POINT,
)


def polyline_length(pts: np.ndarray) -> float:
    """Return total arc length of a polyline (N, 2)."""
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())


def line_midpoint_from_poly(pts: np.ndarray) -> np.ndarray:
    """Return the point at the midpoint arc-length of a polyline."""
    if pts.shape[0] <= 1:
        return pts[0].copy()
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(seg.sum())
    if total <= 1e-9:
        return pts[0].copy()
    half, acc = total / 2.0, 0.0
    for i, s in enumerate(seg):
        if acc + float(s) >= half and s > 1e-9:
            a = (half - acc) / float(s)
            return (pts[i] + a * (pts[i + 1] - pts[i])).astype(np.float32)
        acc += float(s)
    return pts[-1].copy()


class ScenarioGenerator:
    """Generates Scenario objects and future task geometry for TruckMultiDroneCleanEnv.

    All randomness is passed in via rng so the env controls seeding.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, rng: np.random.Generator) -> Scenario:
        """Generate a full episode scenario (initial + scheduled future tasks)."""
        cfg = self.cfg
        if cfg.scenario_data is not None or cfg.scenario_path:
            from .scenario_io import load_scenario_json, scenario_from_dict

            return (
                load_scenario_json(cfg.scenario_path, cfg)
                if cfg.scenario_path
                else scenario_from_dict(cfg.scenario_data, cfg)
            )
        spec = SCALE_CONFIGS[self._pick_scale(rng)]
        map_range = float(spec["map_range"])
        nstops = int(spec["nstops"])
        npoint_init = int(spec["npoint"])
        nline_init = int(spec["nline"])
        max_time_h = float(spec["max_time_h"])
        max_decisions = int(spec["max_decisions"])

        npoint_future, nline_future = self._sample_future_task_counts(
            rng, npoint_init, nline_init
        )

        npoint = npoint_init + npoint_future
        nline = nline_init + nline_future
        ntasks = npoint + nline
        if ntasks > MAX_TASKS:
            raise RuntimeError(
                f"Total tasks {ntasks} (init {npoint_init + nline_init} + future "
                f"{npoint_future + nline_future}) exceeds MAX_TASKS={MAX_TASKS}."
            )

        max_radius_pt, max_radius_ln, max_line_len, min_line_len = \
            self._task_generation_limits(map_range)

        for _ in range(cfg.scenario_max_global_retry):
            stop_xy = rng.uniform(0.0, map_range, (nstops, 2)).astype(np.float32)
            diff = stop_xy[:, None, :] - stop_xy[None, :, :]
            if cfg.road_network_type == "euclidean":
                truck_dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
            else:
                truck_dist = np.abs(diff).sum(axis=-1).astype(np.float32)

            task_type = np.zeros(ntasks, dtype=np.int8)
            task_type[:npoint] = TASK_POINT
            task_type[npoint:ntasks] = TASK_LINE
            point_xy = np.zeros((ntasks, 2), dtype=np.float32)
            line_start_xy = np.zeros((ntasks, 2), dtype=np.float32)
            line_end_xy = np.zeros((ntasks, 2), dtype=np.float32)
            line_polyline_xy = np.zeros((ntasks, MAX_LINE_PTS, 2), dtype=np.float32)
            line_polyline_npts = np.zeros(ntasks, dtype=np.int8)
            line_length = np.zeros(ntasks, dtype=np.float32)
            service = np.zeros(ntasks, dtype=np.float32)
            task_priority = np.zeros(ntasks, dtype=np.int8)
            task_deadline_h = np.zeros(ntasks, dtype=np.float32)
            task_reward_weight = np.zeros(ntasks, dtype=np.float32)
            task_urgency = np.zeros(ntasks, dtype=np.float32)
            task_service_factor = np.ones(ntasks, dtype=np.float32)
            task_risk = np.zeros(ntasks, dtype=np.float32)

            ok = True
            for i in range(npoint_init):
                anchor = stop_xy[int(rng.integers(0, nstops))]
                found = False
                for _ in range(cfg.scenario_max_task_retry):
                    service_h = self._sample_point_service_h(rng)
                    pt = self._sample_circle(rng, anchor, max_radius_pt, map_range)
                    if not self._feasible_point_task(stop_xy, pt, service_h):
                        continue
                    point_xy[i] = pt
                    service[i] = float(service_h)
                    prio, ddl, rwt, urg, sf, risk = self._sample_task_metadata(
                        rng, 0.0, TASK_POINT, service_h, max_time_h)
                    task_priority[i] = prio
                    task_deadline_h[i] = ddl
                    task_reward_weight[i] = rwt
                    task_urgency[i] = urg
                    task_service_factor[i] = sf
                    task_risk[i] = risk
                    found = True
                    break
                if not found:
                    ok = False
                    break
            if not ok:
                continue

            for i in range(npoint, npoint + nline_init):
                anchor = stop_xy[int(rng.integers(0, nstops))]
                found = False
                for _ in range(cfg.scenario_max_task_retry):
                    prio, ddl, rwt, urg, sf, risk = self._sample_task_metadata(
                        rng, 0.0, TASK_LINE, 0.0, max_time_h)
                    s_pos = self._sample_circle(rng, anchor, max_radius_ln, map_range)
                    poly = self._build_random_polyline(
                        rng, s_pos, map_range, max_line_len,
                        max_vertices=min(MAX_LINE_PTS, 5))
                    if not self._feasible_line_task(stop_xy, map_range, poly, sf):
                        continue
                    npts = min(int(poly.shape[0]), MAX_LINE_PTS)
                    poly = poly[:npts]
                    line_polyline_xy[i, :npts] = poly
                    line_polyline_npts[i] = npts
                    line_length[i] = polyline_length(poly)
                    line_start_xy[i] = poly[0]
                    line_end_xy[i] = poly[-1]
                    point_xy[i] = line_midpoint_from_poly(poly)
                    task_priority[i] = prio
                    task_deadline_h[i] = ddl
                    task_reward_weight[i] = rwt
                    task_urgency[i] = urg
                    task_service_factor[i] = sf
                    task_risk[i] = risk
                    found = True
                    break
                if not found:
                    ok = False
                    break
            if not ok:
                continue

            task_spawn_time = np.zeros(ntasks, dtype=np.float32)
            task_spawn_time[npoint_init:npoint] = self._sample_spawn_times(
                rng, npoint_future, max_time_h
            )
            task_spawn_time[npoint + nline_init:ntasks] = self._sample_spawn_times(
                rng, nline_future, max_time_h
            )

            return Scenario(
                scale=next(k for k, v in SCALE_CONFIGS.items() if v is spec),
                map_range=map_range, nstops=nstops,
                npoint=npoint, nline=nline,
                npoint_init=npoint_init, nline_init=nline_init,
                max_time_h=max_time_h, max_decisions=max_decisions,
                stop_xy=stop_xy, truck_dist=truck_dist, task_type=task_type,
                point_xy=point_xy, line_start_xy=line_start_xy, line_end_xy=line_end_xy,
                point_service_h=service,
                line_polyline_xy=line_polyline_xy, line_polyline_npts=line_polyline_npts,
                line_length=line_length,
                task_spawn_time=task_spawn_time,
                task_priority=task_priority,
                task_deadline_h=task_deadline_h,
                task_reward_weight=task_reward_weight,
                task_urgency=task_urgency,
                task_service_factor=task_service_factor,
                task_risk=task_risk,
            )
        raise RuntimeError("Scenario generation failed after max retries.")

    def generate_task_slot(
        self,
        scenario: Scenario,
        tid: int,
        current_t: float,
        rng: np.random.Generator,
    ) -> Optional[Dict]:
        """Generate geometry + metadata for future task slot tid.

        Writes task attributes directly into scenario arrays.
        Returns a dict of env-state array updates on success, or None on failure.
        Bug fix: original code referenced undefined `max_time_h`; now uses scenario.max_time_h.
        """
        cfg = self.cfg
        sc = scenario
        if not (0 <= int(tid) < sc.ntasks):
            return None

        map_range = float(sc.map_range)
        max_radius_pt, max_radius_ln, max_line_len, _ = self._task_generation_limits(map_range)
        is_point = int(tid) < int(sc.npoint)

        if is_point:
            sc.task_type[tid] = TASK_POINT
            for _ in range(cfg.scenario_max_task_retry):
                anchor = sc.stop_xy[int(rng.integers(0, sc.nstops))]
                service_h = self._sample_point_service_h(rng)
                pt = self._sample_circle(rng, anchor, max_radius_pt, map_range)
                if not self._feasible_point_task(sc.stop_xy, pt, service_h):
                    continue
                sc.point_xy[tid] = pt
                sc.line_start_xy[tid] = 0.0
                sc.line_end_xy[tid] = 0.0
                sc.line_polyline_xy[tid] = 0.0
                sc.line_polyline_npts[tid] = 0
                sc.line_length[tid] = 0.0
                sc.point_service_h[tid] = float(service_h)
                self._write_task_profile(rng, sc, tid, TASK_POINT, current_t, service_h)
                return {
                    "point_service_remain": float(service_h),
                    "line_remain_xy": np.zeros((MAX_LINE_PTS, 2), dtype=np.float32),
                    "line_remain_npts": 0,
                    "line_remain_len": 0.0,
                    "line_len": 0.0,
                }
        else:
            sc.task_type[tid] = TASK_LINE
            for _ in range(cfg.scenario_max_task_retry):
                anchor = sc.stop_xy[int(rng.integers(0, sc.nstops))]
                s_pos = self._sample_circle(rng, anchor, max_radius_ln, map_range)
                poly = self._build_random_polyline(
                    rng, s_pos, map_range, max_line_len,
                    max_vertices=min(MAX_LINE_PTS, 5))
                self._write_task_profile(rng, sc, tid, TASK_LINE, current_t, 0.0)
                sf = float(sc.task_service_factor[tid])
                if not self._feasible_line_task(sc.stop_xy, map_range, poly, sf):
                    continue
                npts = min(int(poly.shape[0]), MAX_LINE_PTS)
                poly = poly[:npts]
                sc.point_service_h[tid] = 0.0
                sc.line_polyline_xy[tid] = 0.0
                sc.line_polyline_xy[tid, :npts] = poly
                sc.line_polyline_npts[tid] = npts
                sc.line_length[tid] = polyline_length(poly)
                sc.line_start_xy[tid] = poly[0]
                sc.line_end_xy[tid] = poly[-1]
                sc.point_xy[tid] = line_midpoint_from_poly(poly)
                remain_xy = np.zeros((MAX_LINE_PTS, 2), dtype=np.float32)
                remain_xy[:npts] = poly
                return {
                    "point_service_remain": 0.0,
                    "line_remain_xy": remain_xy,
                    "line_remain_npts": npts,
                    "line_remain_len": float(sc.line_length[tid]),
                    "line_len": float(sc.line_length[tid]),
                }
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pick_scale(self, rng: np.random.Generator) -> str:
        if self.cfg.fixed_scale is not None:
            return self.cfg.fixed_scale
        return list(SCALE_CONFIGS.keys())[int(rng.integers(0, len(SCALE_CONFIGS)))]

    def _task_generation_limits(self, map_range: float) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        vline = cfg.drone_speed_kmph * cfg.line_speed_factor
        usable = cfg.max_battery - cfg.safety_batt_margin
        ovhd = cfg.fly_power_per_h * (cfg.takeoff_time_h + cfg.landing_time_h)
        fly_pt = max(0.0, usable - cfg.hover_power_per_h * max(
            cfg.default_point_service_h, cfg.point_service_h_max) - ovhd)
        fly_ln = max(0.0, usable - ovhd)
        max_radius_pt = fly_pt / (2.0 * cfg.fly_power_per_h) * cfg.drone_speed_kmph * 0.90
        max_radius_ln = fly_ln * 0.22 / cfg.fly_power_per_h * cfg.drone_speed_kmph * 0.85
        max_line_len = fly_ln * 0.50 / cfg.fly_power_per_h * vline * 0.85
        min_line_len = max(0.08, map_range * 0.015)
        return float(max_radius_pt), float(max_radius_ln), float(max_line_len), float(min_line_len)

    def _feasible_point_task(self, stop_xy: np.ndarray, pos: np.ndarray, service_h: float) -> bool:
        cfg = self.cfg
        d = float(np.min(np.linalg.norm(stop_xy - pos, axis=1)))
        e = cfg.fly_power_per_h * (cfg.takeoff_time_h + d / cfg.drone_speed_kmph)
        e += cfg.hover_power_per_h * float(service_h)
        e += cfg.fly_power_per_h * (d / cfg.drone_speed_kmph + cfg.landing_time_h)
        e += cfg.safety_batt_margin
        return e <= cfg.max_battery

    def _feasible_line_task(self, stop_xy: np.ndarray, map_range: float,
                            poly: np.ndarray, service_factor: float) -> bool:
        cfg = self.cfg
        _, _, _, min_line_len = self._task_generation_limits(map_range)
        ll = polyline_length(poly)
        if ll < min_line_len:
            return False
        ds = float(np.min(np.linalg.norm(stop_xy - poly[0], axis=1)))
        de = float(np.min(np.linalg.norm(stop_xy - poly[-1], axis=1)))
        vline = cfg.drone_speed_kmph * cfg.line_speed_factor
        e = cfg.fly_power_per_h * (cfg.takeoff_time_h + ds / cfg.drone_speed_kmph)
        e += cfg.fly_power_per_h * ll * float(service_factor) / vline
        e += cfg.fly_power_per_h * (de / cfg.drone_speed_kmph + cfg.landing_time_h)
        e += cfg.safety_batt_margin
        return e <= cfg.max_battery

    def _sample_task_priority(self, rng: np.random.Generator) -> int:
        if not self.cfg.heter_task_enabled:
            return 1
        u = float(rng.random())
        crit = float(np.clip(self.cfg.critical_task_ratio, 0.0, 1.0))
        urg = float(np.clip(self.cfg.urgent_task_ratio, 0.0, 1.0 - crit))
        if u < crit:
            return 3
        if u < crit + urg:
            return 2
        return 1

    def _sample_task_metadata(
        self,
        rng: np.random.Generator,
        spawn_t_h: float,
        task_type: int,
        service_h: float,
        max_time_h: float,
    ) -> Tuple[int, float, float, float, float]:
        """Return (priority, deadline_h, reward_weight, urgency, service_factor, risk)."""
        cfg = self.cfg
        priority = self._sample_task_priority(rng)
        base_weight = 1.0 + 0.45 * float(priority - 1) + (0.10 if int(task_type) == TASK_LINE else 0.0)
        if float(cfg.task_weight_jitter_std) > 0.0:
            base_weight = base_weight + rng.normal(0.0, float(cfg.task_weight_jitter_std))
        reward_weight = float(
            np.clip(base_weight, float(cfg.task_weight_min), float(cfg.task_weight_max))
        )
        urgency = self._sample_task_urgency(rng, priority)
        if int(task_type) == TASK_POINT:
            service_factor = 1.0
        else:
            service_factor = float(rng.uniform(cfg.line_service_factor_min,
                                               cfg.line_service_factor_max))
        base_risk = 0.10 + 0.12 * float(priority - 1) + (0.08 if int(task_type) == TASK_LINE else 0.0)
        base_risk += 0.25 * max(0.0, float(service_factor) - 1.0)
        risk = float(np.clip(base_risk + rng.normal(0.0, 0.03), 0.02, 0.95))

        slack_frac = float(rng.uniform(cfg.deadline_slack_min_frac, cfg.deadline_slack_max_frac))
        priority_tightness = {1: 1.00, 2: 0.70, 3: 0.45}[int(priority)]
        if int(task_type) == TASK_POINT:
            slack_scale = max(0.15, 0.30 + 2.0 * float(service_h) / max(float(max_time_h), 1e-6))
        else:
            slack_scale = max(0.20, 0.45 * float(service_factor))
        deadline_h = float(spawn_t_h + slack_frac * priority_tightness * slack_scale * float(max_time_h))
        deadline_h = float(np.clip(deadline_h, spawn_t_h + 1e-3, float(max_time_h)))
        return int(priority), deadline_h, reward_weight, urgency, float(service_factor), float(risk)

    def _sample_task_urgency(self, rng: np.random.Generator, priority: int) -> float:
        ranges = {
            1: self.cfg.normal_urgency_range,
            2: self.cfg.urgent_urgency_range,
            3: self.cfg.critical_urgency_range,
        }
        lo, hi = ranges.get(int(priority), self.cfg.normal_urgency_range)
        lo = max(0.0, float(lo))
        hi = max(lo, float(hi))
        if bool(getattr(self.cfg, "task_urgency_random", False)):
            return float(rng.uniform(lo, hi))
        return float(0.5 * (lo + hi))

    def _sample_future_task_counts(
        self, rng: np.random.Generator, npoint_init: int, nline_init: int
    ) -> Tuple[int, int]:
        cfg = self.cfg
        if not (cfg.allow_dynamic_tasks and cfg.dynamic_task_ratio > 0.0):
            return 0, 0
        process = str(cfg.dynamic_arrival_process).lower()
        if process == "poisson":
            npoint = int(rng.poisson(max(0.0, npoint_init * float(cfg.dynamic_task_ratio))))
            nline = int(rng.poisson(max(0.0, nline_init * float(cfg.dynamic_task_ratio))))
            return npoint, nline
        return (
            int(round(npoint_init * float(cfg.dynamic_task_ratio))),
            int(round(nline_init * float(cfg.dynamic_task_ratio))),
        )

    def _sample_spawn_times(
        self, rng: np.random.Generator, n: int, max_time_h: float
    ) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        cfg = self.cfg
        t_min = float(cfg.dynamic_spawn_t_min_frac) * float(max_time_h)
        t_max = float(cfg.dynamic_spawn_t_max_frac) * float(max_time_h)
        if t_max <= t_min:
            t_max = max(t_min + 1e-3, float(max_time_h) * 0.5)
        # Conditional on the number of arrivals, a homogeneous Poisson process
        # has arrival times distributed as sorted uniform order statistics.
        times = rng.uniform(t_min, t_max, size=int(n))
        return np.sort(times).astype(np.float32)

    def _sample_point_service_h(self, rng: np.random.Generator) -> float:
        if not self.cfg.heter_task_enabled:
            return float(self.cfg.default_point_service_h)
        lo = min(float(self.cfg.point_service_h_min), float(self.cfg.point_service_h_max))
        hi = max(float(self.cfg.point_service_h_min), float(self.cfg.point_service_h_max))
        return float(rng.uniform(lo, hi))

    def _build_random_polyline(
        self,
        rng: np.random.Generator,
        start_xy: np.ndarray,
        map_range: float,
        max_total_len: float,
        max_vertices: int = 5,
    ) -> np.ndarray:
        npts = int(rng.integers(2, max_vertices + 1))
        direction = rng.normal(0.0, 1.0, size=2)
        norm = float(np.linalg.norm(direction))
        direction = np.array([1.0, 0.0], dtype=np.float32) if norm < 1e-9 \
            else (direction / norm).astype(np.float32)
        base_len = float(rng.uniform(max_total_len * 0.55, max_total_len))
        end_xy = np.clip(start_xy + direction * base_len, 0.0, map_range).astype(np.float32)
        perp = np.array([-direction[1], direction[0]], dtype=np.float32)
        pts = [start_xy.astype(np.float32)]
        for k in range(1, npts - 1):
            alpha = float(k) / float(npts - 1)
            base = (1.0 - alpha) * start_xy + alpha * end_xy
            jitter = float(rng.uniform(-0.18, 0.18)) * max_total_len
            pts.append(np.clip(base + perp * jitter, 0.0, map_range).astype(np.float32))
        pts.append(end_xy)
        pts = np.asarray(pts, dtype=np.float32)
        total = polyline_length(pts)
        if total > max_total_len and total > 1e-9:
            scale = max_total_len / total
            for k in range(1, len(pts) - 1):
                pts[k] = np.clip(start_xy + (pts[k] - start_xy) * scale, 0.0, map_range)
        return pts.astype(np.float32)

    def _sample_circle(
        self,
        rng: np.random.Generator,
        center: np.ndarray,
        radius: float,
        map_range: float,
    ) -> np.ndarray:
        r = radius * math.sqrt(float(rng.random()))
        theta = float(rng.random()) * 2.0 * math.pi
        pt = center + np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)
        return np.clip(pt, 0.0, map_range).astype(np.float32)

    def _write_task_profile(
        self,
        rng: np.random.Generator,
        scenario: Scenario,
        tid: int,
        task_type: int,
        spawn_t_h: float,
        service_h: float = 0.0,
    ) -> None:
        prio, ddl, rwt, urg, sf, risk = self._sample_task_metadata(
            rng, spawn_t_h, task_type, service_h, float(scenario.max_time_h))
        scenario.task_priority[tid] = int(prio)
        scenario.task_deadline_h[tid] = float(ddl)
        scenario.task_reward_weight[tid] = float(rwt)
        scenario.task_urgency[tid] = float(urg)
        scenario.task_service_factor[tid] = float(sf)
        scenario.task_risk[tid] = float(risk)
