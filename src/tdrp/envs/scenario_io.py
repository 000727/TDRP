from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .configs import EnvConfig, Scenario
from .constants import MAX_LINE_PTS, MAX_TASKS, TASK_LINE, TASK_POINT
from .scenario_gen import line_midpoint_from_poly, polyline_length


def load_scenario_json(path: str, cfg: EnvConfig | None = None) -> Scenario:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return scenario_from_dict(data, cfg)


def scenario_from_dict(data: dict[str, Any], cfg: EnvConfig | None = None) -> Scenario:
    """Build a Scenario from external training/evaluation data.

    Expected minimal schema:
    {
      "stop_xy": [[x, y], ...],
      "tasks": [
        {"type": "point", "xy": [x, y], "spawn_time": 0.0, ...},
        {"type": "line", "polyline_xy": [[x, y], ...], "spawn_time": 0.4, ...}
      ]
    }
    """

    cfg = cfg or EnvConfig()
    stop_xy = np.asarray(data["stop_xy"], dtype=np.float32)
    if stop_xy.ndim != 2 or stop_xy.shape[1] != 2:
        raise ValueError("scenario stop_xy must have shape (nstops, 2)")
    nstops = int(stop_xy.shape[0])
    map_range = float(data.get("map_range", np.max(stop_xy) if stop_xy.size else 1.0))
    max_time_h = float(data.get("max_time_h", 2.0))
    max_decisions = int(data.get("max_decisions", 100))
    scale = str(data.get("scale", "imported"))

    truck_dist = data.get("truck_dist")
    if truck_dist is None:
        diff = stop_xy[:, None, :] - stop_xy[None, :, :]
        if cfg.road_network_type == "manhattan":
            truck_dist = np.abs(diff).sum(axis=-1)
        else:
            truck_dist = np.linalg.norm(diff, axis=-1)
    truck_dist = np.asarray(truck_dist, dtype=np.float32)
    if truck_dist.shape != (nstops, nstops):
        raise ValueError("scenario truck_dist must have shape (nstops, nstops)")

    raw_tasks = list(data.get("tasks", []))
    point_tasks = [t for t in raw_tasks if _task_kind(t) == TASK_POINT]
    line_tasks = [t for t in raw_tasks if _task_kind(t) == TASK_LINE]
    point_tasks = _initial_first(point_tasks)
    line_tasks = _initial_first(line_tasks)
    tasks = point_tasks + line_tasks
    npoint = len(point_tasks)
    nline = len(line_tasks)
    ntasks = npoint + nline
    if ntasks > MAX_TASKS:
        raise ValueError(f"scenario has {ntasks} tasks, exceeds MAX_TASKS={MAX_TASKS}")

    task_type = np.zeros(ntasks, dtype=np.int8)
    task_type[:npoint] = TASK_POINT
    task_type[npoint:] = TASK_LINE
    point_xy = np.zeros((ntasks, 2), dtype=np.float32)
    line_start_xy = np.zeros((ntasks, 2), dtype=np.float32)
    line_end_xy = np.zeros((ntasks, 2), dtype=np.float32)
    line_polyline_xy = np.zeros((ntasks, MAX_LINE_PTS, 2), dtype=np.float32)
    line_polyline_npts = np.zeros(ntasks, dtype=np.int8)
    line_length = np.zeros(ntasks, dtype=np.float32)
    point_service_h = np.zeros(ntasks, dtype=np.float32)
    task_spawn_time = np.zeros(ntasks, dtype=np.float32)
    task_priority = np.ones(ntasks, dtype=np.int8)
    task_deadline_h = np.full(ntasks, max_time_h, dtype=np.float32)
    task_reward_weight = np.ones(ntasks, dtype=np.float32)
    task_urgency = np.zeros(ntasks, dtype=np.float32)
    task_service_factor = np.ones(ntasks, dtype=np.float32)
    task_risk = np.zeros(ntasks, dtype=np.float32)

    for tid, task in enumerate(tasks):
        spawn_t = float(task.get("spawn_time", task.get("task_spawn_time", 0.0)))
        task_spawn_time[tid] = spawn_t
        task_priority[tid] = int(task.get("priority", task.get("task_priority", 1)))
        task_deadline_h[tid] = float(task.get("deadline_h", task.get("task_deadline_h", max_time_h)))
        task_reward_weight[tid] = float(
            task.get("reward_weight", task.get("task_reward_weight", 1.0))
        )
        task_urgency[tid] = float(
            task.get("urgency", task.get("task_urgency", _default_urgency(task_priority[tid], cfg)))
        )
        task_service_factor[tid] = float(
            task.get("service_factor", task.get("task_service_factor", 1.0))
        )
        task_risk[tid] = float(task.get("risk", task.get("task_risk", 0.0)))

        if int(task_type[tid]) == TASK_POINT:
            xy = np.asarray(task.get("xy", task.get("point_xy")), dtype=np.float32)
            if xy.shape != (2,):
                raise ValueError(f"point task {tid} requires xy/point_xy shape (2,)")
            point_xy[tid] = xy
            point_service_h[tid] = float(
                task.get("service_h", task.get("point_service_h", cfg.default_point_service_h))
            )
            continue

        poly = np.asarray(task.get("polyline_xy", task.get("line_polyline_xy")), dtype=np.float32)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 2:
            raise ValueError(f"line task {tid} requires polyline_xy shape (n>=2, 2)")
        npts = min(int(poly.shape[0]), MAX_LINE_PTS)
        poly = poly[:npts]
        line_polyline_xy[tid, :npts] = poly
        line_polyline_npts[tid] = npts
        line_start_xy[tid] = poly[0]
        line_end_xy[tid] = poly[-1]
        line_length[tid] = float(task.get("line_length", polyline_length(poly)))
        point_xy[tid] = line_midpoint_from_poly(poly)

    npoint_init = int(sum(float(t.get("spawn_time", t.get("task_spawn_time", 0.0))) <= cfg.eps_time for t in point_tasks))
    nline_init = int(sum(float(t.get("spawn_time", t.get("task_spawn_time", 0.0))) <= cfg.eps_time for t in line_tasks))

    return Scenario(
        scale=scale,
        map_range=map_range,
        nstops=nstops,
        npoint=npoint,
        nline=nline,
        npoint_init=npoint_init,
        nline_init=nline_init,
        max_time_h=max_time_h,
        max_decisions=max_decisions,
        stop_xy=stop_xy,
        truck_dist=truck_dist,
        task_type=task_type,
        point_xy=point_xy,
        line_start_xy=line_start_xy,
        line_end_xy=line_end_xy,
        point_service_h=point_service_h,
        line_polyline_xy=line_polyline_xy,
        line_polyline_npts=line_polyline_npts,
        line_length=line_length,
        task_spawn_time=task_spawn_time,
        task_priority=task_priority,
        task_deadline_h=task_deadline_h,
        task_reward_weight=task_reward_weight,
        task_urgency=task_urgency,
        task_service_factor=task_service_factor,
        task_risk=task_risk,
    )


def _task_kind(task: dict[str, Any]) -> int:
    kind = task.get("type", task.get("task_type", "point"))
    if isinstance(kind, str):
        return TASK_LINE if kind.lower() in {"line", "polyline", "corridor"} else TASK_POINT
    return TASK_LINE if int(kind) == TASK_LINE else TASK_POINT


def _initial_first(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        tasks,
        key=lambda t: float(t.get("spawn_time", t.get("task_spawn_time", 0.0))) > 0.0,
    )


def _default_urgency(priority: int, cfg: EnvConfig) -> float:
    ranges = {
        1: cfg.normal_urgency_range,
        2: cfg.urgent_urgency_range,
        3: cfg.critical_urgency_range,
    }
    lo, hi = ranges.get(int(priority), cfg.normal_urgency_range)
    return float(0.5 * (float(lo) + float(hi)))
