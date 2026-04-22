from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..rewards import build_reward_model
from ..uncertainty import build_uncertainty_modules
from .configs import EnvConfig, Scenario
from .constants import (
    DS_CRUISING, DS_EXECUTING, DS_LANDING, DS_ON_GROUND, DS_ON_TRUCK,
    DS_TAKING_OFF, DS_WAITING,
    EV_DRONE_ARRIVE_STOP, EV_DRONE_ARRIVE_TASK, EV_DRONE_LAND_DONE,
    EV_DRONE_TAKEOFF_DONE, EV_DRONE_TASK_DONE, EV_NEW_TASK_SPAWN,
    EV_SWAP_DONE, EV_TRUCK_ARRIVE_STOP,
    MAX_LINE_PTS, MAX_STOPS, MAX_TASK_ACTIONS, MAX_TASKS,
    N_WIND_KERNELS, SCALE_TIME_REFS_H,
    TASK_LINE, TASK_POINT,
)
from .scenario_gen import ScenarioGenerator, polyline_length


class TruckMultiDroneCleanEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: EnvConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        D = cfg.ndrones
        unc = cfg.uncertainty

        # Action space layout: [0, MAX_STOPS) = stop actions,
        # [MAX_STOPS, MAX_STOPS+MAX_TASK_ACTIONS) = task actions, AWAIT = wait
        self.act_offset_stop = 0
        self.act_offset_task = MAX_STOPS
        self.AWAIT = MAX_STOPS + MAX_TASK_ACTIONS
        self.nactions = self.AWAIT + 1
        self.nagents = 1 + D

        self._task_action_tid = -np.ones(MAX_TASK_ACTIONS, dtype=np.int32)
        self._task_action_side = np.zeros(MAX_TASK_ACTIONS, dtype=np.int8)
        self._task_action_count = 0
        self._response_mode = False
        self._joint_replan_requested = False
        self._replan_reason = ""
        self.planned_stop = -np.ones(D, dtype=np.int32)
        self._queued_truck_target = -np.ones(1, dtype=np.int32)
        self._maintenance_queue: List[int] = []
        self._active_rendezvous_stop = -1

        self.action_space = spaces.MultiDiscrete([self.nactions] * self.nagents)

        self.observation_space = spaces.Dict({
            "time_progress": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "done_ratio": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "scale_id": spaces.Box(0, 4, (1,), dtype=np.int8),

            "stop_xy": spaces.Box(-1.0, 1.0, (MAX_STOPS, 2), dtype=np.float32),
            "stop_mask": spaces.Box(0, 1, (MAX_STOPS,), dtype=np.int8),

            "task_type": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
            "task_mask": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
            "point_xy": spaces.Box(-1.0, 1.0, (MAX_TASKS, 2), dtype=np.float32),
            "line_start_xy": spaces.Box(-1.0, 1.0, (MAX_TASKS, 2), dtype=np.float32),
            "line_end_xy": spaces.Box(-1.0, 1.0, (MAX_TASKS, 2), dtype=np.float32),
            "point_service_h": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
            "task_priority": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
            "task_deadline_h": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
            "task_reward_weight": spaces.Box(0.0, 4.0, (MAX_TASKS,), dtype=np.float32),
            "task_urgency": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
            "task_service_factor": spaces.Box(0.0, 2.0, (MAX_TASKS,), dtype=np.float32),
            "task_risk": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),

            "task_done": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
            "task_claimed": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
            "task_spawned": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),

            "truck_rel_pos": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "truck_stop": spaces.Box(0, MAX_STOPS - 1, (1,), dtype=np.int32),
            "truck_busy": spaces.Box(0, 1, (1,), dtype=np.float32),
            "truck_eta_h": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "truck_target_stop": spaces.Box(0, MAX_STOPS - 1, (1,), dtype=np.int32),

            "drone_status": spaces.Box(0, 6, (D,), dtype=np.int8),
            "drone_stop_id": spaces.Box(0, MAX_STOPS - 1, (D,), dtype=np.int32),
            "drone_task_id": spaces.Box(-1, MAX_TASKS - 1, (D,), dtype=np.int32),
            "drone_task_side": spaces.Box(0, 1, (D,), dtype=np.int8),
            "drone_eta_h": spaces.Box(0.0, 1.0, (D,), dtype=np.float32),
            "drone_batt_ratio": spaces.Box(0.0, 1.0, (D,), dtype=np.float32),
            "drone_xy": spaces.Box(-1.0, 1.0, (D, 2), dtype=np.float32),

            "wind_vec": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "wind_samples": spaces.Box(-1.0, 1.0, (N_WIND_KERNELS, 2), dtype=np.float32),
            "wind_kernel_xy": spaces.Box(-1.0, 1.0, (N_WIND_KERNELS, 2), dtype=np.float32),

            "drone_payload_remain": spaces.Box(0.0, 1.0, (D,), dtype=np.float32),
            "drone_payload_failed": spaces.Box(0, 1, (D,), dtype=np.int8),
            "drone_planned_stop": spaces.Box(-1, MAX_STOPS - 1, (D,), dtype=np.int32),
            "response_mode": spaces.Box(0, 1, (1,), dtype=np.float32),

            "action_mask": spaces.Box(0, 1, (1 + D, self.AWAIT + 1), dtype=np.int8),
        })

        self._scenario_gen = ScenarioGenerator(cfg)
        self.reward_model = build_reward_model(cfg)
        self._uncertainty_modules = []

        self.scenario: Optional[Scenario] = None
        self.np_random: Optional[np.random.Generator] = None
        self._unc_rng: Optional[np.random.Generator] = None

        self.t = 0.0
        self.decision_count = 0
        self.drift_counter = 0
        self.evq: List[Tuple[float, int, int, int, int]] = []
        self._ev_counter = 0
        self._truck_pending_swaps = 0

        self.truck_stop = np.zeros(1, dtype=np.int32)
        self.truck_busy = np.zeros(1, dtype=np.int8)
        self.truck_eta = np.zeros(1, dtype=np.float32)
        self.truck_target = np.zeros(1, dtype=np.int32)

        self._truck_moving = np.zeros(1, dtype=np.int8)
        self._truck_move_from_stop = np.zeros(1, dtype=np.int32)
        self._truck_move_start_t = np.zeros(1, dtype=np.float32)
        self._truck_arrival_seq = np.zeros(1, dtype=np.int32)

        self.drone_status = np.zeros(D, dtype=np.int8)
        self.drone_stop_id = np.zeros(D, dtype=np.int32)
        self.drone_task_id = -np.ones(D, dtype=np.int32)
        self.drone_task_side = np.zeros(D, dtype=np.int8)
        self.drone_eta = np.zeros(D, dtype=np.float32)
        self.drone_batt = np.full(D, cfg.max_battery, dtype=np.float32)
        self._drone_xy = np.zeros((D, 2), dtype=np.float32)

        self._seg_active = np.zeros(D, dtype=np.int8)
        self._seg_from_xy = np.zeros((D, 2), dtype=np.float32)
        self._seg_to_xy = np.zeros((D, 2), dtype=np.float32)
        self._seg_t_start = np.zeros(D, dtype=np.float64)
        self._seg_t_end = np.zeros(D, dtype=np.float64)

        self.task_done = np.zeros(MAX_TASKS, dtype=np.int8)
        self.task_claimed = np.zeros(MAX_TASKS, dtype=np.int8)
        self._task_spawned = np.zeros(MAX_TASKS, dtype=np.int8)

        self._line_len = np.zeros(MAX_TASKS, dtype=np.float32)
        self._point_service_remain = np.zeros(MAX_TASKS, dtype=np.float32)
        self._line_remain_xy = np.zeros((MAX_TASKS, MAX_LINE_PTS, 2), dtype=np.float32)
        self._line_remain_npts = np.zeros(MAX_TASKS, dtype=np.int8)
        self._line_remain_len = np.zeros(MAX_TASKS, dtype=np.float32)
        self._exec_chunk_dt = np.zeros(D, dtype=np.float32)
        self._exec_next_line_xy = np.zeros((D, MAX_LINE_PTS, 2), dtype=np.float32)
        self._exec_next_line_npts = np.zeros(D, dtype=np.int8)
        self._exec_task_entry_xy = np.zeros((D, 2), dtype=np.float32)
        self._min_dist_point_to_stop = np.zeros(MAX_TASKS, dtype=np.float32)
        self._min_dist_lstart_to_stop = np.zeros(MAX_TASKS, dtype=np.float32)
        self._min_dist_lend_to_stop = np.zeros(MAX_TASKS, dtype=np.float32)

        self._mask_dirty = True
        self._cached_mask: Optional[np.ndarray] = None

        self._wind_field: Optional[WindField] = None
        self._drone_payload_remain = np.full(D, unc.payload_lifetime_h, dtype=np.float64)
        self._drone_sortie_start_t = np.zeros(D, dtype=np.float64)
        self._drone_payload_failed = np.zeros(D, dtype=np.bool_)
        self._drone_exec_is_hover = np.zeros(D, dtype=np.bool_)

        self.step_events: Dict[str, float] = {}
        self.episode_invalid_actions = 0
        self.episode_claim_conflicts = 0
        self.episode_wait_step_count = 0
        self.episode_wait_step_time_h = 0.0
        self.episode_ground_wait_by_stop_h = np.zeros(MAX_STOPS, dtype=np.float32)
        self.episode_new_task_spawned = 0
        self._step_event_reasons: List[str] = []

        self.time_ref_h = 1.0
        self._pending_task_reward = np.zeros(D, dtype=np.float32)
        self._pending_rendezvous_start_t = -np.ones(D, dtype=np.float32)
        self._pending_rendezvous_stop = -np.ones(D, dtype=np.int32)

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        try:
            super().reset(seed=seed)
        except Exception:
            pass

        self.np_random = np.random.default_rng(seed)
        unc_seed = self.cfg.uncertainty.fixed_seed if self.cfg.uncertainty.fixed_seed is not None \
            else ((seed + 12345) if seed is not None else None)
        self._unc_rng = np.random.default_rng(unc_seed)

        self.scenario = (options["scenario"] if options and "scenario" in options
                         else self._scenario_gen.generate(self.np_random))
        sc, cfg = self.scenario, self.cfg

        self._line_len[:] = 0.0
        self._line_len[:sc.ntasks] = sc.line_length[:sc.ntasks]
        self._point_service_remain[:] = 0.0
        self._point_service_remain[:sc.ntasks] = sc.point_service_h[:sc.ntasks]
        self._line_remain_xy[:] = 0.0
        self._line_remain_xy[:sc.ntasks] = sc.line_polyline_xy[:sc.ntasks]
        self._line_remain_npts[:] = 0
        self._line_remain_npts[:sc.ntasks] = sc.line_polyline_npts[:sc.ntasks]
        self._line_remain_len[:] = 0.0
        self._line_remain_len[:sc.ntasks] = sc.line_length[:sc.ntasks]
        self._exec_chunk_dt[:] = 0.0
        self._exec_next_line_xy[:] = 0.0
        self._exec_next_line_npts[:] = 0
        self._exec_task_entry_xy[:] = 0.0

        self._task_spawned[:] = 0
        for tid in range(sc.ntasks):
            if float(sc.task_spawn_time[tid]) <= cfg.eps_time:
                self._task_spawned[tid] = 1
            elif not cfg.dynamic_generate_online:
                result = self._scenario_gen.generate_task_slot(
                    sc, int(tid), float(sc.task_spawn_time[tid]), self.np_random)
                if result is not None:
                    self._apply_generated_task_slot(int(tid), result)

        self._build_task_action_map()

        self.t = 0.0
        self._response_mode = False
        self._joint_replan_requested = False
        self._replan_reason = ""
        self.decision_count = 0
        self.drift_counter = 0
        self.evq.clear()
        self._ev_counter = 0
        self._truck_pending_swaps = 0

        for tid in range(sc.ntasks):
            t_spawn = float(sc.task_spawn_time[tid])
            if t_spawn > cfg.eps_time and t_spawn < sc.max_time_h - cfg.eps_time:
                self._push_event(t_spawn, EV_NEW_TASK_SPAWN, int(tid), 0)

        self.truck_stop[0] = cfg.depot_stop_id
        self.truck_busy[0] = 0
        self.truck_eta[0] = 0.0
        self.truck_target[0] = cfg.depot_stop_id
        self._truck_moving[0] = 0
        self._truck_move_from_stop[0] = cfg.depot_stop_id
        self._truck_move_start_t[0] = 0.0
        self._truck_arrival_seq[0] = 0

        self.drone_status[:] = DS_ON_TRUCK
        self.drone_stop_id[:] = cfg.depot_stop_id
        self.drone_task_id[:] = -1
        self.drone_task_side[:] = 0
        self.drone_eta[:] = 0.0
        self.drone_batt[:] = float(cfg.max_battery)
        self.planned_stop[:] = -1
        self._queued_truck_target[:] = -1
        self._maintenance_queue.clear()
        self._active_rendezvous_stop = -1

        self.task_done[:] = 0
        self.task_claimed[:] = 0
        self._seg_active[:] = 0
        self._drone_exec_is_hover[:] = False

        self._init_uncertainty_state()
        self._precompute_task_stop_dists()
        self._set_reward_reference_scales()
        self.episode_invalid_actions = 0
        self.episode_claim_conflicts = 0
        self.episode_wait_step_count = 0
        self.episode_wait_step_time_h = 0.0
        self.episode_ground_wait_by_stop_h[:] = 0.0
        self.episode_new_task_spawned = 0
        self._step_event_reasons = []
        self._pending_task_reward[:] = 0.0
        self._pending_rendezvous_start_t[:] = -1.0
        self._pending_rendezvous_stop[:] = -1

        self._invalidate_mask()
        self._update_drone_xy()
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------ step

    def step(self, action):
        cfg, sc = self.cfg, self.scenario
        D = cfg.ndrones
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.nagents:
            raise ValueError(f"action has shape {action.shape}, expected ({self.nagents},)")

        self.step_events = {}
        self._step_event_reasons = []
        t_before = float(self.t)

        mask = self._get_action_mask()

        corrected = np.zeros(self.nagents, dtype=np.int64)
        for ag in range(self.nagents):
            a = int(action[ag])
            if a < 0 or a >= self.nactions or int(mask[ag, a]) == 0:
                if a != self.AWAIT or int(mask[ag, self.AWAIT]) == 0:
                    self.episode_invalid_actions += 1
                corrected[ag] = self._first_legal_action(ag, mask)
            else:
                corrected[ag] = a
        action = corrected

        claimed_this_step: Dict[int, int] = {}
        for d in range(D):
            a = int(action[1 + d])
            tid = self._decode_task_id(a)
            if tid < 0:
                continue
            if tid in claimed_this_step:
                self.episode_claim_conflicts += 1
                action[1 + d] = self._first_legal_action(1 + d, mask, forbidden_task_id=tid)
            else:
                claimed_this_step[tid] = d

        truck_a = int(action[0])
        if truck_a != self.AWAIT and self.act_offset_stop <= truck_a < self.act_offset_stop + sc.nstops:
            s = int(truck_a - self.act_offset_stop)
            if int(self.truck_busy[0]) == 0:
                self._start_truck_move_to(s)
            elif int(self._truck_moving[0]) != 0 and int(self.truck_target[0]) != s:
                self._reroute_truck_to(s)

        for d in range(D):
            a = int(action[1 + d])
            if a != self.AWAIT:
                self._apply_drone_action(d, a)

        self.decision_count += 1

        terminated, truncated, extra = self._advance_until_decision()

        dt_elapsed = max(0.0, float(self.t) - t_before)
        reward = self._compute_reward(dt_elapsed)

        if bool(np.all(action == self.AWAIT)):
            self.episode_wait_step_count += 1
            self.episode_wait_step_time_h += float(dt_elapsed)

        obs = self._get_obs()
        info = self._get_info(extra)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass

    # ------------------------------------------------------------------ scenario

    def _precompute_task_stop_dists(self) -> None:
        sc, T = self.scenario, self.scenario.ntasks
        stops = sc.stop_xy
        if T == 0:
            return
        self._min_dist_point_to_stop[:T] = np.min(
            np.linalg.norm(sc.point_xy[:T, None, :] - stops[None, :, :], axis=-1), axis=1)
        self._min_dist_lstart_to_stop[:T] = np.min(
            np.linalg.norm(sc.line_start_xy[:T, None, :] - stops[None, :, :], axis=-1), axis=1)
        self._min_dist_lend_to_stop[:T] = np.min(
            np.linalg.norm(sc.line_end_xy[:T, None, :] - stops[None, :, :], axis=-1), axis=1)

    def _generate_future_task_into_slot(self, tid: int) -> bool:
        """Spawn a future task slot: delegate geometry generation to ScenarioGenerator,
        then apply the returned env-state updates."""
        result = self._scenario_gen.generate_task_slot(
            self.scenario, tid, float(self.t), self.np_random)
        if result is None:
            return False
        self._apply_generated_task_slot(tid, result)
        self._precompute_task_stop_dists()
        return True

    def _spawn_dynamic_task_from_modules(self, tid: int) -> bool:
        for module in self._uncertainty_modules:
            result = module.on_new_task_spawn(self, int(tid))
            if result is not None:
                if bool(result):
                    self._precompute_task_stop_dists()
                return bool(result)
        return self._generate_future_task_into_slot(tid)

    def _apply_generated_task_slot(self, tid: int, result: Dict) -> None:
        """Apply generated future-task geometry to mutable env-state buffers."""
        self._point_service_remain[tid] = float(result["point_service_remain"])
        self._line_remain_xy[tid] = result["line_remain_xy"]
        self._line_remain_npts[tid] = int(result["line_remain_npts"])
        self._line_remain_len[tid] = float(result["line_remain_len"])
        self._line_len[tid] = float(result["line_len"])

    # ------------------------------------------------------------------ uncertainty

    def _init_uncertainty_state(self) -> None:
        self._wind_field = None
        self._uncertainty_modules = build_uncertainty_modules(self.cfg)
        for module in self._uncertainty_modules:
            if hasattr(module, "bind"):
                module.bind(self)
            module.on_reset(self)
        if hasattr(self.reward_model, "on_reset"):
            self.reward_model.on_reset(self)

    def _update_wind(self, dt: float) -> None:
        if dt < self.cfg.eps_time:
            return
        for module in self._uncertainty_modules:
            module.on_time_advance(self, float(dt))

    def _wind_at(self, xy: np.ndarray) -> np.ndarray:
        if self._wind_field is None:
            return np.zeros(2, dtype=np.float64)
        return self._wind_field.get(xy)

    def _effective_flight_time(self, from_xy: np.ndarray, to_xy: np.ndarray,
                               base_speed: Optional[float] = None) -> float:
        cfg = self.cfg
        base_speed = cfg.drone_speed_kmph if base_speed is None else base_speed
        from_xy = np.asarray(from_xy, dtype=np.float64)
        to_xy = np.asarray(to_xy, dtype=np.float64)
        vec = to_xy - from_xy
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            return 0.0
        current = float(dist / base_speed)
        for module in self._uncertainty_modules:
            current = module.flight_time(self, from_xy, to_xy, float(base_speed), current)
        return float(current)

    def _effective_hover_power(self, xy: np.ndarray) -> float:
        current = float(self.cfg.hover_power_per_h)
        for module in self._uncertainty_modules:
            current = module.hover_power(self, xy, current)
        return float(current)

    def _effective_flight_power(
        self,
        from_xy: np.ndarray,
        to_xy: np.ndarray,
        base_power: Optional[float] = None,
    ) -> float:
        base_power = self.cfg.fly_power_per_h if base_power is None else float(base_power)
        current = float(base_power)
        for module in self._uncertainty_modules:
            current = module.flight_power(self, from_xy, to_xy, float(base_power), current)
        return float(current)

    def _flight_energy(
        self,
        from_xy: np.ndarray,
        to_xy: np.ndarray,
        base_speed: Optional[float] = None,
        base_power: Optional[float] = None,
    ) -> float:
        return float(self._effective_flight_time(from_xy, to_xy, base_speed)) * float(
            self._effective_flight_power(from_xy, to_xy, base_power)
        )

    def _apply_battery_shock(self, d: int) -> float:
        shock = 0.0
        for module in self._uncertainty_modules:
            result = module.on_takeoff(self, int(d)) or {}
            shock += float(result.get("battery_shock", 0.0))
        return float(shock)

    def _init_sortie_payload(self, d: int) -> None:
        for module in self._uncertainty_modules:
            module.on_sortie_start(self, int(d))

    def _check_payload_failure(self, d: int) -> bool:
        failed = False
        for module in self._uncertainty_modules:
            result = module.payload_failed(self, int(d))
            if result is not None:
                failed = failed or bool(result)
        return bool(failed)

    # ------------------------------------------------------------------ observation

    def _get_scale_id(self) -> int:
        scales = ["XS", "S", "M", "L", "XL"]
        scale = str(self.scenario.scale)
        if scale in scales:
            return scales.index(scale)
        for i, s in enumerate(scales):
            if scale.startswith(s):
                return i
        return 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        sc, T = self.scenario, self.scenario.ntasks
        cfg = self.cfg
        map_range = sc.map_range
        depot_xy = sc.stop_xy[cfg.depot_stop_id]
        D = cfg.ndrones

        def normalize_pos(xy):
            rel = xy - depot_xy
            return np.clip(rel / (map_range + 1e-9), -1.0, 1.0).astype(np.float32)

        def normalize_01(val, max_val):
            return np.clip(val / (max_val + 1e-9), 0.0, 1.0).astype(np.float32)

        time_progress = normalize_01(self.t, sc.max_time_h)
        done_ratio = normalize_01(np.sum(self.task_done[:T]), T)
        scale_id = np.array([self._get_scale_id()], dtype=np.int8)

        stop_xy = np.zeros((MAX_STOPS, 2), dtype=np.float32)
        stop_xy[:sc.nstops] = normalize_pos(sc.stop_xy[:sc.nstops])
        stop_mask = np.zeros(MAX_STOPS, dtype=np.int8)
        stop_mask[:sc.nstops] = 1

        task_type = np.zeros(MAX_TASKS, dtype=np.int8)
        task_mask = np.zeros(MAX_TASKS, dtype=np.int8)
        point_xy = np.zeros((MAX_TASKS, 2), dtype=np.float32)
        line_start_xy = np.zeros((MAX_TASKS, 2), dtype=np.float32)
        line_end_xy = np.zeros((MAX_TASKS, 2), dtype=np.float32)
        point_service_h = np.zeros(MAX_TASKS, dtype=np.float32)
        task_priority = np.zeros(MAX_TASKS, dtype=np.float32)
        task_deadline_h = np.zeros(MAX_TASKS, dtype=np.float32)
        task_reward_weight = np.zeros(MAX_TASKS, dtype=np.float32)
        task_urgency = np.zeros(MAX_TASKS, dtype=np.float32)
        task_service_factor = np.zeros(MAX_TASKS, dtype=np.float32)
        task_risk = np.zeros(MAX_TASKS, dtype=np.float32)

        if T > 0:
            task_type[:T] = sc.task_type[:T]
            task_mask[:T] = self._task_spawned[:T]
            point_xy[:T] = normalize_pos(sc.point_xy[:T])
            line_start_xy[:T] = normalize_pos(
                np.stack([self._line_start_xy(tid) for tid in range(T)], axis=0))
            line_end_xy[:T] = normalize_pos(
                np.stack([self._line_end_xy(tid) for tid in range(T)], axis=0))
            point_service_h[:T] = normalize_01(self._point_service_obs_array(T), sc.max_time_h)
            task_priority[:T] = np.clip(sc.task_priority[:T] / 3.0, 0.0, 1.0)
            task_deadline_h[:T] = normalize_01(sc.task_deadline_h[:T], sc.max_time_h)
            task_reward_weight[:T] = sc.task_reward_weight[:T].astype(np.float32)
            urgency_ref = max(float(getattr(cfg, "critical_urgency_range", (0.5, 0.9))[1]), 1e-9)
            task_urgency[:T] = np.clip(sc.task_urgency[:T] / urgency_ref, 0.0, 1.0).astype(np.float32)
            task_service_factor[:T] = sc.task_service_factor[:T].astype(np.float32)
            task_risk[:T] = np.clip(sc.task_risk[:T], 0.0, 1.0).astype(np.float32)

        task_done = np.zeros(MAX_TASKS, dtype=np.int8)
        task_claimed = np.zeros(MAX_TASKS, dtype=np.int8)
        task_spawned = np.zeros(MAX_TASKS, dtype=np.int8)
        task_done[:T] = self.task_done[:T]
        task_claimed[:T] = self.task_claimed[:T]
        task_spawned[:T] = self._task_spawned[:T]

        truck_rel_pos = normalize_pos(sc.stop_xy[self.truck_stop[0]])
        truck_busy = self.truck_busy.astype(np.float32)
        truck_eta_norm = normalize_01(self.truck_eta, sc.max_time_h)

        drone_status = self.drone_status.astype(np.int8)
        drone_batt_ratio = normalize_01(self.drone_batt, cfg.max_battery)
        drone_xy = normalize_pos(self._drone_xy)
        drone_eta_norm = normalize_01(self.drone_eta, sc.max_time_h)
        if cfg.uncertainty.enabled and cfg.uncertainty.payload_enabled and cfg.uncertainty.payload_lifetime_h > 1e-9:
            drone_payload_ratio = normalize_01(self._drone_payload_remain, cfg.uncertainty.payload_lifetime_h)
        else:
            drone_payload_ratio = np.ones(D, dtype=np.float32)

        wind_max = max(float(cfg.uncertainty.wind_max_speed), 1e-9)
        wind_vec = np.zeros(2, dtype=np.float32)
        wind_samples = np.zeros((N_WIND_KERNELS, 2), dtype=np.float32)
        wind_kernel_xy = np.zeros((N_WIND_KERNELS, 2), dtype=np.float32)
        if self._wind_field is not None:
            local = self._wind_at(self._drone_xy[0])
            wind_vec = np.clip(local / wind_max, -1.0, 1.0).astype(np.float32)
            kw = self._wind_field.kernel_wind
            kc = self._wind_field.centers
            K = min(self._wind_field.n_kernels, N_WIND_KERNELS)
            wind_samples[:K] = np.clip(kw[:K] / wind_max, -1.0, 1.0).astype(np.float32)
            wind_kernel_xy[:K] = normalize_pos(kc[:K].astype(np.float32))

        return {
            "time_progress": np.array([time_progress], dtype=np.float32),
            "done_ratio": np.array([done_ratio], dtype=np.float32),
            "scale_id": scale_id,
            "stop_xy": stop_xy,
            "stop_mask": stop_mask,
            "task_type": task_type,
            "task_mask": task_mask,
            "point_xy": point_xy,
            "line_start_xy": line_start_xy,
            "line_end_xy": line_end_xy,
            "point_service_h": point_service_h,
            "task_priority": task_priority,
            "task_deadline_h": task_deadline_h,
            "task_reward_weight": task_reward_weight,
            "task_urgency": task_urgency,
            "task_service_factor": task_service_factor,
            "task_risk": task_risk,
            "task_done": task_done,
            "task_claimed": task_claimed,
            "task_spawned": task_spawned,
            "truck_rel_pos": truck_rel_pos.astype(np.float32),
            "truck_stop": self.truck_stop.astype(np.int32),
            "truck_busy": truck_busy,
            "truck_eta_h": truck_eta_norm,
            "truck_target_stop": self.truck_target.astype(np.int32),
            "drone_status": drone_status,
            "drone_stop_id": self.drone_stop_id.astype(np.int32),
            "drone_task_id": self.drone_task_id.astype(np.int32),
            "drone_task_side": self.drone_task_side.astype(np.int8),
            "drone_eta_h": drone_eta_norm,
            "drone_batt_ratio": drone_batt_ratio,
            "drone_xy": drone_xy,
            "drone_payload_remain": drone_payload_ratio,
            "drone_payload_failed": self._drone_payload_failed.astype(np.int8),
            "drone_planned_stop": self.planned_stop.astype(np.int32),
            "response_mode": np.array([1.0 if self._response_mode else 0.0], dtype=np.float32),
            "wind_vec": wind_vec,
            "wind_samples": wind_samples,
            "wind_kernel_xy": wind_kernel_xy,
            "action_mask": self._get_action_mask(),
        }

    def _classify_end(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, object]:
        extra = extra or {}
        reason = str(extra.get("end_reason", ""))
        total = int(self.scenario.ntasks)
        done = int(self.task_done[:total].sum())
        remain = max(0, total - done)
        if reason == "success" and remain == 0:
            cat = "success"
        elif reason == "battery_depleted":
            cat = "energy"
        elif reason == "time_limit":
            cat = "time"
        elif reason == "decision_limit":
            cat = "decision"
        elif reason == "deadlock_drift":
            cat = "deadlock"
        else:
            cat = "other"
        return {"end_reason": reason, "end_category": cat,
                "tasks_total": total, "tasks_done": done, "tasks_remaining": remain,
                "success": bool(reason == "success" and remain == 0)}

    def _get_info(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, object]:
        T = self.scenario.ntasks
        info = {
            "time_h": float(self.t),
            "decision_count": int(self.decision_count),
            "task_done_count": int(self.task_done[:T].sum()),
            "task_spawned_count": int(self._task_spawned[:T].sum()),
            "task_total": int(T),
            "task_total_initial": int(self.scenario.npoint_init + self.scenario.nline_init),
            "completion_rate": float(self.task_done[:T].sum()) / max(float(T), 1.0),
            "invalid_action_count": int(self.episode_invalid_actions),
            "claim_conflict_count": int(self.episode_claim_conflicts),
            "episode_new_task_spawned": int(self.episode_new_task_spawned),
            "overdue_task_count": int(np.sum(
                (self._task_spawned[:T] == 1) & (self.task_done[:T] == 0) &
                (self.scenario.task_deadline_h[:T] > 0) &
                (self.scenario.task_deadline_h[:T] < float(self.t)))),
            "episode_wait_step_count": int(self.episode_wait_step_count),
            "episode_wait_step_time_h": float(self.episode_wait_step_time_h),
            "episode_ground_wait_by_stop_h":
                self.episode_ground_wait_by_stop_h[:self.scenario.nstops].astype(float).tolist(),
        }
        info.update(self._classify_end(extra))
        if extra is not None:
            if "event_reason" in extra:
                info["event_reason"] = extra["event_reason"]
            if "event_reasons" in extra:
                info["event_reasons"] = list(extra["event_reasons"])
        return info

    # ------------------------------------------------------------------ geometry helpers

    def _line_poly_points(self, tid: int) -> np.ndarray:
        n = int(self._line_remain_npts[tid]) if tid < len(self._line_remain_npts) else 0
        if n >= 2:
            return self._line_remain_xy[tid, :n].astype(np.float32)
        return np.asarray([self.scenario.line_start_xy[tid], self.scenario.line_end_xy[tid]],
                          dtype=np.float32)

    def _line_start_xy(self, tid: int) -> np.ndarray:
        return self._line_poly_points(tid)[0].copy()

    def _line_end_xy(self, tid: int) -> np.ndarray:
        return self._line_poly_points(tid)[-1].copy()

    def _point_service_obs_array(self, T: int) -> np.ndarray:
        arr = np.zeros(MAX_TASKS, dtype=np.float32)
        for tid in range(T):
            if int(self.scenario.task_type[tid]) == TASK_POINT:
                arr[tid] = float(self._point_service_remain[tid])
        return arr[:T]

    def _line_entry_exit(self, tid: int, side: int) -> Tuple[np.ndarray, np.ndarray]:
        return (self._line_start_xy(tid), self._line_end_xy(tid)) if side == 0 \
            else (self._line_end_xy(tid), self._line_start_xy(tid))

    def _line_traverse_time(self, tid: int, side: int) -> float:
        pts = self._line_poly_points(tid)
        if side == 1:
            pts = pts[::-1].copy()
        vline = self.cfg.drone_speed_kmph * self.cfg.line_speed_factor
        base = float(sum(self._effective_flight_time(p0, p1, vline)
                         for p0, p1 in zip(pts[:-1], pts[1:])))
        sf = 1.0
        if 0 <= int(tid) < self.scenario.ntasks:
            sf = max(0.5, float(self.scenario.task_service_factor[tid]))
        return float(base * sf)

    def _line_traverse_energy(self, tid: int, side: int) -> float:
        pts = self._line_poly_points(tid)
        if side == 1:
            pts = pts[::-1].copy()
        vline = self.cfg.drone_speed_kmph * self.cfg.line_speed_factor
        base = float(
            sum(
                self._flight_energy(p0, p1, vline, self.cfg.fly_power_per_h)
                for p0, p1 in zip(pts[:-1], pts[1:])
            )
        )
        sf = 1.0
        if 0 <= int(tid) < self.scenario.ntasks:
            sf = max(0.5, float(self.scenario.task_service_factor[tid]))
        return float(base * sf)

    def _payload_time_left(self, d: int) -> float:
        unc = self.cfg.uncertainty
        if not (unc.enabled and unc.payload_enabled and unc.payload_intensity > 0):
            return float('inf')
        return max(0.0, float(self._drone_payload_remain[d]) -
                   (float(self.t) - float(self._drone_sortie_start_t[d])))

    def _set_line_remaining_poly(self, tid: int, pts: np.ndarray) -> None:
        pts = np.asarray(pts, dtype=np.float32)
        n = int(min(len(pts), MAX_LINE_PTS))
        self._line_remain_xy[tid] = 0.0
        self._line_remain_npts[tid] = max(0, n)
        if n > 0:
            self._line_remain_xy[tid, :n] = pts[:n]
        if n >= 2:
            self._line_remain_len[tid] = polyline_length(self._line_remain_xy[tid, :n])
        else:
            self._line_remain_len[tid] = 0.0

    def _polyline_chunk(self, pts: np.ndarray, side: int,
                        time_budget: float) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        pts = np.asarray(pts, dtype=np.float32)
        if side == 1:
            pts_work = pts[::-1].copy()
        else:
            pts_work = pts.copy()
        if len(pts_work) < 2 or time_budget <= self.cfg.eps_time:
            return pts_work[0].copy(), pts.copy(), 0.0, len(pts_work) < 2
        vline = self.cfg.drone_speed_kmph * self.cfg.line_speed_factor
        remain_t = float(time_budget)
        for i in range(len(pts_work) - 1):
            p0 = pts_work[i]
            p1 = pts_work[i + 1]
            seg_t = self._effective_flight_time(p0, p1, vline)
            if seg_t <= self.cfg.eps_time:
                continue
            if remain_t + self.cfg.eps_time < seg_t:
                alpha = remain_t / seg_t
                cut = (p0 + alpha * (p1 - p0)).astype(np.float32)
                rem_oriented = np.vstack([cut, pts_work[i + 1:]]).astype(np.float32)
                rem_original = rem_oriented if side == 0 else rem_oriented[::-1].copy()
                return cut, rem_original, float(time_budget), False
            remain_t -= seg_t
        cut = pts_work[-1].copy()
        return cut, np.asarray([cut], dtype=np.float32), float(time_budget - remain_t), True

    def _schedule_task_execution_chunk(self, d: int, tid: int) -> bool:
        sc = self.scenario
        side = int(self.drone_task_side[d])
        payload_left = self._payload_time_left(d)
        chunk_cap = max(float(self.cfg.exec_check_interval_h), float(self.cfg.eps_time))
        if payload_left <= self.cfg.eps_time:
            self._exec_chunk_dt[d] = 0.0
            return False
        if int(sc.task_type[tid]) == TASK_POINT:
            remain = float(self._point_service_remain[tid])
            dt = min(remain, chunk_cap, payload_left)
            if dt <= self.cfg.eps_time:
                self._exec_chunk_dt[d] = 0.0
                return False
            self._exec_chunk_dt[d] = float(dt)
            pos = sc.point_xy[tid].copy()
            self._drone_xy[d] = pos
            self.drone_status[d] = DS_EXECUTING
            self._drone_exec_is_hover[d] = True
            self._set_segment(d, pos, pos, self.t, self.t + self._exec_chunk_dt[d])
            self._push_event(self.t + self._exec_chunk_dt[d], EV_DRONE_TASK_DONE, d, tid)
            return True
        pts = self._line_poly_points(tid)
        entry, _ = self._line_entry_exit(tid, side)
        total = self._line_traverse_time(tid, side)
        dt_budget = min(total, chunk_cap, payload_left)
        cut_xy, rem_pts, actual_dt, finished = self._polyline_chunk(pts, side, dt_budget)
        if actual_dt <= self.cfg.eps_time:
            self._exec_chunk_dt[d] = 0.0
            return False
        self._exec_chunk_dt[d] = float(actual_dt)
        self._exec_next_line_xy[d] = 0.0
        n = min(len(rem_pts), MAX_LINE_PTS)
        self._exec_next_line_npts[d] = int(n)
        if n > 0:
            self._exec_next_line_xy[d, :n] = rem_pts[:n]
        self.drone_status[d] = DS_EXECUTING
        self._drone_exec_is_hover[d] = False
        self._set_segment(d, entry, cut_xy, self.t, self.t + actual_dt)
        self._push_event(self.t + actual_dt, EV_DRONE_TASK_DONE, d, tid)
        return True

    # ------------------------------------------------------------------ misc helpers

    def _done_ratio(self) -> float:
        return float(np.sum(self.task_done[:self.scenario.ntasks])) / max(
            float(self.scenario.ntasks), 1.0)

    def _all_tasks_done(self) -> bool:
        return bool(np.all(self.task_done[:self.scenario.ntasks]))

    def _task_entry_exit_xy(self, tid: int, side: int) -> Tuple[np.ndarray, np.ndarray]:
        sc = self.scenario
        if int(sc.task_type[tid]) == TASK_POINT:
            p = sc.point_xy[tid].copy()
            return p, p
        return self._line_entry_exit(tid, side)

    def _nearest_stop_flight_time(self, xy: np.ndarray) -> float:
        sc = self.scenario
        best = float('inf')
        for s in range(sc.nstops):
            best = min(best, self._effective_flight_time(xy, sc.stop_xy[s]))
        return 0.0 if not np.isfinite(best) else float(best)

    def _nearest_stop_flight_energy(self, xy: np.ndarray) -> float:
        sc = self.scenario
        best = float('inf')
        for s in range(sc.nstops):
            best = min(best, self._flight_energy(xy, sc.stop_xy[s]))
        return 0.0 if not np.isfinite(best) else float(best)

    def _task_local_density(self, ref_xy: np.ndarray, radius_km: float) -> float:
        sc = self.scenario
        cnt = 0.0
        for tid in range(sc.ntasks):
            if int(self.task_done[tid]) == 1 or int(self.task_claimed[tid]) == 1:
                continue
            if int(self._task_spawned[tid]) == 0:
                continue
            p = sc.point_xy[tid] if int(sc.task_type[tid]) == TASK_POINT \
                else 0.5 * (self._line_start_xy(tid) + self._line_end_xy(tid))
            if float(np.linalg.norm(p - ref_xy)) <= float(radius_km):
                cnt += 1.0
        return float(cnt)

    def _set_reward_reference_scales(self) -> None:
        scale = str(self.scenario.scale)
        self.time_ref_h = float(SCALE_TIME_REFS_H.get(
            scale, max(float(self.scenario.max_time_h) * 0.5, 1.0)))

    def _norm_global_time(self, value_h: float) -> float:
        return float(value_h) / max(float(self.time_ref_h), 1e-6)

    # ------------------------------------------------------------------ action mask

    def _invalidate_mask(self) -> None:
        self._mask_dirty = True

    def _build_task_action_map(self) -> None:
        sc = self.scenario
        k = 0
        self._task_action_tid[:] = -1
        self._task_action_side[:] = 0
        if sc is None:
            self._task_action_count = 0
            return
        for tid in range(sc.ntasks):
            if int(self._task_spawned[tid]) == 0:
                continue
            self._task_action_tid[k] = tid
            self._task_action_side[k] = 0
            k += 1
            if int(sc.task_type[tid]) == TASK_LINE:
                self._task_action_tid[k] = tid
                self._task_action_side[k] = 1
                k += 1
        self._task_action_count = k

    def _task_action_id(self, tid: int, side: int) -> int:
        for k in range(self._task_action_count):
            if int(self._task_action_tid[k]) == int(tid) and int(self._task_action_side[k]) == int(side):
                return self.act_offset_task + k
        return -1

    def _decode_task_id(self, a: int) -> int:
        if self.act_offset_task <= a < self.act_offset_task + self._task_action_count:
            return int(self._task_action_tid[a - self.act_offset_task])
        return -1

    def _decode_task_id_and_side(self, a: int) -> Tuple[int, int]:
        if self.act_offset_task <= a < self.act_offset_task + self._task_action_count:
            idx = a - self.act_offset_task
            return int(self._task_action_tid[idx]), int(self._task_action_side[idx])
        return -1, 0

    def _request_joint_replan(self, reason: str) -> None:
        self._joint_replan_requested = True
        self._replan_reason = str(reason)

    def _clear_replan_request(self) -> None:
        self._joint_replan_requested = False
        self._replan_reason = ""

    def _append_step_event_reason(self, reason: str) -> None:
        reason = str(reason)
        if reason and reason not in self._step_event_reasons:
            self._step_event_reasons.append(reason)

    def _drone_is_idle(self, d: int) -> bool:
        return int(self.drone_status[d]) in (DS_ON_TRUCK, DS_WAITING)

    def _truck_travel_time(self, s_from: int, s_to: int) -> float:
        return float(self.scenario.truck_dist[int(s_from), int(s_to)]) / float(self.cfg.truck_speed_kmph)

    def _can_waiting_drone_reach_stop(self, d: int, s: int) -> bool:
        if int(self.drone_status[d]) != DS_WAITING:
            return False
        cfg = self.cfg
        e_need = self._flight_energy(self._drone_xy[d], self.scenario.stop_xy[int(s)])
        e_need += cfg.fly_power_per_h * cfg.landing_time_h + cfg.safety_batt_margin
        return float(self.drone_batt[d]) + cfg.eps_batt >= e_need

    def _task_action_feasible(self, d: int, tid: int, side: int) -> bool:
        cfg, sc = self.cfg, self.scenario
        if not (0 <= tid < sc.ntasks):
            return False
        if int(self._task_spawned[tid]) == 0:
            return False
        if self.task_done[tid] or self.task_claimed[tid] or self._drone_payload_failed[d]:
            return False
        status = int(self.drone_status[d])
        if status not in (DS_ON_TRUCK, DS_WAITING):
            return False
        if status == DS_ON_TRUCK:
            if float(self.drone_eta[d]) > float(self.t) + cfg.eps_time:
                return False
            if int(self._truck_moving[0]) != 0:
                return False
        from_xy = self._drone_xy[d].copy()
        takeoff_h = cfg.takeoff_time_h if status == DS_ON_TRUCK else 0.0
        batt = float(self.drone_batt[d])
        if int(sc.task_type[tid]) == TASK_POINT:
            e_need = cfg.fly_power_per_h * takeoff_h
            e_need += self._flight_energy(from_xy, sc.point_xy[tid])
            e_need += self._effective_hover_power(sc.point_xy[tid]) * float(self._point_service_remain[tid])
            e_need += self._nearest_stop_flight_energy(sc.point_xy[tid])
            e_need += cfg.fly_power_per_h * cfg.landing_time_h
            e_need += cfg.safety_batt_margin
            return batt + cfg.eps_batt >= e_need
        entry, _ = self._line_entry_exit(tid, side)
        exit_xy = self._line_end_xy(tid) if side == 0 else self._line_start_xy(tid)
        e_need = cfg.fly_power_per_h * takeoff_h
        e_need += self._flight_energy(from_xy, entry)
        e_need += self._line_traverse_energy(tid, side)
        e_need += self._nearest_stop_flight_energy(exit_xy)
        e_need += cfg.fly_power_per_h * cfg.landing_time_h
        e_need += cfg.safety_batt_margin
        return batt + cfg.eps_batt >= e_need

    def _waiting_drone_has_legal_action(self, d: int) -> bool:
        if int(self.drone_status[d]) != DS_WAITING:
            return True
        sc = self.scenario
        for s in range(sc.nstops):
            if self._can_waiting_drone_reach_stop(d, s):
                return True
        for tid in range(sc.ntasks):
            if int(self._task_spawned[tid]) == 0:
                continue
            if self._task_action_feasible(d, tid, 0):
                return True
            if int(sc.task_type[tid]) == TASK_LINE and self._task_action_feasible(d, tid, 1):
                return True
        return False

    # ------------------------------------------------------------------ rendezvous / truck coordination

    def _rendezvous_candidates(self) -> List[int]:
        return sorted({int(s) for s in self.planned_stop.tolist()
                       if 0 <= int(s) < self.scenario.nstops})

    def _truck_eta_to_stop(self, s: int) -> float:
        s = int(s)
        if int(self._truck_moving[0]) != 0:
            remain = max(0.0, float(self.truck_eta[0]) - float(self.t))
            return remain + self._truck_travel_time(int(self.truck_target[0]), s)
        if int(self.truck_busy[0]) != 0:
            remain = max(0.0, float(self.truck_eta[0]) - float(self.t))
            return remain + self._truck_travel_time(int(self.truck_stop[0]), s)
        return self._truck_travel_time(int(self.truck_stop[0]), s)

    def _truck_current_xy(self) -> np.ndarray:
        sc = self.scenario
        if int(self._truck_moving[0]) == 0:
            return sc.stop_xy[int(self.truck_stop[0])].copy()
        src = int(self._truck_move_from_stop[0])
        dst = int(self.truck_target[0])
        src_xy = sc.stop_xy[src]
        dst_xy = sc.stop_xy[dst]
        total = self._truck_travel_time(src, dst)
        if total <= self.cfg.eps_time:
            return dst_xy.copy()
        elapsed = max(0.0, float(self.t) - float(self._truck_move_start_t[0]))
        alpha = float(np.clip(elapsed / total, 0.0, 1.0))
        return (src_xy + alpha * (dst_xy - src_xy)).astype(np.float32)

    def _truck_travel_time_from_xy(self, from_xy: np.ndarray, s: int) -> float:
        to_xy = self.scenario.stop_xy[int(s)]
        delta = np.asarray(to_xy, dtype=np.float32) - np.asarray(from_xy, dtype=np.float32)
        dist = float(np.linalg.norm(delta)) if self.cfg.road_network_type == "euclidean" \
            else float(np.abs(delta).sum())
        return dist / max(self.cfg.truck_speed_kmph, 1e-9)

    def _reroute_truck_to(self, s: int) -> bool:
        s = int(s)
        if int(self._truck_moving[0]) == 0:
            return False
        cur_xy = self._truck_current_xy()
        ttravel = self._truck_travel_time_from_xy(cur_xy, s)
        self.truck_busy[0] = 1
        self.truck_eta[0] = float(self.t + ttravel)
        self.truck_target[0] = s
        self._truck_move_from_stop[0] = int(self.truck_stop[0])
        self._truck_move_start_t[0] = float(self.t)
        self._truck_arrival_seq[0] = int(self._truck_arrival_seq[0]) + 1
        self._push_event(self.t + ttravel, EV_TRUCK_ARRIVE_STOP, s, int(self._truck_arrival_seq[0]))
        self._invalidate_mask()
        return True

    def _truck_should_wait_for_inbound(self, s: int) -> bool:
        s = int(s)
        for d in range(self.cfg.ndrones):
            st = int(self.drone_status[d])
            if st in (DS_CRUISING, DS_LANDING) and int(self.drone_stop_id[d]) == s:
                return True
        return False

    def _stop_pending_rendezvous_drones(self, s: int) -> List[int]:
        s = int(s)
        drones: List[int] = []
        for d in range(self.cfg.ndrones):
            if int(self.planned_stop[d]) != s:
                continue
            st = int(self.drone_status[d])
            if st == DS_ON_TRUCK:
                continue
            if (st == DS_ON_GROUND and int(self.drone_stop_id[d]) == s) or \
               (st in (DS_CRUISING, DS_LANDING) and int(self.drone_stop_id[d]) == s):
                drones.append(d)
        return drones

    def _stop_has_pending_rendezvous(self, s: int) -> bool:
        return len(self._stop_pending_rendezvous_drones(int(s))) > 0

    def _queue_maintenance_stop(self, s: int) -> None:
        s = int(s)
        if s < 0 or s >= self.scenario.nstops:
            return
        if s == int(self._active_rendezvous_stop):
            return
        if s not in self._maintenance_queue:
            self._maintenance_queue.append(s)

    def _prune_maintenance_queue(self) -> None:
        kept: List[int] = []
        seen = set()
        for s in self._maintenance_queue:
            s = int(s)
            if s in seen:
                continue
            if not self._stop_has_pending_rendezvous(s):
                continue
            kept.append(s)
            seen.add(s)
        self._maintenance_queue = kept

    def _drone_ready_time_for_stop(self, d: int, s: int) -> float:
        s = int(s)
        st = int(self.drone_status[d])
        if st == DS_ON_GROUND and int(self.drone_stop_id[d]) == s:
            return float(self.t)
        if st in (DS_CRUISING, DS_LANDING) and int(self.drone_stop_id[d]) == s:
            return float(self.drone_eta[d])
        return float('inf')

    def _project_stop_service_finish_from(self, from_xy: np.ndarray, start_t: float, s: int) -> float:
        s = int(s)
        drones = self._stop_pending_rendezvous_drones(s)
        if not drones:
            return float(start_t)
        t_arr = float(start_t) + self._truck_travel_time_from_xy(from_xy, s)
        ready_times = sorted(self._drone_ready_time_for_stop(d, s) for d in drones)
        cur_t = float(t_arr)
        for rt in ready_times:
            cur_t = max(cur_t, float(rt))
            cur_t += float(self.cfg.swap_time_h)
        return float(cur_t)

    def _select_best_maintenance_stop(self) -> int:
        self._prune_maintenance_queue()
        candidates = [int(s) for s in self._maintenance_queue
                      if self._stop_has_pending_rendezvous(int(s))]
        if not candidates:
            return -1
        from_xy = self._truck_current_xy()
        start_t = float(self.t)
        best_s, best_key = -1, None
        for s in candidates:
            finish_t = self._project_stop_service_finish_from(from_xy, start_t, s)
            travel_t = self._truck_travel_time_from_xy(from_xy, s)
            key = (finish_t, travel_t, s)
            if best_key is None or key < best_key:
                best_key = key
                best_s = s
        return int(best_s)

    def _ensure_active_rendezvous_progress(self) -> bool:
        self._prune_maintenance_queue()
        if 0 <= int(self._active_rendezvous_stop) < self.scenario.nstops:
            if not self._stop_has_pending_rendezvous(int(self._active_rendezvous_stop)):
                self._active_rendezvous_stop = -1

        if int(self._active_rendezvous_stop) < 0:
            next_s = self._select_best_maintenance_stop()
            if next_s < 0:
                if int(self._truck_moving[0]) == 0 and int(self._truck_pending_swaps) == 0:
                    self.truck_busy[0] = 0
                    self.truck_eta[0] = float(self.t)
                self._invalidate_mask()
                return False
            self._active_rendezvous_stop = int(next_s)
            self._maintenance_queue = [int(s) for s in self._maintenance_queue
                                       if int(s) != int(next_s)]

        s = int(self._active_rendezvous_stop)

        if int(self._truck_moving[0]) != 0:
            if int(self.truck_target[0]) != s:
                self._reroute_truck_to(s)
            self.truck_busy[0] = 1
            self._invalidate_mask()
            return True

        cur_s = int(self.truck_stop[0])
        if cur_s != s:
            if int(self._truck_pending_swaps) > 0 or int(self.truck_busy[0]) != 0:
                self._invalidate_mask()
                return True
            self._start_truck_move_to(s)
            self._invalidate_mask()
            return True

        waiting = [d for d in self._stop_pending_rendezvous_drones(s)
                   if int(self.drone_status[d]) == DS_ON_GROUND]
        if waiting:
            for d in waiting:
                self._start_swap(d)
            self._invalidate_mask()
            return True

        inbound_eta = [
            float(self.drone_eta[d])
            for d in self._stop_pending_rendezvous_drones(s)
            if int(self.drone_status[d]) in (DS_CRUISING, DS_LANDING)
        ]
        if inbound_eta:
            self.truck_busy[0] = 1
            self.truck_eta[0] = min(inbound_eta)
            self._truck_pending_swaps = 0
            self._invalidate_mask()
            return True

        self._active_rendezvous_stop = -1
        self._invalidate_mask()
        return self._ensure_active_rendezvous_progress()

    def _register_stop_request(self, d: int, s: int) -> bool:
        s = int(s)
        if s < 0 or s >= self.scenario.nstops:
            return False
        if int(self._active_rendezvous_stop) < 0:
            self._active_rendezvous_stop = s
        elif s != int(self._active_rendezvous_stop):
            self._queue_maintenance_stop(s)
        started = self._ensure_active_rendezvous_progress()
        self._invalidate_mask()
        return started

    def _queue_or_start_truck_response(self, s: int) -> bool:
        if s is None or int(s) < 0:
            return False
        self._queue_maintenance_stop(int(s))
        return self._ensure_active_rendezvous_progress()

    def _ground_waiting_drone_stops(self) -> List[int]:
        stops = set()
        for d in range(self.cfg.ndrones):
            if int(self.drone_status[d]) == DS_ON_GROUND:
                s = int(self.drone_stop_id[d])
                if 0 <= s < self.scenario.nstops:
                    stops.add(s)
        return sorted(stops)

    def _compute_action_mask(self) -> np.ndarray:
        cfg, sc, D = self.cfg, self.scenario, self.cfg.ndrones
        S, T = sc.nstops, sc.ntasks
        self._update_drone_xy()
        mask = np.zeros((1 + D, self.nactions), dtype=np.int8)
        mask[:, self.AWAIT] = 1

        if self._response_mode:
            mask[0, :] = 0
            if int(self._truck_moving[0]) != 0:
                mask[0, self.AWAIT] = 1
            for s in self._rendezvous_candidates():
                if int(self._truck_moving[0]) != 0 and int(s) == int(self.truck_target[0]):
                    continue
                mask[0, int(s)] = 1
            return mask

        mask[0, self.AWAIT] = 1
        all_tasks_done = bool(self._all_tasks_done())
        waiting_stops = self._ground_waiting_drone_stops()
        all_drones_on_truck = all(int(self.drone_status[d]) == DS_ON_TRUCK for d in range(D)) \
            if all_tasks_done else False

        if int(self.truck_busy[0]) == 0:
            cur = int(self.truck_stop[0])
            if all_tasks_done and all_drones_on_truck:
                if cur != int(cfg.depot_stop_id):
                    mask[0, int(cfg.depot_stop_id)] = 1
                    mask[0, self.AWAIT] = 0
            elif all_tasks_done and waiting_stops:
                for s in waiting_stops:
                    if 0 <= s < S and s != cur:
                        mask[0, s] = 1
                if not np.any(mask[0, :self.AWAIT]):
                    mask[0, self.AWAIT] = 1
                else:
                    mask[0, self.AWAIT] = 0
            else:
                for s in range(S):
                    if s != cur:
                        mask[0, s] = 1

        for d in range(D):
            status = int(self.drone_status[d])
            if not self._drone_is_idle(d):
                mask[1 + d, self.AWAIT] = 1
                continue
            if status == DS_ON_TRUCK:
                mask[1 + d, self.AWAIT] = 1
            elif status == DS_WAITING:
                mask[1 + d, self.AWAIT] = 0
                for s in range(S):
                    if self._can_waiting_drone_reach_stop(d, s):
                        mask[1 + d, s] = 1
            else:
                mask[1 + d, self.AWAIT] = 0
            for k in range(self._task_action_count):
                tid = int(self._task_action_tid[k])
                side = int(self._task_action_side[k])
                if 0 <= tid < T and self._task_action_feasible(d, tid, side):
                    mask[1 + d, self.act_offset_task + k] = 1
        return mask

    def _get_action_mask(self) -> np.ndarray:
        if self._mask_dirty or self._cached_mask is None:
            self._cached_mask = self._compute_action_mask()
            self._mask_dirty = False
        return self._cached_mask

    def _first_legal_action(self, agent_idx: int, mask: np.ndarray, forbidden_task_id: int = -1) -> int:
        row = mask[agent_idx]
        if forbidden_task_id >= 0:
            row = row.copy()
            for a in range(self.nactions):
                if row[a] and self._decode_task_id(a) == forbidden_task_id:
                    row[a] = 0
        legal = np.flatnonzero(row)
        if legal.size == 0:
            return self.AWAIT
        return int(legal[0])

    def _task_action_cost(self, d: int, a: int) -> float:
        tid, side = self._decode_task_id_and_side(int(a))
        if tid < 0 or tid >= self.scenario.ntasks:
            return float('inf')
        self._update_drone_xy()
        cur_xy = self._drone_xy[d].copy()
        target = self.scenario.point_xy[tid].copy() if int(self.scenario.task_type[tid]) == TASK_POINT \
            else (self._line_start_xy(tid) if int(side) == 0 else self._line_end_xy(tid))
        return float(self._effective_flight_time(cur_xy, target))

    # ------------------------------------------------------------------ event system

    def _push_event(self, time_h: float, ev_type: int, a: int, b: int) -> None:
        self._ev_counter += 1
        heapq.heappush(self.evq, (float(time_h), self._ev_counter, ev_type, a, b))

    def _set_segment(self, d: int, frm: np.ndarray, to: np.ndarray, t0: float, t1: float) -> None:
        self._seg_active[d] = 1
        self._seg_from_xy[d] = np.asarray(frm, dtype=np.float32)
        self._seg_to_xy[d] = np.asarray(to, dtype=np.float32)
        self._seg_t_start[d] = float(t0)
        self._seg_t_end[d] = float(t1)

    def _clear_segment(self, d: int) -> None:
        self._seg_active[d] = 0

    def _update_drone_xy(self) -> None:
        sc = self.scenario
        for d in range(self.cfg.ndrones):
            st = int(self.drone_status[d])
            if st == DS_ON_TRUCK:
                self._drone_xy[d] = sc.stop_xy[int(self.truck_stop[0])]
            elif st == DS_ON_GROUND:
                self._drone_xy[d] = sc.stop_xy[int(self.drone_stop_id[d])]
            elif st in (DS_TAKING_OFF, DS_CRUISING, DS_LANDING, DS_EXECUTING) and self._seg_active[d]:
                t0, t1 = float(self._seg_t_start[d]), float(self._seg_t_end[d])
                if abs(t1 - t0) < 1e-9:
                    self._drone_xy[d] = self._seg_to_xy[d]
                else:
                    alpha = float(np.clip((self.t - t0) / (t1 - t0), 0.0, 1.0))
                    self._drone_xy[d] = (self._seg_from_xy[d] +
                                          alpha * (self._seg_to_xy[d] - self._seg_from_xy[d])
                                         ).astype(np.float32)

    def _integrate_energy(self, dt: float) -> Tuple[bool, str]:
        cfg = self.cfg
        if dt <= cfg.eps_time:
            return True, ""
        self._update_wind(dt)
        for d in range(cfg.ndrones):
            st = int(self.drone_status[d])
            if st in (DS_ON_TRUCK, DS_ON_GROUND):
                continue
            if st == DS_WAITING:
                p = self._effective_hover_power(self._drone_xy[d])
            elif st == DS_EXECUTING:
                p = self._effective_hover_power(self._drone_xy[d]) \
                    if bool(self._drone_exec_is_hover[d]) else self._effective_flight_power(
                        self._seg_from_xy[d], self._seg_to_xy[d]
                    )
            else:
                p = self._effective_flight_power(self._seg_from_xy[d], self._seg_to_xy[d])
            energy = float(p) * float(dt)
            self.drone_batt[d] -= energy
            self.step_events["energy_used"] = (
                float(self.step_events.get("energy_used", 0.0)) + float(energy)
            )
        return (False, "battery_depleted") if np.any(self.drone_batt < -cfg.eps_batt) else (True, "")

    # ------------------------------------------------------------------ actions

    def _claim_task(self, d: int, tid: int) -> None:
        self.task_claimed[tid] = 1
        self.drone_task_id[d] = int(tid)
        self._invalidate_mask()

    def _release_task_claim(self, d: int) -> None:
        tid = int(self.drone_task_id[d])
        if tid >= 0 and not self.task_done[tid]:
            self.task_claimed[tid] = 0
        self.drone_task_id[d] = -1
        self._clear_pending_task_reward(d)
        self._invalidate_mask()

    def _start_truck_move_to(self, s: int) -> bool:
        if int(self.truck_busy[0]) != 0 or int(s) == int(self.truck_stop[0]):
            return False
        ttravel = self._truck_travel_time(int(self.truck_stop[0]), int(s))
        self.truck_busy[0] = 1
        self.truck_eta[0] = float(self.t + ttravel)
        self.truck_target[0] = int(s)
        self._truck_moving[0] = 1
        self._truck_move_from_stop[0] = int(self.truck_stop[0])
        self._truck_move_start_t[0] = float(self.t)
        self._truck_arrival_seq[0] = int(self._truck_arrival_seq[0]) + 1
        self._push_event(self.t + ttravel, EV_TRUCK_ARRIVE_STOP, int(s),
                         int(self._truck_arrival_seq[0]))
        self._invalidate_mask()
        return True

    def _apply_drone_action(self, d: int, a: int) -> None:
        cfg, sc = self.cfg, self.scenario
        status = int(self.drone_status[d])
        if status == DS_ON_GROUND:
            return
        on_truck = (status == DS_ON_TRUCK)
        in_air = (status == DS_WAITING)
        cur_xy = self._drone_xy[d].copy()
        takeoff_h = cfg.takeoff_time_h if on_truck else 0.0

        if self.act_offset_stop <= a < self.act_offset_stop + sc.nstops:
            if on_truck:
                return
            s = int(a)
            self.planned_stop[d] = s
            self._pending_rendezvous_start_t[d] = float(self.t)
            self._pending_rendezvous_stop[d] = s
            self._append_step_event_reason("drone_selected_stop")
            target = sc.stop_xy[s].copy()
            t_cr_end = self.t + self._effective_flight_time(cur_xy, target)
            t_ld_end = t_cr_end + cfg.landing_time_h
            self.drone_stop_id[d] = s
            self.drone_eta[d] = float(t_ld_end)
            if bool(self._drone_payload_failed[d]):
                self.step_events["payload_recover_count"] = \
                    float(self.step_events.get("payload_recover_count", 0.0)) + 1.0
            if in_air:
                self.drone_status[d] = DS_CRUISING
                self._set_segment(d, cur_xy, target, self.t, t_cr_end)
                self._push_event(t_cr_end, EV_DRONE_ARRIVE_STOP, d, s)
            self._register_stop_request(d, s)
            self._invalidate_mask()
            return

        tid, side = self._decode_task_id_and_side(a)
        if not (0 <= tid < sc.ntasks):
            return

        self.planned_stop[d] = -1
        self._pending_rendezvous_start_t[d] = -1.0
        self._pending_rendezvous_stop[d] = -1
        self._clear_pending_task_reward(d)
        self._prepare_task_reward(d, tid, side)
        self._claim_task(d, tid)
        self.drone_task_side[d] = int(side)

        if sc.task_type[tid] == TASK_POINT:
            target = sc.point_xy[tid].copy()
        else:
            target = self._line_start_xy(tid) if side == 0 else self._line_end_xy(tid)
        t_tk_end = self.t + takeoff_h
        t_cr_end = t_tk_end + self._effective_flight_time(cur_xy, target)
        self.drone_eta[d] = float(t_cr_end)

        if on_truck:
            self.drone_status[d] = DS_TAKING_OFF
            self._set_segment(d, cur_xy, cur_xy, self.t, t_tk_end)
            self._push_event(t_tk_end, EV_DRONE_TAKEOFF_DONE, d, -(tid + 1))
        elif in_air:
            self.drone_status[d] = DS_CRUISING
            self._set_segment(d, cur_xy, target, self.t, t_cr_end)
            self._push_event(t_cr_end, EV_DRONE_ARRIVE_TASK, d, tid)
        self._invalidate_mask()

    # ------------------------------------------------------------------ event handlers

    def _start_swap(self, d: int) -> None:
        self.truck_busy[0] = 1
        self.drone_status[d] = DS_ON_TRUCK
        if float(self._pending_rendezvous_start_t[d]) >= 0.0:
            rendezvous_h = max(0.0, float(self.t) - float(self._pending_rendezvous_start_t[d]))
            self.step_events["rendezvous_reward"] = (
                float(self.step_events.get("rendezvous_reward", 0.0)) - 2.0 * rendezvous_h)
            self._pending_rendezvous_start_t[d] = -1.0
            self._pending_rendezvous_stop[d] = -1
        t_end = self.t + self.cfg.swap_time_h
        self.drone_eta[d] = float(t_end)
        self._truck_pending_swaps += 1
        self._push_event(t_end, EV_SWAP_DONE, d, 0)
        self._invalidate_mask()

    def _handle_event(self, ev_type: int, a: int, b: int) -> None:
        cfg, sc = self.cfg, self.scenario

        if ev_type == EV_NEW_TASK_SPAWN:
            tid = int(a)
            if not (0 <= tid < sc.ntasks):
                return
            if int(self._task_spawned[tid]) == 1:
                return
            if cfg.dynamic_generate_online:
                if not self._spawn_dynamic_task_from_modules(tid):
                    return
            self._task_spawned[tid] = 1
            self.episode_new_task_spawned += 1
            self.step_events["new_task_spawned"] = \
                float(self.step_events.get("new_task_spawned", 0.0)) + 1.0
            self._append_step_event_reason("new_task_spawned")
            self._build_task_action_map()
            self._request_joint_replan("new_task_spawned")
            self._invalidate_mask()
            return

        if ev_type == EV_TRUCK_ARRIVE_STOP:
            s, seq = int(a), int(b)
            if seq != int(self._truck_arrival_seq[0]):
                return
            self.truck_stop[0] = s
            self.truck_target[0] = s
            self.truck_eta[0] = float(self.t)
            self._truck_moving[0] = 0
            waiting = [d for d in range(cfg.ndrones)
                       if int(self.drone_status[d]) == DS_ON_GROUND and int(self.drone_stop_id[d]) == s]
            if waiting:
                for d in waiting:
                    self._start_swap(d)
            elif self._truck_should_wait_for_inbound(s):
                inbound_eta = [float(self.drone_eta[d]) for d in range(cfg.ndrones)
                               if int(self.drone_status[d]) in (DS_CRUISING, DS_LANDING) and
                               int(self.drone_stop_id[d]) == s]
                self.truck_busy[0] = 1
                self.truck_eta[0] = min(inbound_eta) if inbound_eta else float(self.t)
                self._truck_pending_swaps = 0
            else:
                self.truck_busy[0] = 0
                self._truck_pending_swaps = 0
                self._ensure_active_rendezvous_progress()
            self._invalidate_mask()
            return

        if ev_type == EV_DRONE_TAKEOFF_DONE:
            d, dst = int(a), int(b)
            if int(self.drone_status[d]) != DS_TAKING_OFF:
                return
            self._clear_segment(d)
            cur_xy = self._drone_xy[d].copy()
            shock = self._apply_battery_shock(d)
            self._init_sortie_payload(d)
            if shock > self.cfg.eps_batt:
                self.step_events["battery_shock_count"] = \
                    float(self.step_events.get("battery_shock_count", 0.0)) + 1.0
                self._append_step_event_reason("battery_shock")
                self._request_joint_replan("battery_shock")
            if dst >= 0:
                target = sc.stop_xy[dst].copy()
                t_end = self.t + self._effective_flight_time(cur_xy, target)
                self.drone_status[d] = DS_CRUISING
                self._set_segment(d, cur_xy, target, self.t, t_end)
                self._push_event(t_end, EV_DRONE_ARRIVE_STOP, d, dst)
            else:
                tid = -dst - 1
                side = int(self.drone_task_side[d])
                target = sc.point_xy[tid].copy() if sc.task_type[tid] == TASK_POINT \
                    else (self._line_start_xy(tid) if side == 0 else self._line_end_xy(tid))
                t_end = self.t + self._effective_flight_time(cur_xy, target)
                self.drone_status[d] = DS_CRUISING
                self._set_segment(d, cur_xy, target, self.t, t_end)
                self._push_event(t_end, EV_DRONE_ARRIVE_TASK, d, tid)
            self._invalidate_mask()
            return

        if ev_type == EV_DRONE_ARRIVE_STOP:
            d, s = int(a), int(b)
            if int(self.drone_status[d]) != DS_CRUISING:
                return
            self._clear_segment(d)
            target = sc.stop_xy[s].copy()
            self._drone_xy[d] = target
            t_end = self.t + cfg.landing_time_h
            self.drone_status[d] = DS_LANDING
            self._set_segment(d, target, target, self.t, t_end)
            self._push_event(t_end, EV_DRONE_LAND_DONE, d, s)
            self._invalidate_mask()
            return

        if ev_type == EV_DRONE_LAND_DONE:
            d, s = int(a), int(b)
            if int(self.drone_status[d]) != DS_LANDING:
                return
            self._clear_segment(d)
            self._release_task_claim(d)
            self._drone_xy[d] = sc.stop_xy[s].copy()
            self.drone_stop_id[d] = s
            self.drone_status[d] = DS_ON_GROUND
            if int(self.truck_stop[0]) == s:
                self._start_swap(d)
            else:
                self._ensure_active_rendezvous_progress()
            self._invalidate_mask()
            return

        if ev_type == EV_DRONE_ARRIVE_TASK:
            d, tid = int(a), int(b)
            if int(self.drone_status[d]) != DS_CRUISING:
                return
            self._clear_segment(d)
            self.drone_task_id[d] = tid
            if self._check_payload_failure(d):
                if sc.task_type[tid] == TASK_POINT:
                    self._drone_xy[d] = sc.point_xy[tid].copy()
                else:
                    self._drone_xy[d] = self._line_start_xy(tid) \
                        if int(self.drone_task_side[d]) == 0 else self._line_end_xy(tid)
                self.task_claimed[tid] = 0
                self.task_done[tid] = 0
                self.drone_task_id[d] = -1
                self._clear_pending_task_reward(d)
                self.drone_status[d] = DS_WAITING
                self._drone_payload_failed[d] = True
                self.step_events["payload_failure_count"] = \
                    float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
                self._append_step_event_reason("payload_failure")
                self._request_joint_replan("payload_failure")
                self._invalidate_mask()
                return
            if not self._schedule_task_execution_chunk(d, tid):
                self.task_claimed[tid] = 0
                self.drone_task_id[d] = -1
                self.drone_status[d] = DS_WAITING
                self._drone_exec_is_hover[d] = False
                self._drone_payload_failed[d] = True
                self._clear_pending_task_reward(d)
                self.step_events["payload_failure_count"] = \
                    float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
                self._append_step_event_reason("payload_failure")
                self._request_joint_replan("payload_failure")
            self._invalidate_mask()
            return

        if ev_type == EV_DRONE_TASK_DONE:
            d, tid = int(a), int(b)
            if int(self.drone_status[d]) != DS_EXECUTING:
                return
            self._clear_segment(d)
            dt_exec = float(self._exec_chunk_dt[d])
            finished = False
            if int(sc.task_type[tid]) == TASK_POINT:
                self._drone_xy[d] = sc.point_xy[tid].copy()
                self._point_service_remain[tid] = max(
                    0.0, float(self._point_service_remain[tid]) - dt_exec)
                finished = float(self._point_service_remain[tid]) <= self.cfg.eps_time
            else:
                self._drone_xy[d] = self._seg_to_xy[d].copy()
                n = int(self._exec_next_line_npts[d])
                rem = self._exec_next_line_xy[d, :n].copy() if n > 0 \
                    else np.asarray([self._drone_xy[d].copy()], dtype=np.float32)
                self._set_line_remaining_poly(tid, rem)
                finished = (float(self._line_remain_len[tid]) <= self.cfg.eps_time or
                            int(self._line_remain_npts[tid]) <= 1)

            payload_failed_now = self._check_payload_failure(d)
            if finished:
                self.task_done[tid] = 1
                self.task_claimed[tid] = 0
                self.drone_task_id[d] = -1
                self.drone_status[d] = DS_WAITING
                self._drone_exec_is_hover[d] = False
                self.step_events["task_reward"] = \
                    float(self.step_events.get("task_reward", 0.0)) + float(self._pending_task_reward[d])
                self.step_events["task_reward"] = \
                    float(self.step_events.get("task_reward", 0.0)) + self._task_deadline_adjustment(tid, self.t)
                self._clear_pending_task_reward(d)
            elif payload_failed_now:
                self.task_claimed[tid] = 0
                self.drone_task_id[d] = -1
                self.drone_status[d] = DS_WAITING
                self._drone_exec_is_hover[d] = False
                self._drone_payload_failed[d] = True
                self._clear_pending_task_reward(d)
                self.step_events["payload_failure_count"] = \
                    float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
                self._append_step_event_reason("payload_failure")
                self._request_joint_replan("payload_failure")
            else:
                if not self._schedule_task_execution_chunk(d, tid):
                    self.task_claimed[tid] = 0
                    self.drone_task_id[d] = -1
                    self.drone_status[d] = DS_WAITING
                    self._drone_exec_is_hover[d] = False
                    self._drone_payload_failed[d] = True
                    self._clear_pending_task_reward(d)
                    self.step_events["payload_failure_count"] = \
                        float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
                    self._append_step_event_reason("payload_failure")
                    self._request_joint_replan("payload_failure")
            self._invalidate_mask()
            return

        if ev_type == EV_SWAP_DONE:
            d = int(a)
            if int(self.drone_status[d]) != DS_ON_TRUCK:
                return
            self.drone_batt[d] = float(cfg.max_battery)
            self.drone_task_id[d] = -1
            self.planned_stop[d] = -1
            self._drone_payload_failed[d] = False
            self._drone_payload_remain[d] = float(cfg.uncertainty.payload_lifetime_h)
            self._truck_pending_swaps = max(0, self._truck_pending_swaps - 1)
            if self._truck_pending_swaps == 0:
                cur_s = int(self.truck_stop[0])
                if self._truck_should_wait_for_inbound(cur_s):
                    inbound_eta = [float(self.drone_eta[od]) for od in range(cfg.ndrones)
                                   if int(self.drone_status[od]) in (DS_CRUISING, DS_LANDING) and
                                   int(self.drone_stop_id[od]) == cur_s]
                    self.truck_busy[0] = 1
                    self.truck_eta[0] = min(inbound_eta) if inbound_eta else float(self.t)
                else:
                    self.truck_busy[0] = 0
                    self.truck_eta[0] = float(self.t)
                    self._ensure_active_rendezvous_progress()
            self._invalidate_mask()

    # ------------------------------------------------------------------ reward

    def _task_entry_flight_time(self, d: int, tid: int, side: int) -> float:
        sc = self.scenario
        self._update_drone_xy()
        cur_xy = self._drone_xy[d].copy()
        target = sc.point_xy[tid] if int(sc.task_type[tid]) == TASK_POINT \
            else (self._line_start_xy(tid) if int(side) == 0 else self._line_end_xy(tid))
        return float(self._effective_flight_time(cur_xy, target))

    def _prepare_task_reward(self, d: int, tid: int, side: int) -> None:
        """Stash the weighted completion bonus for this task claim."""
        if 0 <= int(tid) < self.scenario.ntasks:
            self.step_events["risk_exposure"] = (
                float(self.step_events.get("risk_exposure", 0.0))
                + float(self.scenario.task_risk[int(tid)])
            )
        self.reward_model.prepare_task_claim(self, int(d), int(tid), int(side))

    def _task_deadline_adjustment(self, tid: int, completion_t: float) -> float:
        return float(self.reward_model.task_deadline_adjustment(
            self, int(tid), float(completion_t)))

    def _clear_pending_task_reward(self, d: int) -> None:
        self.reward_model.clear_pending_task_reward(self, int(d))

    def _compute_reward(self, dt: float) -> float:
        """Step reward = time penalty + task completion bonuses + rendezvous shaping."""
        return float(self.reward_model.compute(self, float(dt)))

    # ------------------------------------------------------------------ simulation advance

    def _advance_until_decision(self) -> Tuple[bool, bool, Dict[str, object]]:
        """Advance simulation time until a decision point or terminal state.

        Returns (terminated, truncated, extra).
        """
        cfg, sc = self.cfg, self.scenario
        extra: Dict[str, object] = {}

        while True:
            processed_event = False
            while self.evq and abs(float(self.evq[0][0]) - self.t) <= cfg.eps_time:
                _, _, et, ea, eb = heapq.heappop(self.evq)
                self._handle_event(et, ea, eb)
                processed_event = True
            if processed_event:
                self._update_drone_xy()
                self.drift_counter = 0

            if self._all_tasks_done():
                depot = int(cfg.depot_stop_id)
                drones_home = all(int(self.drone_status[d]) == DS_ON_TRUCK
                                  for d in range(cfg.ndrones))
                if (int(self.truck_busy[0]) == 0 and
                        int(self.truck_stop[0]) == depot and drones_home):
                    extra["end_reason"] = "success"
                    return True, False, extra

            if self.t >= sc.max_time_h - cfg.eps_time:
                extra["end_reason"] = "time_limit"
                return False, True, extra
            if self.decision_count >= sc.max_decisions:
                extra["end_reason"] = "decision_limit"
                return False, True, extra

            if self._joint_replan_requested:
                if self._step_event_reasons:
                    extra["event_reasons"] = list(self._step_event_reasons)
                extra["event_reason"] = str(self._replan_reason)
                self._clear_replan_request()
                self.drift_counter = 0
                return False, False, extra

            idle = np.concatenate([
                np.array([int(self.truck_busy[0]) == 0], dtype=bool),
                np.array([self._drone_is_idle(d) for d in range(cfg.ndrones)], dtype=bool),
            ])
            if np.any(idle):
                mask = self._get_action_mask()
                has_non_await = False
                for r in np.where(idle)[0]:
                    if np.any(mask[int(r), :self.AWAIT] == 1):
                        has_non_await = True
                        break
                if has_non_await:
                    self.drift_counter = 0
                    return False, False, extra
                for d in range(cfg.ndrones):
                    if (int(self.drone_status[d]) == DS_WAITING and
                            not self._waiting_drone_has_legal_action(d)):
                        extra["end_reason"] = "battery_depleted"
                        return False, True, extra

            if self.evq:
                dt_next = max(0.0, float(self.evq[0][0]) - self.t)
            else:
                dt_next = float(cfg.wait_step_h)

            if dt_next <= cfg.eps_time:
                self.drift_counter += 1
                if self.drift_counter > cfg.max_drift_steps:
                    extra["end_reason"] = "deadlock_drift"
                    return False, True, extra
                continue

            for d in range(cfg.ndrones):
                if int(self.drone_status[d]) == DS_ON_GROUND:
                    s = int(self.drone_stop_id[d])
                    if 0 <= s < sc.nstops:
                        self.episode_ground_wait_by_stop_h[s] += float(dt_next)

            ok, reason = self._integrate_energy(float(dt_next))
            self.t = float(self.t) + float(dt_next)
            if not ok:
                extra["end_reason"] = reason
                if reason == "battery_depleted":
                    self.step_events["battery_depleted_count"] = (
                        float(self.step_events.get("battery_depleted_count", 0.0)) + 1.0
                    )
                self._update_drone_xy()
                return False, True, extra
            self._update_drone_xy()

            if not self.evq:
                self.drift_counter += 1
                if self.drift_counter > cfg.max_drift_steps:
                    extra["end_reason"] = "deadlock_drift"
                    return False, True, extra
