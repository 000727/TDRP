from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..rewards import build_reward_model
from ..uncertainty import build_uncertainty_modules
from .configs import EnvConfig, Scenario
from .constants import (
    DS_CRUISING,
    DS_EXECUTING,
    DS_LANDING,
    DS_ON_GROUND,
    DS_ON_TRUCK,
    DS_TAKING_OFF,
    MAX_LINE_PTS,
    MAX_STOPS,
    MAX_TASK_ACTIONS,
    MAX_TASKS,
    N_WIND_KERNELS,
    SCALE_TIME_REFS_H,
    TASK_LINE,
    TASK_POINT,
)
from .fleet import FleetRuntimeState, FleetSpec
from .scenario_gen import ScenarioGenerator, polyline_length

PH_IDLE = -1
PH_TAKEOFF = 0
PH_TO_TASK = 1
PH_SERVICE = 2
PH_TO_RECOVERY = 3
PH_LANDING = 4


class MultiTruckMultiDroneEnv(gym.Env):
    """Turn-based multi-truck, multi-drone environment skeleton.

    This class is intentionally smaller than `TruckMultiDroneCleanEnv`. It gives
    the project a modular multi-fleet surface where uncertainty and reward hooks
    can be exercised before the detailed event simulator is fully generalized.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: EnvConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.fleet_spec = FleetSpec.from_config(cfg)
        self.cfg.ndrones = self.fleet_spec.ndrones

        self.ntrucks = self.fleet_spec.ntrucks
        self.ndrones = self.fleet_spec.ndrones
        self.nagents = self.ntrucks + self.ndrones

        self.act_offset_stop = 0
        self.act_offset_task = MAX_STOPS
        self.AWAIT = MAX_STOPS + MAX_TASK_ACTIONS
        self.nactions = self.AWAIT + 1

        self.action_space = spaces.MultiDiscrete([self.nactions] * self.nagents)
        self.observation_space = spaces.Dict(
            {
                "time_progress": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "done_ratio": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "stop_xy": spaces.Box(-1.0, 1.0, (MAX_STOPS, 2), dtype=np.float32),
                "stop_mask": spaces.Box(0, 1, (MAX_STOPS,), dtype=np.int8),
                "task_type": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
                "task_mask": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
                "task_done": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
                "task_spawned": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
                "point_xy": spaces.Box(-1.0, 1.0, (MAX_TASKS, 2), dtype=np.float32),
                "task_priority": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
                "task_deadline_h": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
                "task_reward_weight": spaces.Box(0.0, 4.0, (MAX_TASKS,), dtype=np.float32),
                "task_urgency": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
                "task_risk": spaces.Box(0.0, 1.0, (MAX_TASKS,), dtype=np.float32),
                "truck_stop": spaces.Box(0, MAX_STOPS - 1, (self.ntrucks,), dtype=np.int32),
                "truck_xy": spaces.Box(-1.0, 1.0, (self.ntrucks, 2), dtype=np.float32),
                "truck_busy": spaces.Box(0, 1, (self.ntrucks,), dtype=np.int8),
                "truck_eta_h": spaces.Box(0.0, 1.0, (self.ntrucks,), dtype=np.float32),
                "truck_target_stop": spaces.Box(0, MAX_STOPS - 1, (self.ntrucks,), dtype=np.int32),
                "drone_status": spaces.Box(0, 6, (self.ndrones,), dtype=np.int8),
                "drone_carrier_truck": spaces.Box(0, max(self.ntrucks - 1, 0), (self.ndrones,), dtype=np.int32),
                "drone_stop_id": spaces.Box(0, MAX_STOPS - 1, (self.ndrones,), dtype=np.int32),
                "drone_task_id": spaces.Box(-1, MAX_TASKS - 1, (self.ndrones,), dtype=np.int32),
                "drone_task_side": spaces.Box(0, 1, (self.ndrones,), dtype=np.int8),
                "drone_phase": spaces.Box(-1, 4, (self.ndrones,), dtype=np.int8),
                "drone_planned_recovery_stop": spaces.Box(-1, MAX_STOPS - 1, (self.ndrones,), dtype=np.int32),
                "drone_return_truck": spaces.Box(-1, max(self.ntrucks - 1, 0), (self.ndrones,), dtype=np.int32),
                "drone_return_stop": spaces.Box(-1, MAX_STOPS - 1, (self.ndrones,), dtype=np.int32),
                "drone_eta_h": spaces.Box(0.0, 1.0, (self.ndrones,), dtype=np.float32),
                "drone_xy": spaces.Box(-1.0, 1.0, (self.ndrones, 2), dtype=np.float32),
                "drone_batt_ratio": spaces.Box(0.0, 1.0, (self.ndrones,), dtype=np.float32),
                "drone_payload_remain": spaces.Box(0.0, 1.0, (self.ndrones,), dtype=np.float32),
                "drone_payload_failed": spaces.Box(0, 1, (self.ndrones,), dtype=np.int8),
                "task_claimed": spaces.Box(0, 1, (MAX_TASKS,), dtype=np.int8),
                "wind_vec": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                "wind_samples": spaces.Box(-1.0, 1.0, (N_WIND_KERNELS, 2), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (self.nagents, self.nactions), dtype=np.int8),
            }
        )

        self._scenario_gen = ScenarioGenerator(cfg)
        self.reward_model = build_reward_model(cfg)
        self._uncertainty_modules = []
        self._task_action_tid = -np.ones(MAX_TASK_ACTIONS, dtype=np.int32)
        self._task_action_side = np.zeros(MAX_TASK_ACTIONS, dtype=np.int8)
        self._task_action_count = 0

        self.scenario: Optional[Scenario] = None
        self.np_random: Optional[np.random.Generator] = None
        self._unc_rng: Optional[np.random.Generator] = None
        self._wind_field = None
        self.fleet: Optional[FleetRuntimeState] = None

        self.t = 0.0
        self.decision_count = 0
        self.time_ref_h = 1.0
        self.step_events: Dict[str, float] = {}

        self.truck_stop = np.zeros(self.ntrucks, dtype=np.int32)
        self.truck_busy = np.zeros(self.ntrucks, dtype=np.int8)
        self.truck_target = np.zeros(self.ntrucks, dtype=np.int32)
        self.truck_eta = np.zeros(self.ntrucks, dtype=np.float32)
        self._truck_xy = np.zeros((self.ntrucks, 2), dtype=np.float32)
        self._truck_move_start_xy = np.zeros((self.ntrucks, 2), dtype=np.float32)
        self._truck_move_end_xy = np.zeros((self.ntrucks, 2), dtype=np.float32)
        self._truck_move_total_h = np.zeros(self.ntrucks, dtype=np.float32)
        self._truck_move_remaining_h = np.zeros(self.ntrucks, dtype=np.float32)
        self.drone_status = np.full(self.ndrones, DS_ON_TRUCK, dtype=np.int8)
        self.drone_carrier_truck = np.zeros(self.ndrones, dtype=np.int32)
        self.drone_stop_id = np.zeros(self.ndrones, dtype=np.int32)
        self.drone_task_id = -np.ones(self.ndrones, dtype=np.int32)
        self.drone_task_side = np.zeros(self.ndrones, dtype=np.int8)
        self.drone_phase = np.full(self.ndrones, PH_IDLE, dtype=np.int8)
        self.drone_eta = np.zeros(self.ndrones, dtype=np.float32)
        self.drone_batt = np.full(self.ndrones, cfg.max_battery, dtype=np.float32)
        self._drone_xy = np.zeros((self.ndrones, 2), dtype=np.float32)
        self._drone_payload_remain = np.full(
            self.ndrones, cfg.uncertainty.payload_lifetime_h, dtype=np.float64
        )
        self._drone_sortie_start_t = np.zeros(self.ndrones, dtype=np.float64)
        self._drone_payload_failed = np.zeros(self.ndrones, dtype=np.bool_)
        self._drone_exec_is_hover = np.zeros(self.ndrones, dtype=np.bool_)
        self._pending_task_reward = np.zeros(self.ndrones, dtype=np.float32)
        self._drone_planned_recovery_stop = -np.ones(self.ndrones, dtype=np.int32)
        self._drone_return_truck = -np.ones(self.ndrones, dtype=np.int32)
        self._drone_return_stop = -np.ones(self.ndrones, dtype=np.int32)
        self._drone_completion_t = np.zeros(self.ndrones, dtype=np.float32)
        self._drone_sortie_energy = np.zeros(self.ndrones, dtype=np.float32)
        self._drone_swap_pending = np.zeros(self.ndrones, dtype=np.int8)
        self._drone_entry_xy = np.zeros((self.ndrones, 2), dtype=np.float32)
        self._drone_exit_xy = np.zeros((self.ndrones, 2), dtype=np.float32)
        self._drone_recovery_xy = np.zeros((self.ndrones, 2), dtype=np.float32)
        self._drone_service_h = np.zeros(self.ndrones, dtype=np.float32)
        self._drone_service_power = np.zeros(self.ndrones, dtype=np.float32)
        self._drone_phase_start_xy = np.zeros((self.ndrones, 2), dtype=np.float32)
        self._drone_phase_end_xy = np.zeros((self.ndrones, 2), dtype=np.float32)
        self._drone_phase_total_h = np.zeros(self.ndrones, dtype=np.float32)
        self._drone_phase_remaining_h = np.zeros(self.ndrones, dtype=np.float32)

        self.episode_swap_count = 0
        self.episode_battery_shock_count = 0
        self.episode_payload_failure_count = 0
        self.episode_ground_wait_time_h = 0.0
        self.episode_energy_used = 0.0
        self.episode_risk_exposure = 0.0

        self.task_done = np.zeros(MAX_TASKS, dtype=np.int8)
        self.task_claimed = np.zeros(MAX_TASKS, dtype=np.int8)
        self._task_spawned = np.zeros(MAX_TASKS, dtype=np.int8)
        self._point_service_remain = np.zeros(MAX_TASKS, dtype=np.float32)
        self._line_remain_xy = np.zeros((MAX_TASKS, MAX_LINE_PTS, 2), dtype=np.float32)
        self._line_remain_npts = np.zeros(MAX_TASKS, dtype=np.int8)
        self._line_remain_len = np.zeros(MAX_TASKS, dtype=np.float32)
        self._line_len = np.zeros(MAX_TASKS, dtype=np.float32)
        self._min_dist_point_to_stop = np.zeros(MAX_TASKS, dtype=np.float32)
        self._min_dist_lstart_to_stop = np.zeros(MAX_TASKS, dtype=np.float32)
        self._min_dist_lend_to_stop = np.zeros(MAX_TASKS, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        try:
            super().reset(seed=seed)
        except Exception:
            pass

        self.np_random = np.random.default_rng(seed)
        unc_seed = (
            self.cfg.uncertainty.fixed_seed
            if self.cfg.uncertainty.fixed_seed is not None
            else ((seed + 12345) if seed is not None else None)
        )
        self._unc_rng = np.random.default_rng(unc_seed)
        self.scenario = (
            options["scenario"]
            if options and "scenario" in options
            else self._scenario_gen.generate(self.np_random)
        )
        sc = self.scenario

        self.fleet = FleetRuntimeState.from_spec(
            self.fleet_spec, self.cfg.depot_stop_id, self.cfg.max_battery
        )
        self.truck_stop[:] = self.cfg.depot_stop_id
        self.truck_busy[:] = 0
        self.truck_target[:] = self.cfg.depot_stop_id
        self.truck_eta[:] = 0.0
        depot_xy = sc.stop_xy[self.cfg.depot_stop_id]
        self._truck_xy[:] = depot_xy
        self._truck_move_start_xy[:] = depot_xy
        self._truck_move_end_xy[:] = depot_xy
        self._truck_move_total_h[:] = 0.0
        self._truck_move_remaining_h[:] = 0.0
        self.drone_status[:] = DS_ON_TRUCK
        self.drone_stop_id[:] = self.cfg.depot_stop_id
        self.drone_task_id[:] = -1
        self.drone_task_side[:] = 0
        self.drone_phase[:] = PH_IDLE
        self.drone_eta[:] = 0.0
        self.drone_batt[:] = float(self.cfg.max_battery)
        for d in range(self.ndrones):
            self.drone_carrier_truck[d] = self.fleet_spec.carrier_for_drone(d)

        self.t = 0.0
        self.decision_count = 0
        self.step_events = {}
        self.task_done[:] = 0
        self.task_claimed[:] = 0
        self._pending_task_reward[:] = 0.0
        self._drone_payload_failed[:] = False
        self._drone_exec_is_hover[:] = False
        self._drone_planned_recovery_stop[:] = -1
        self._drone_return_truck[:] = -1
        self._drone_return_stop[:] = -1
        self._drone_completion_t[:] = 0.0
        self._drone_sortie_energy[:] = 0.0
        self._drone_swap_pending[:] = 0
        self._drone_entry_xy[:] = 0.0
        self._drone_exit_xy[:] = 0.0
        self._drone_recovery_xy[:] = 0.0
        self._drone_service_h[:] = 0.0
        self._drone_service_power[:] = 0.0
        self._drone_phase_start_xy[:] = 0.0
        self._drone_phase_end_xy[:] = 0.0
        self._drone_phase_total_h[:] = 0.0
        self._drone_phase_remaining_h[:] = 0.0
        self.episode_swap_count = 0
        self.episode_battery_shock_count = 0
        self.episode_payload_failure_count = 0
        self.episode_ground_wait_time_h = 0.0
        self.episode_energy_used = 0.0
        self.episode_risk_exposure = 0.0

        self._line_len[:] = 0.0
        self._line_len[: sc.ntasks] = sc.line_length[: sc.ntasks]
        self._point_service_remain[:] = 0.0
        self._point_service_remain[: sc.ntasks] = sc.point_service_h[: sc.ntasks]
        self._line_remain_xy[:] = 0.0
        self._line_remain_xy[: sc.ntasks] = sc.line_polyline_xy[: sc.ntasks]
        self._line_remain_npts[:] = 0
        self._line_remain_npts[: sc.ntasks] = sc.line_polyline_npts[: sc.ntasks]
        self._line_remain_len[:] = 0.0
        self._line_remain_len[: sc.ntasks] = sc.line_length[: sc.ntasks]

        self._task_spawned[:] = 0
        for tid in range(sc.ntasks):
            if float(sc.task_spawn_time[tid]) <= self.cfg.eps_time:
                self._task_spawned[tid] = 1
            elif not self.cfg.dynamic_generate_online:
                result = self._scenario_gen.generate_task_slot(
                    sc, int(tid), float(sc.task_spawn_time[tid]), self.np_random
                )
                if result is not None:
                    self._apply_generated_task_slot(int(tid), result)

        self._build_task_action_map()
        self._precompute_task_stop_dists()
        self._set_reward_reference_scales()
        self._init_uncertainty_state()
        self._sync_positions()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.nagents:
            raise ValueError(f"action has shape {action.shape}, expected ({self.nagents},)")

        self.step_events = {}
        mask = self._compute_action_mask()
        corrected = np.zeros(self.nagents, dtype=np.int64)
        for ag in range(self.nagents):
            a = int(action[ag])
            if a < 0 or a >= self.nactions or int(mask[ag, a]) == 0:
                corrected[ag] = self._first_legal_action(mask[ag])
            else:
                corrected[ag] = a

        used_tasks: set[int] = set()
        carriers_with_new_sorties: set[int] = set()

        for d in range(self.ndrones):
            a = int(corrected[self.ntrucks + d])
            if self._is_drone_recovery_stop_action(d, a):
                self._set_drone_planned_recovery_stop(d, a)
                continue
            tid, side = self._decode_task_id_and_side(a)
            if tid < 0 or tid in used_tasks:
                continue
            used_tasks.add(tid)
            if self._start_drone_task(d, tid, side):
                carriers_with_new_sorties.add(int(self.drone_carrier_truck[d]))

        for truck_id in range(self.ntrucks):
            if truck_id in carriers_with_new_sorties:
                continue
            a = int(corrected[truck_id])
            if self.act_offset_stop <= a < self.act_offset_stop + self.scenario.nstops:
                self._start_truck_move(truck_id, int(a))

        dt_h = self._next_event_dt()
        if dt_h is None:
            dt_h = float(self.cfg.wait_step_h)

        self.decision_count += 1
        self._advance_time(dt_h)
        self._complete_finished_events()

        reward = self.reward_model.compute(self, dt_h)
        all_done = bool(np.all(self.task_done[: self.scenario.ntasks] == 1))
        all_recovered = bool(np.all(self.drone_status == DS_ON_TRUCK))
        all_trucks_idle = bool(np.all(self.truck_busy == 0))
        terminated = all_done and all_recovered and all_trucks_idle
        truncated = bool(
            self.t >= self.scenario.max_time_h - self.cfg.eps_time
            or self.decision_count >= self.scenario.max_decisions
        )
        self._sync_positions()
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _start_truck_move(self, truck_id: int, target: int) -> bool:
        if int(self.truck_busy[truck_id]) != 0:
            return False
        cur = int(self.truck_stop[truck_id])
        if int(target) == cur:
            return False
        dt = self._truck_travel_time(cur, int(target))
        self.truck_busy[truck_id] = 1
        self.truck_target[truck_id] = int(target)
        self.truck_eta[truck_id] = float(dt)
        self._truck_move_start_xy[truck_id] = self._truck_xy[truck_id]
        self._truck_move_end_xy[truck_id] = self.scenario.stop_xy[int(target)]
        self._truck_move_total_h[truck_id] = float(dt)
        self._truck_move_remaining_h[truck_id] = float(dt)
        return True

    def _start_drone_task(self, d: int, tid: int, side: int) -> bool:
        if not self._task_action_feasible(d, tid, side):
            return False

        carrier = int(self.drone_carrier_truck[d])
        truck_xy = self._truck_xy[carrier].copy()
        entry, exit_xy = self._task_entry_exit_xy(tid, side)

        self._init_sortie_payload(d)
        shock = self._apply_battery_shock(d)
        if shock > self.cfg.eps_batt:
            self.step_events["battery_shock_count"] = (
                float(self.step_events.get("battery_shock_count", 0.0)) + 1.0
            )
            self.episode_battery_shock_count += 1

        recovery_truck, recovery_stop = self._selected_recovery_plan_for_drone(d, exit_xy, carrier)
        recovery_xy = self.scenario.stop_xy[recovery_stop].copy()
        to_task_h = self._effective_flight_time(truck_xy, entry)
        return_h = self._effective_flight_time(exit_xy, recovery_xy)
        if int(self.scenario.task_type[tid]) == TASK_POINT:
            service_h = float(self._point_service_remain[tid])
            service_power = self._effective_hover_power(entry)
            service_energy = service_power * service_h
        else:
            service_h = self._line_traverse_time(tid, side)
            service_energy = self._line_traverse_energy(tid, side)
            service_power = service_energy / max(service_h, self.cfg.eps_time)

        total_h = (
            float(self.cfg.takeoff_time_h)
            + to_task_h
            + service_h
            + return_h
            + float(self.cfg.landing_time_h)
        )
        energy = float(self.cfg.fly_power_per_h) * (
            float(self.cfg.takeoff_time_h) + float(self.cfg.landing_time_h)
        )
        energy += self._flight_energy(truck_xy, entry)
        energy += self._flight_energy(exit_xy, recovery_xy)
        energy += service_energy
        self.step_events["risk_exposure"] = (
            float(self.step_events.get("risk_exposure", 0.0))
            + float(self.scenario.task_risk[int(tid)])
        )
        self.episode_risk_exposure += float(self.scenario.task_risk[int(tid)])
        self.drone_status[d] = DS_TAKING_OFF
        self.drone_task_id[d] = int(tid)
        self.drone_task_side[d] = int(side)
        self.drone_phase[d] = PH_TAKEOFF
        self.drone_eta[d] = float(self.cfg.takeoff_time_h)
        self._drone_completion_t[d] = float(self.t) + float(total_h)
        self._drone_return_truck[d] = int(recovery_truck)
        self._drone_return_stop[d] = int(recovery_stop)
        self._drone_planned_recovery_stop[d] = -1
        self._drone_sortie_energy[d] = float(energy)
        self._drone_swap_pending[d] = 1
        self._drone_entry_xy[d] = entry.astype(np.float32)
        self._drone_exit_xy[d] = exit_xy.astype(np.float32)
        self._drone_recovery_xy[d] = recovery_xy.astype(np.float32)
        self._drone_service_h[d] = float(service_h)
        self._drone_service_power[d] = float(service_power)
        self._drone_exec_is_hover[d] = False
        self._set_drone_phase_segment(d, truck_xy, truck_xy, float(self.cfg.takeoff_time_h))

        self.reward_model.prepare_task_claim(self, d, tid, side)
        self.task_claimed[tid] = 1
        return True

    def _finish_drone_task(self, d: int) -> None:
        tid = int(self.drone_task_id[d])
        if tid < 0:
            self.drone_status[d] = DS_ON_TRUCK
            self.drone_eta[d] = 0.0
            return
        payload_failed = self._check_payload_failure(d)
        completion_t = float(self._drone_completion_t[d])
        battery_failed = float(self.drone_batt[d]) < float(self.cfg.safety_batt_margin)
        if battery_failed:
            self.step_events["battery_depleted_count"] = (
                float(self.step_events.get("battery_depleted_count", 0.0)) + 1.0
            )
        if payload_failed:
            self.step_events["payload_failure_count"] = (
                float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
            )
            self.episode_payload_failure_count += 1
        if not payload_failed and not battery_failed:
            self.task_done[tid] = 1
            self.step_events["task_reward"] = (
                float(self.step_events.get("task_reward", 0.0))
                + float(self._pending_task_reward[d])
                + float(self.reward_model.task_deadline_adjustment(self, tid, completion_t))
            )
        self.reward_model.clear_pending_task_reward(self, d)
        self.task_claimed[tid] = 0
        self.drone_status[d] = DS_ON_GROUND
        self.drone_task_id[d] = -1
        self.drone_task_side[d] = 0
        self.drone_eta[d] = 0.0
        self.drone_stop_id[d] = int(self._drone_return_stop[d])
        self._drone_xy[d] = self.scenario.stop_xy[int(self.drone_stop_id[d])]
        self._drone_exec_is_hover[d] = False
        self._drone_payload_failed[d] = bool(payload_failed)
        self._recover_drone_if_truck_available(d, int(self.drone_stop_id[d]))

    def _advance_drone_phase(self, d: int) -> None:
        tid = int(self.drone_task_id[d])
        if tid < 0:
            self.drone_status[d] = DS_ON_TRUCK
            self.drone_phase[d] = PH_IDLE
            self.drone_eta[d] = 0.0
            return

        phase = int(self.drone_phase[d])
        if phase == PH_TAKEOFF:
            self.drone_status[d] = DS_CRUISING
            self.drone_phase[d] = PH_TO_TASK
            cur = self._drone_xy[d].copy()
            eta = self._effective_flight_time(cur, self._drone_entry_xy[d])
            self.drone_eta[d] = float(eta)
            self._set_drone_phase_segment(d, cur, self._drone_entry_xy[d], eta)
            return

        if phase == PH_TO_TASK:
            self._drone_xy[d] = self._drone_entry_xy[d].copy()
            self.drone_status[d] = DS_EXECUTING
            self.drone_phase[d] = PH_SERVICE
            eta = float(self._drone_service_h[d])
            self.drone_eta[d] = float(eta)
            self._drone_exec_is_hover[d] = int(self.scenario.task_type[tid]) == TASK_POINT
            end_xy = self._drone_entry_xy[d] if self._drone_exec_is_hover[d] else self._drone_exit_xy[d]
            self._set_drone_phase_segment(d, self._drone_entry_xy[d], end_xy, eta)
            return

        if phase == PH_SERVICE:
            self._drone_xy[d] = self._drone_exit_xy[d].copy()
            payload_failed = self._check_payload_failure(d)
            battery_failed = float(self.drone_batt[d]) < float(self.cfg.safety_batt_margin)
            if battery_failed:
                self.step_events["battery_depleted_count"] = (
                    float(self.step_events.get("battery_depleted_count", 0.0)) + 1.0
                )
            if payload_failed:
                self.step_events["payload_failure_count"] = (
                    float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
                )
                self.episode_payload_failure_count += 1
            if not payload_failed and not battery_failed:
                self.task_done[tid] = 1
                if int(self.scenario.task_type[tid]) == TASK_LINE:
                    self._set_line_remaining_poly(
                        tid, np.asarray([self._drone_exit_xy[d]], dtype=np.float32)
                    )
                self.step_events["task_reward"] = (
                    float(self.step_events.get("task_reward", 0.0))
                    + float(self._pending_task_reward[d])
                    + float(self.reward_model.task_deadline_adjustment(self, tid, float(self.t)))
                )
            self.reward_model.clear_pending_task_reward(self, d)
            self.task_claimed[tid] = 0
            self._drone_payload_failed[d] = bool(payload_failed)
            self._drone_exec_is_hover[d] = False
            self.drone_status[d] = DS_CRUISING
            self.drone_phase[d] = PH_TO_RECOVERY
            eta = self._effective_flight_time(self._drone_xy[d], self._drone_recovery_xy[d])
            self.drone_eta[d] = float(eta)
            self._set_drone_phase_segment(d, self._drone_xy[d], self._drone_recovery_xy[d], eta)
            return

        if phase == PH_TO_RECOVERY:
            self._drone_xy[d] = self._drone_recovery_xy[d].copy()
            self.drone_status[d] = DS_LANDING
            self.drone_phase[d] = PH_LANDING
            eta = float(self.cfg.landing_time_h)
            self.drone_eta[d] = eta
            self._set_drone_phase_segment(d, self._drone_xy[d], self._drone_xy[d], eta)
            return

        if phase == PH_LANDING:
            self._finish_drone_task(d)

    def _recover_drone(self, d: int, truck_id: int) -> None:
        if int(self.drone_status[d]) != DS_ON_GROUND:
            return
        if int(self.drone_stop_id[d]) != int(self.truck_stop[truck_id]):
            return
        self.drone_status[d] = DS_ON_TRUCK
        self.drone_carrier_truck[d] = int(truck_id)
        self.drone_task_id[d] = -1
        self.drone_task_side[d] = 0
        self.drone_phase[d] = PH_IDLE
        self.drone_batt[d] = float(self.cfg.max_battery)
        self.drone_eta[d] = 0.0
        self._drone_xy[d] = self.scenario.stop_xy[int(self.truck_stop[truck_id])]
        self._drone_payload_failed[d] = False
        if int(self._drone_swap_pending[d]) == 1:
            self.episode_swap_count += 1
        self._drone_swap_pending[d] = 0
        self._drone_return_truck[d] = -1
        self._drone_return_stop[d] = -1
        self._drone_sortie_energy[d] = 0.0
        self._drone_planned_recovery_stop[d] = -1

    def _advance_time(self, dt_h: float) -> None:
        if dt_h <= self.cfg.eps_time:
            return
        for module in self._uncertainty_modules:
            module.on_time_advance(self, float(dt_h))
        ground = self.drone_status == DS_ON_GROUND
        if np.any(ground):
            self.episode_ground_wait_time_h += float(dt_h) * float(np.sum(ground))
        busy_trucks = self.truck_busy.astype(bool)
        self.truck_eta[busy_trucks] = np.maximum(
            0.0, self.truck_eta[busy_trucks].astype(np.float64) - float(dt_h)
        ).astype(np.float32)
        self._update_truck_positions(float(dt_h))
        self._integrate_drone_energy(float(dt_h))
        busy_drones = self.drone_status != DS_ON_TRUCK
        self.drone_eta[busy_drones] = np.maximum(
            0.0, self.drone_eta[busy_drones].astype(np.float64) - float(dt_h)
        ).astype(np.float32)
        self._update_drone_phase_positions(float(dt_h))
        self.t = float(self.t) + float(dt_h)
        self._spawn_due_tasks()

    def _integrate_drone_energy(self, dt_h: float) -> None:
        for d in range(self.ndrones):
            st = int(self.drone_status[d])
            if st in (DS_ON_TRUCK, DS_ON_GROUND):
                continue
            power = self._drone_phase_power(d)
            energy = float(power) * float(dt_h)
            self.drone_batt[d] = max(0.0, float(self.drone_batt[d]) - energy)
            self.step_events["energy_used"] = (
                float(self.step_events.get("energy_used", 0.0)) + energy
            )
            self.episode_energy_used += energy

    def _drone_phase_power(self, d: int) -> float:
        if int(self.drone_status[d]) == DS_EXECUTING:
            return (
                self._effective_hover_power(self._drone_xy[d])
                if bool(self._drone_exec_is_hover[d])
                else float(self._drone_service_power[d])
            )
        return self._effective_flight_power(
            self._drone_phase_start_xy[d], self._drone_phase_end_xy[d]
        )

    def _payload_time_to_failure(self, d: int) -> Optional[float]:
        if int(self.drone_status[d]) in (DS_ON_TRUCK, DS_ON_GROUND):
            return None
        unc = self.cfg.uncertainty
        if not (unc.enabled and unc.payload_enabled and unc.payload_intensity > 0):
            return None
        if bool(getattr(unc, "payload_probabilistic_failure", False)):
            return None
        remain = float(self._drone_payload_remain[d]) - (
            float(self.t) - float(self._drone_sortie_start_t[d])
        )
        if remain <= self.cfg.eps_time:
            return self.cfg.eps_time
        return float(remain)

    def _battery_time_to_depletion(self, d: int) -> Optional[float]:
        if int(self.drone_status[d]) in (DS_ON_TRUCK, DS_ON_GROUND):
            return None
        power = max(float(self._drone_phase_power(d)), 1e-9)
        usable = float(self.drone_batt[d]) - float(self.cfg.safety_batt_margin)
        if usable <= self.cfg.eps_batt:
            return self.cfg.eps_time
        return float(usable / power)

    def _update_drone_phase_positions(self, dt_h: float) -> None:
        for d in range(self.ndrones):
            if int(self.drone_status[d]) in (DS_ON_TRUCK, DS_ON_GROUND):
                continue
            total = float(self._drone_phase_total_h[d])
            if total <= self.cfg.eps_time:
                self._drone_xy[d] = self._drone_phase_end_xy[d].copy()
                continue
            remain = max(0.0, float(self._drone_phase_remaining_h[d]) - float(dt_h))
            self._drone_phase_remaining_h[d] = float(remain)
            alpha = float(np.clip(1.0 - remain / total, 0.0, 1.0))
            tid = int(self.drone_task_id[d])
            if (
                int(self.drone_phase[d]) == PH_SERVICE
                and 0 <= tid < self.scenario.ntasks
                and int(self.scenario.task_type[tid]) == TASK_LINE
            ):
                elapsed = max(0.0, total - remain)
                current, _, _ = self._line_progress_state(
                    tid, int(self.drone_task_side[d]), elapsed
                )
                self._drone_xy[d] = current.astype(np.float32)
                continue
            self._drone_xy[d] = (
                self._drone_phase_start_xy[d]
                + alpha * (self._drone_phase_end_xy[d] - self._drone_phase_start_xy[d])
            ).astype(np.float32)

    def _update_truck_positions(self, dt_h: float) -> None:
        for truck_id in range(self.ntrucks):
            if int(self.truck_busy[truck_id]) == 0:
                self._truck_xy[truck_id] = self.scenario.stop_xy[int(self.truck_stop[truck_id])]
                continue
            total = float(self._truck_move_total_h[truck_id])
            if total <= self.cfg.eps_time:
                self._truck_xy[truck_id] = self._truck_move_end_xy[truck_id].copy()
                self._truck_move_remaining_h[truck_id] = 0.0
                continue
            remain = max(0.0, float(self._truck_move_remaining_h[truck_id]) - float(dt_h))
            self._truck_move_remaining_h[truck_id] = float(remain)
            alpha = float(np.clip(1.0 - remain / total, 0.0, 1.0))
            self._truck_xy[truck_id] = (
                self._truck_move_start_xy[truck_id]
                + alpha * (self._truck_move_end_xy[truck_id] - self._truck_move_start_xy[truck_id])
            ).astype(np.float32)

    def _next_event_dt(self) -> Optional[float]:
        candidates = []
        sc = self.scenario
        for tid in range(sc.ntasks):
            if int(self._task_spawned[tid]) == 1:
                continue
            dt_spawn = float(sc.task_spawn_time[tid]) - float(self.t)
            if dt_spawn > self.cfg.eps_time:
                candidates.append(float(dt_spawn))
        if np.any(self.truck_busy):
            candidates.extend(float(x) for x in self.truck_eta[self.truck_busy.astype(bool)])
        busy_drones = self.drone_status != DS_ON_TRUCK
        if np.any(busy_drones):
            candidates.extend(float(x) for x in self.drone_eta[busy_drones])
            for d in np.flatnonzero(busy_drones):
                payload_dt = self._payload_time_to_failure(int(d))
                if payload_dt is not None:
                    candidates.append(float(payload_dt))
                battery_dt = self._battery_time_to_depletion(int(d))
                if battery_dt is not None:
                    candidates.append(float(battery_dt))
        candidates = [x for x in candidates if x > self.cfg.eps_time]
        if not candidates:
            return None
        return float(min(candidates))

    def _complete_finished_events(self) -> None:
        for truck_id in range(self.ntrucks):
            if int(self.truck_busy[truck_id]) == 0:
                continue
            if float(self.truck_eta[truck_id]) > self.cfg.eps_time:
                continue
            self.truck_stop[truck_id] = int(self.truck_target[truck_id])
            self._truck_xy[truck_id] = self.scenario.stop_xy[int(self.truck_stop[truck_id])]
            self.truck_busy[truck_id] = 0
            self.truck_eta[truck_id] = 0.0
            self._truck_move_total_h[truck_id] = 0.0
            self._truck_move_remaining_h[truck_id] = 0.0
            self._recover_waiting_drones_at_truck(truck_id)

        self._handle_midphase_drone_failures()

        for d in range(self.ndrones):
            if int(self.drone_status[d]) == DS_ON_TRUCK:
                continue
            if float(self.drone_eta[d]) > self.cfg.eps_time:
                continue
            self._advance_drone_phase(d)

    def _recover_waiting_drones_at_truck(self, truck_id: int) -> None:
        stop = int(self.truck_stop[truck_id])
        for d in range(self.ndrones):
            if int(self.drone_status[d]) == DS_ON_GROUND and int(self.drone_stop_id[d]) == stop:
                self._recover_drone(d, truck_id)

    def _recover_drone_if_truck_available(self, d: int, stop: int) -> bool:
        preferred = int(self._drone_return_truck[d])
        if 0 <= preferred < self.ntrucks:
            if int(self.truck_busy[preferred]) == 0 and int(self.truck_stop[preferred]) == int(stop):
                self._recover_drone(d, preferred)
                return True
        for truck_id in range(self.ntrucks):
            if int(self.truck_busy[truck_id]) == 0 and int(self.truck_stop[truck_id]) == int(stop):
                self._recover_drone(d, truck_id)
                return True
        return False

    def _handle_midphase_drone_failures(self) -> None:
        for d in range(self.ndrones):
            if int(self.drone_status[d]) in (DS_ON_TRUCK, DS_ON_GROUND):
                continue
            payload_failed = self._check_payload_failure(d)
            battery_failed = float(self.drone_batt[d]) < float(self.cfg.safety_batt_margin)
            if not (payload_failed or battery_failed):
                continue
            self._abort_drone_sortie(d, payload_failed=payload_failed,
                                     battery_failed=battery_failed)

    def _abort_drone_sortie(self, d: int, payload_failed: bool, battery_failed: bool) -> None:
        tid = int(self.drone_task_id[d])
        if battery_failed:
            self.step_events["battery_depleted_count"] = (
                float(self.step_events.get("battery_depleted_count", 0.0)) + 1.0
            )
        if payload_failed:
            self.step_events["payload_failure_count"] = (
                float(self.step_events.get("payload_failure_count", 0.0)) + 1.0
            )
            self.episode_payload_failure_count += 1
        if 0 <= tid < self.scenario.ntasks:
            self.task_claimed[tid] = 0
            if (
                int(self.drone_phase[d]) == PH_SERVICE
                and int(self.scenario.task_type[tid]) == TASK_LINE
            ):
                elapsed = max(
                    0.0,
                    float(self._drone_phase_total_h[d])
                    - float(self._drone_phase_remaining_h[d]),
                )
                current, rem_pts, finished = self._line_progress_state(
                    tid, int(self.drone_task_side[d]), elapsed
                )
                self._drone_xy[d] = current.astype(np.float32)
                if not finished:
                    self._set_line_remaining_poly(tid, rem_pts)
        self.reward_model.clear_pending_task_reward(self, d)
        recovery_truck = int(self._drone_return_truck[d])
        if recovery_truck < 0:
            recovery_truck = int(self.drone_carrier_truck[d])
        recovery_truck, stop = self._select_recovery_plan(self._drone_xy[d], recovery_truck)
        self.drone_status[d] = DS_ON_GROUND
        self.drone_stop_id[d] = int(stop)
        self.drone_task_id[d] = -1
        self.drone_task_side[d] = 0
        self.drone_phase[d] = PH_IDLE
        self.drone_eta[d] = 0.0
        self._drone_xy[d] = self.scenario.stop_xy[int(stop)]
        self._drone_exec_is_hover[d] = False
        self._drone_payload_failed[d] = bool(payload_failed)
        self._drone_return_truck[d] = int(recovery_truck)
        self._drone_return_stop[d] = int(stop)
        self._drone_phase_total_h[d] = 0.0
        self._drone_phase_remaining_h[d] = 0.0
        self._recover_drone_if_truck_available(d, int(stop))

    def _spawn_due_tasks(self) -> None:
        sc = self.scenario
        for tid in range(sc.ntasks):
            if int(self._task_spawned[tid]) == 1:
                continue
            if float(sc.task_spawn_time[tid]) > self.t + self.cfg.eps_time:
                continue
            ok = True
            if self.cfg.dynamic_generate_online:
                ok = self._spawn_dynamic_task_from_modules(int(tid))
            if ok:
                self._task_spawned[tid] = 1
                self.step_events["new_task_spawned"] = (
                    float(self.step_events.get("new_task_spawned", 0.0)) + 1.0
                )
                self._build_task_action_map()

    def _init_uncertainty_state(self) -> None:
        self._wind_field = None
        self._uncertainty_modules = build_uncertainty_modules(self.cfg)
        for module in self._uncertainty_modules:
            if hasattr(module, "bind"):
                module.bind(self)
            module.on_reset(self)
        if hasattr(self.reward_model, "on_reset"):
            self.reward_model.on_reset(self)

    def _wind_at(self, xy: np.ndarray) -> np.ndarray:
        if self._wind_field is None:
            return np.zeros(2, dtype=np.float64)
        return self._wind_field.get(xy)

    def _effective_flight_time(
        self, from_xy: np.ndarray, to_xy: np.ndarray, base_speed: Optional[float] = None
    ) -> float:
        base_speed = self.cfg.drone_speed_kmph if base_speed is None else float(base_speed)
        dist = float(np.linalg.norm(np.asarray(to_xy) - np.asarray(from_xy)))
        current = 0.0 if dist < 1e-9 else float(dist / base_speed)
        for module in self._uncertainty_modules:
            current = module.flight_time(self, from_xy, to_xy, base_speed, current)
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

    def _generate_future_task_into_slot(self, tid: int) -> bool:
        result = self._scenario_gen.generate_task_slot(
            self.scenario, int(tid), float(self.t), self.np_random
        )
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
        self._point_service_remain[tid] = float(result["point_service_remain"])
        self._line_remain_xy[tid] = result["line_remain_xy"]
        self._line_remain_npts[tid] = int(result["line_remain_npts"])
        self._line_remain_len[tid] = float(result["line_remain_len"])
        self._line_len[tid] = float(result["line_len"])

    def _precompute_task_stop_dists(self) -> None:
        sc, T = self.scenario, self.scenario.ntasks
        stops = sc.stop_xy
        self._min_dist_point_to_stop[:T] = np.min(
            np.linalg.norm(sc.point_xy[:T, None, :] - stops[None, :, :], axis=-1), axis=1
        )
        self._min_dist_lstart_to_stop[:T] = np.min(
            np.linalg.norm(sc.line_start_xy[:T, None, :] - stops[None, :, :], axis=-1),
            axis=1,
        )
        self._min_dist_lend_to_stop[:T] = np.min(
            np.linalg.norm(sc.line_end_xy[:T, None, :] - stops[None, :, :], axis=-1),
            axis=1,
        )

    def _build_task_action_map(self) -> None:
        self._task_action_tid[:] = -1
        self._task_action_side[:] = 0
        k = 0
        for tid in range(self.scenario.ntasks):
            self._task_action_tid[k] = tid
            self._task_action_side[k] = 0
            k += 1
            if int(self.scenario.task_type[tid]) == TASK_LINE:
                self._task_action_tid[k] = tid
                self._task_action_side[k] = 1
                k += 1
        self._task_action_count = k

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.nagents, self.nactions), dtype=np.int8)
        mask[:, self.AWAIT] = 1
        for truck_id in range(self.ntrucks):
            if int(self.truck_busy[truck_id]) != 0:
                continue
            waiting_stops = self._waiting_drone_stops()
            if waiting_stops:
                mask[truck_id, self.AWAIT] = 0
                for s in waiting_stops:
                    if s != int(self.truck_stop[truck_id]):
                        mask[truck_id, s] = 1
                if np.any(mask[truck_id]):
                    continue
                mask[truck_id, self.AWAIT] = 1
            for s in range(self.scenario.nstops):
                if s != int(self.truck_stop[truck_id]):
                    mask[truck_id, s] = 1
        for d in range(self.ndrones):
            row = self.ntrucks + d
            if int(self.drone_status[d]) != DS_ON_TRUCK:
                continue
            if bool(getattr(self.cfg, "enable_recovery_stop_actions", True)):
                for s in range(self.scenario.nstops):
                    mask[row, self.act_offset_stop + s] = 1
            for k in range(self._task_action_count):
                tid = int(self._task_action_tid[k])
                side = int(self._task_action_side[k])
                if self._task_action_feasible(d, tid, side):
                    mask[row, self.act_offset_task + k] = 1
        return mask

    def _is_drone_recovery_stop_action(self, d: int, action: int) -> bool:
        if not bool(getattr(self.cfg, "enable_recovery_stop_actions", True)):
            return False
        return (
            int(self.drone_status[d]) == DS_ON_TRUCK
            and self.act_offset_stop <= int(action) < self.act_offset_stop + self.scenario.nstops
        )

    def _set_drone_planned_recovery_stop(self, d: int, stop: int) -> None:
        self._drone_planned_recovery_stop[d] = int(stop)
        self.step_events["recovery_stop_planned_count"] = (
            float(self.step_events.get("recovery_stop_planned_count", 0.0)) + 1.0
        )

    def _task_action_feasible(self, d: int, tid: int, side: int) -> bool:
        if int(self.drone_status[d]) != DS_ON_TRUCK:
            return False
        if not (0 <= int(tid) < self.scenario.ntasks):
            return False
        if int(self._task_spawned[tid]) == 0:
            return False
        if int(self.task_done[tid]) == 1 or int(self.task_claimed[tid]) == 1:
            return False
        carrier = int(self.drone_carrier_truck[d])
        if int(self.truck_busy[carrier]) != 0:
            return False
        truck_xy = self._truck_xy[carrier].copy()
        entry, exit_xy = self._task_entry_exit_xy(tid, side)
        _, recovery_stop = self._selected_recovery_plan_for_drone(d, exit_xy, carrier)
        recovery_xy = self.scenario.stop_xy[recovery_stop].copy()
        energy = float(self.cfg.fly_power_per_h) * (
            float(self.cfg.takeoff_time_h) + float(self.cfg.landing_time_h)
        )
        energy += self._flight_energy(truck_xy, entry)
        energy += self._flight_energy(exit_xy, recovery_xy)
        if int(self.scenario.task_type[tid]) == TASK_POINT:
            energy += self._effective_hover_power(entry) * float(self._point_service_remain[tid])
        else:
            energy += self._line_traverse_energy(tid, side)
        energy += float(self.cfg.safety_batt_margin)
        return float(self.drone_batt[d]) + float(self.cfg.eps_batt) >= energy

    def _waiting_drone_stops(self) -> list[int]:
        stops = set()
        for d in range(self.ndrones):
            if int(self.drone_status[d]) == DS_ON_GROUND:
                stops.add(int(self.drone_stop_id[d]))
        return sorted(stops)

    def _first_legal_action(self, row: np.ndarray) -> int:
        legal = np.flatnonzero(row)
        return int(legal[0]) if legal.size else int(self.AWAIT)

    def _decode_task_id_and_side(self, a: int) -> Tuple[int, int]:
        k = int(a) - self.act_offset_task
        if 0 <= k < self._task_action_count:
            return int(self._task_action_tid[k]), int(self._task_action_side[k])
        return -1, 0

    def _task_entry_exit_xy(self, tid: int, side: int) -> Tuple[np.ndarray, np.ndarray]:
        if int(self.scenario.task_type[tid]) == TASK_POINT:
            p = self.scenario.point_xy[tid].copy()
            return p, p
        pts = self._line_poly_points(tid, side)
        if pts.shape[0] >= 2:
            return pts[0].astype(np.float32), pts[-1].astype(np.float32)
        if int(side) == 0:
            return self.scenario.line_start_xy[tid].copy(), self.scenario.line_end_xy[tid].copy()
        return self.scenario.line_end_xy[tid].copy(), self.scenario.line_start_xy[tid].copy()

    def _line_poly_points(self, tid: int, side: int = 0) -> np.ndarray:
        npts = int(self._line_remain_npts[tid])
        pts = self._line_remain_xy[tid, :npts].astype(np.float64)
        if pts.shape[0] < 2:
            pts = np.asarray(
                [self.scenario.line_start_xy[tid], self.scenario.line_end_xy[tid]],
                dtype=np.float64,
            )
        if int(side) == 1:
            pts = pts[::-1].copy()
        return pts

    def _set_line_remaining_poly(self, tid: int, pts: np.ndarray) -> None:
        pts = np.asarray(pts, dtype=np.float32)
        npts = int(min(len(pts), MAX_LINE_PTS))
        self._line_remain_xy[tid] = 0.0
        self._line_remain_npts[tid] = max(0, npts)
        if npts > 0:
            self._line_remain_xy[tid, :npts] = pts[:npts]
        if npts >= 2:
            self._line_remain_len[tid] = float(polyline_length(self._line_remain_xy[tid, :npts]))
        else:
            self._line_remain_len[tid] = 0.0

    def _line_progress_state(
        self, tid: int, side: int, elapsed_service_h: float
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        pts_original = self._line_poly_points(tid, 0).astype(np.float32)
        pts_work = pts_original[::-1].copy() if int(side) == 1 else pts_original.copy()
        if pts_work.shape[0] < 2:
            point = pts_work[0].copy() if pts_work.shape[0] else self.scenario.point_xy[tid].copy()
            return point.astype(np.float32), pts_original.astype(np.float32), True

        factor = max(float(self.scenario.task_service_factor[tid]), 1e-9)
        budget = max(0.0, float(elapsed_service_h) / factor)
        speed = float(self.cfg.drone_speed_kmph) * float(self.cfg.line_speed_factor)
        for i in range(pts_work.shape[0] - 1):
            p0 = pts_work[i]
            p1 = pts_work[i + 1]
            seg_t = self._effective_flight_time(p0, p1, speed)
            if seg_t <= self.cfg.eps_time:
                continue
            if budget + self.cfg.eps_time < seg_t:
                alpha = float(np.clip(budget / seg_t, 0.0, 1.0))
                current = (p0 + alpha * (p1 - p0)).astype(np.float32)
                rem_oriented = np.vstack([current, pts_work[i + 1:]]).astype(np.float32)
                rem_original = rem_oriented if int(side) == 0 else rem_oriented[::-1].copy()
                return current, rem_original.astype(np.float32), False
            budget -= float(seg_t)

        current = pts_work[-1].copy().astype(np.float32)
        return current, np.asarray([current], dtype=np.float32), True

    def _line_traverse_time(self, tid: int, side: int = 0) -> float:
        speed = float(self.cfg.drone_speed_kmph) * float(self.cfg.line_speed_factor)
        factor = float(self.scenario.task_service_factor[tid])
        pts = self._line_poly_points(tid, side)
        if pts.shape[0] < 2:
            return 0.0
        total = sum(self._effective_flight_time(p0, p1, speed) for p0, p1 in zip(pts[:-1], pts[1:]))
        return float(total) * factor

    def _line_traverse_energy(self, tid: int, side: int = 0) -> float:
        speed = float(self.cfg.drone_speed_kmph) * float(self.cfg.line_speed_factor)
        factor = float(self.scenario.task_service_factor[tid])
        pts = self._line_poly_points(tid, side)
        if pts.shape[0] < 2:
            return 0.0
        energy = sum(
            self._flight_energy(p0, p1, speed, self.cfg.fly_power_per_h)
            for p0, p1 in zip(pts[:-1], pts[1:])
        )
        return float(energy) * factor

    def _truck_travel_time(self, s_from: int, s_to: int) -> float:
        return float(self.scenario.truck_dist[int(s_from), int(s_to)]) / max(
            float(self.cfg.truck_speed_kmph), 1e-9
        )

    def _truck_eta_to_stop(self, truck_id: int, stop: int) -> float:
        truck_id = int(truck_id)
        stop = int(stop)
        if int(self.truck_busy[truck_id]) != 0:
            target = int(self.truck_target[truck_id])
            return float(self.truck_eta[truck_id]) + self._truck_travel_time(target, stop)
        return self._truck_travel_time(int(self.truck_stop[truck_id]), stop)

    def _select_recovery_plan(self, xy: np.ndarray, preferred_truck: int = 0) -> Tuple[int, int]:
        preferred_truck = int(np.clip(int(preferred_truck), 0, max(self.ntrucks - 1, 0)))
        policy = str(getattr(self.cfg, "recovery_policy_name", "nearest_stop")).lower()
        if policy in {"nearest_stop", "nearest"}:
            return preferred_truck, self._nearest_stop_id(xy)

        best_key: tuple[float, float, int, int] | None = None
        best_pair = (preferred_truck, self._nearest_stop_id(xy))
        for truck_id in range(self.ntrucks):
            for stop in range(self.scenario.nstops):
                drone_t = self._effective_flight_time(xy, self.scenario.stop_xy[stop])
                truck_t = self._truck_eta_to_stop(truck_id, stop)
                drone_e = self._flight_energy(xy, self.scenario.stop_xy[stop])
                if policy in {"min_energy", "energy"}:
                    key = (float(drone_e), float(drone_t + truck_t), int(truck_id), int(stop))
                else:
                    key = (float(max(drone_t, truck_t)), float(drone_t + truck_t), int(truck_id), int(stop))
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (int(truck_id), int(stop))
        return best_pair

    def _selected_recovery_plan_for_drone(
        self, d: int, xy: np.ndarray, preferred_truck: int = 0
    ) -> Tuple[int, int]:
        planned_stop = int(self._drone_planned_recovery_stop[d])
        if 0 <= planned_stop < self.scenario.nstops:
            return self._select_recovery_truck_for_stop(planned_stop, preferred_truck), planned_stop
        return self._select_recovery_plan(xy, preferred_truck)

    def _select_recovery_truck_for_stop(self, stop: int, preferred_truck: int = 0) -> int:
        preferred_truck = int(np.clip(int(preferred_truck), 0, max(self.ntrucks - 1, 0)))
        policy = str(getattr(self.cfg, "recovery_policy_name", "nearest_stop")).lower()
        if policy in {"nearest_stop", "nearest"}:
            return preferred_truck
        best_key: tuple[float, int] | None = None
        best_truck = preferred_truck
        for truck_id in range(self.ntrucks):
            key = (float(self._truck_eta_to_stop(truck_id, int(stop))), int(truck_id))
            if best_key is None or key < best_key:
                best_key = key
                best_truck = int(truck_id)
        return best_truck

    def _set_drone_phase_segment(
        self,
        d: int,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        duration_h: float,
    ) -> None:
        self._drone_phase_start_xy[d] = np.asarray(start_xy, dtype=np.float32)
        self._drone_phase_end_xy[d] = np.asarray(end_xy, dtype=np.float32)
        duration_h = max(float(duration_h), 0.0)
        self._drone_phase_total_h[d] = float(duration_h)
        self._drone_phase_remaining_h[d] = float(duration_h)

    def _nearest_stop_id(self, xy: np.ndarray) -> int:
        dists = np.linalg.norm(self.scenario.stop_xy[: self.scenario.nstops] - xy, axis=1)
        return int(np.argmin(dists))

    def _sync_positions(self) -> None:
        for d in range(self.ndrones):
            if int(self.drone_status[d]) == DS_ON_GROUND:
                self._drone_xy[d] = self.scenario.stop_xy[int(self.drone_stop_id[d])]
                continue
            if int(self.drone_status[d]) != DS_ON_TRUCK:
                continue
            carrier = int(self.drone_carrier_truck[d])
            self._drone_xy[d] = self._truck_xy[carrier].copy()

    def _set_reward_reference_scales(self) -> None:
        self.time_ref_h = float(
            SCALE_TIME_REFS_H.get(
                str(self.scenario.scale), max(float(self.scenario.max_time_h) * 0.5, 1.0)
            )
        )

    def _norm_global_time(self, value_h: float) -> float:
        return float(value_h) / max(float(self.time_ref_h), 1e-6)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        sc = self.scenario
        depot_xy = sc.stop_xy[self.cfg.depot_stop_id]
        map_range = max(float(sc.map_range), 1e-9)

        def norm_xy(xy):
            return np.clip((xy - depot_xy) / map_range, -1.0, 1.0).astype(np.float32)

        stop_xy = np.zeros((MAX_STOPS, 2), dtype=np.float32)
        stop_xy[: sc.nstops] = norm_xy(sc.stop_xy[: sc.nstops])
        stop_mask = np.zeros(MAX_STOPS, dtype=np.int8)
        stop_mask[: sc.nstops] = 1

        task_type = np.zeros(MAX_TASKS, dtype=np.int8)
        task_mask = np.zeros(MAX_TASKS, dtype=np.int8)
        task_done = np.zeros(MAX_TASKS, dtype=np.int8)
        task_spawned = np.zeros(MAX_TASKS, dtype=np.int8)
        task_claimed = np.zeros(MAX_TASKS, dtype=np.int8)
        point_xy = np.zeros((MAX_TASKS, 2), dtype=np.float32)
        task_priority = np.zeros(MAX_TASKS, dtype=np.float32)
        task_deadline_h = np.zeros(MAX_TASKS, dtype=np.float32)
        task_reward_weight = np.zeros(MAX_TASKS, dtype=np.float32)
        task_urgency = np.zeros(MAX_TASKS, dtype=np.float32)
        task_risk = np.zeros(MAX_TASKS, dtype=np.float32)
        T = sc.ntasks
        task_type[:T] = sc.task_type[:T]
        task_mask[:T] = self._task_spawned[:T]
        task_done[:T] = self.task_done[:T]
        task_spawned[:T] = self._task_spawned[:T]
        task_claimed[:T] = self.task_claimed[:T]
        point_xy[:T] = norm_xy(sc.point_xy[:T])
        task_priority[:T] = np.clip(sc.task_priority[:T] / 3.0, 0.0, 1.0)
        task_deadline_h[:T] = np.clip(sc.task_deadline_h[:T] / sc.max_time_h, 0.0, 1.0)
        task_reward_weight[:T] = sc.task_reward_weight[:T].astype(np.float32)
        urgency_ref = max(float(getattr(self.cfg, "critical_urgency_range", (0.5, 0.9))[1]), 1e-9)
        task_urgency[:T] = np.clip(sc.task_urgency[:T] / urgency_ref, 0.0, 1.0).astype(np.float32)
        task_risk[:T] = np.clip(sc.task_risk[:T], 0.0, 1.0)

        truck_xy = np.zeros((self.ntrucks, 2), dtype=np.float32)
        for truck_id in range(self.ntrucks):
            truck_xy[truck_id] = norm_xy(self._truck_xy[truck_id])

        wind_vec = np.zeros(2, dtype=np.float32)
        wind_samples = np.zeros((N_WIND_KERNELS, 2), dtype=np.float32)
        if self._wind_field is not None:
            wind_max = max(float(self.cfg.uncertainty.wind_max_speed), 1e-9)
            wind_vec = np.clip(self._wind_at(self._drone_xy[0]) / wind_max, -1.0, 1.0).astype(
                np.float32
            )
            K = min(self._wind_field.n_kernels, N_WIND_KERNELS)
            wind_samples[:K] = np.clip(
                self._wind_field.kernel_wind[:K] / wind_max, -1.0, 1.0
            ).astype(np.float32)

        if (
            self.cfg.uncertainty.enabled
            and self.cfg.uncertainty.payload_enabled
            and self.cfg.uncertainty.payload_lifetime_h > 1e-9
        ):
            elapsed = np.maximum(0.0, float(self.t) - self._drone_sortie_start_t)
            payload_left = np.maximum(0.0, self._drone_payload_remain - elapsed)
            drone_payload_remain = np.clip(
                payload_left / float(self.cfg.uncertainty.payload_lifetime_h), 0.0, 1.0
            ).astype(np.float32)
        else:
            drone_payload_remain = np.ones(self.ndrones, dtype=np.float32)

        return {
            "time_progress": np.array([np.clip(self.t / sc.max_time_h, 0.0, 1.0)], dtype=np.float32),
            "done_ratio": np.array([np.sum(self.task_done[:T]) / max(float(T), 1.0)], dtype=np.float32),
            "stop_xy": stop_xy,
            "stop_mask": stop_mask,
            "task_type": task_type,
            "task_mask": task_mask,
            "task_done": task_done,
            "task_spawned": task_spawned,
            "task_claimed": task_claimed,
            "point_xy": point_xy,
            "task_priority": task_priority,
            "task_deadline_h": task_deadline_h,
            "task_reward_weight": task_reward_weight,
            "task_urgency": task_urgency,
            "task_risk": task_risk,
            "truck_stop": self.truck_stop.astype(np.int32),
            "truck_xy": truck_xy,
            "truck_busy": self.truck_busy.astype(np.int8),
            "truck_eta_h": np.clip(self.truck_eta / sc.max_time_h, 0.0, 1.0).astype(np.float32),
            "truck_target_stop": self.truck_target.astype(np.int32),
            "drone_status": self.drone_status.astype(np.int8),
            "drone_carrier_truck": self.drone_carrier_truck.astype(np.int32),
            "drone_stop_id": self.drone_stop_id.astype(np.int32),
            "drone_task_id": self.drone_task_id.astype(np.int32),
            "drone_task_side": self.drone_task_side.astype(np.int8),
            "drone_phase": self.drone_phase.astype(np.int8),
            "drone_planned_recovery_stop": self._drone_planned_recovery_stop.astype(np.int32),
            "drone_return_truck": self._drone_return_truck.astype(np.int32),
            "drone_return_stop": self._drone_return_stop.astype(np.int32),
            "drone_eta_h": np.clip(self.drone_eta / sc.max_time_h, 0.0, 1.0).astype(np.float32),
            "drone_xy": norm_xy(self._drone_xy),
            "drone_batt_ratio": np.clip(
                self.drone_batt / max(float(self.cfg.max_battery), 1e-9), 0.0, 1.0
            ).astype(np.float32),
            "drone_payload_remain": drone_payload_remain,
            "drone_payload_failed": self._drone_payload_failed.astype(np.int8),
            "wind_vec": wind_vec,
            "wind_samples": wind_samples,
            "action_mask": self._compute_action_mask(),
        }

    def _get_info(self) -> Dict[str, object]:
        T = self.scenario.ntasks
        return {
            "time_h": float(self.t),
            "decision_count": int(self.decision_count),
            "task_done_count": int(self.task_done[:T].sum()),
            "task_spawned_count": int(self._task_spawned[:T].sum()),
            "task_total": int(T),
            "completion_rate": float(self.task_done[:T].sum()) / max(float(T), 1.0),
            "ntrucks": int(self.ntrucks),
            "ndrones": int(self.ndrones),
            "active_truck_count": int(np.sum(self.truck_busy)),
            "active_drone_count": int(np.sum(self.drone_status != DS_ON_TRUCK)),
            "executing_drone_count": int(np.sum(self.drone_status == DS_EXECUTING)),
            "ground_waiting_drone_count": int(np.sum(self.drone_status == DS_ON_GROUND)),
            "episode_swap_count": int(self.episode_swap_count),
            "episode_battery_shock_count": int(self.episode_battery_shock_count),
            "episode_payload_failure_count": int(self.episode_payload_failure_count),
            "episode_ground_wait_time_h": float(self.episode_ground_wait_time_h),
            "episode_energy_used": float(self.episode_energy_used),
            "episode_risk_exposure": float(self.episode_risk_exposure),
        }

    def close(self):
        pass
