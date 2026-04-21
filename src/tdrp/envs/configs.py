from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Scenario:
    scale: str
    map_range: float
    nstops: int
    # Final counts include future (to-be-spawned) tasks. *_init holds the counts already
    # present at t=0. Tasks in [0, npoint) are point slots; [npoint, ntasks) are line slots.
    # Future-task geometry/attributes are generated online when EV_NEW_TASK_SPAWN fires.
    npoint: int
    nline: int
    npoint_init: int
    nline_init: int
    max_time_h: float
    max_decisions: int
    stop_xy: np.ndarray
    truck_dist: np.ndarray
    task_type: np.ndarray
    point_xy: np.ndarray
    line_start_xy: np.ndarray
    line_end_xy: np.ndarray
    point_service_h: np.ndarray
    line_polyline_xy: np.ndarray
    line_polyline_npts: np.ndarray
    line_length: np.ndarray
    task_spawn_time: np.ndarray       # 0.0 = present at t=0; >0 = spawns mid-mission
    task_priority: np.ndarray         # 1=normal, 2=urgent, 3=critical
    task_deadline_h: np.ndarray       # absolute deadline in mission clock
    task_reward_weight: np.ndarray    # completion reward multiplier
    task_service_factor: np.ndarray   # extra workload factor (mainly for line tasks)
    task_risk: np.ndarray             # normalized risk score in [0, 1]

    @property
    def ntasks(self) -> int:
        return self.npoint + self.nline


@dataclass
class UncertaintyConfig:
    enabled: bool = False
    fixed_seed: Optional[int] = None

    # Wind field (spatially non-uniform, time-varying via OU process per kernel).
    wind_enabled: bool = True
    wind_intensity: float = 1.0
    wind_ou_theta: float = 1.0
    wind_ou_sigma: float = 7.0
    wind_sigma_frac: float = 0.35   # RBF sigma = wind_sigma_frac * map_range
    wind_max_speed: float = 15.0
    wind_hover_k: float = 0.15
    wind_speed_floor: float = 0.20

    # Battery shock at takeoff.
    battery_enabled: bool = True
    battery_intensity: float = 1.0
    battery_shock_std: float = 5.0

    # Payload degradation during sortie.
    payload_enabled: bool = True
    payload_intensity: float = 1.0
    payload_lifetime_h: float = 1.5
    payload_deg_std: float = 0.15


@dataclass
class RewardConfig:
    time_coef: float = 5.0
    task_done_bonus: float = 12.0
    early_finish_coef: float = 1.5
    deadline_miss_coef: float = 6.0


@dataclass
class EnvConfig:
    ndrones: int = 1
    depot_stop_id: int = 0

    road_network_type: str = "euclidean"
    truck_speed_kmph: float = 36.0
    drone_speed_kmph: float = 54.0
    line_speed_factor: float = 0.8

    max_battery: float = 100.0
    fly_power_per_h: float = 125.0
    hover_power_per_h: float = 135.0
    safety_batt_margin: float = 0.5

    takeoff_time_h: float = 0.5 / 60.0
    landing_time_h: float = 0.5 / 60.0
    swap_time_h: float = 2.0 / 60.0
    default_point_service_h: float = 0.05

    wait_step_h: float = 0.01
    exec_check_interval_h: float = 1.0 / 60.0
    eps_time: float = 1e-6
    eps_batt: float = 1e-9
    max_drift_steps: int = 200

    fixed_scale: Optional[str] = None
    scenario_max_global_retry: int = 20
    scenario_max_task_retry: int = 50

    # Heterogeneous task settings.
    heter_task_enabled: bool = True
    urgent_task_ratio: float = 0.20
    critical_task_ratio: float = 0.08
    point_service_h_min: float = 0.04
    point_service_h_max: float = 0.12
    line_service_factor_min: float = 0.90
    line_service_factor_max: float = 1.30
    deadline_slack_min_frac: float = 0.20
    deadline_slack_max_frac: float = 0.80

    # Dynamic task spawning (4th uncertainty source: tasks appear mid-mission).
    allow_dynamic_tasks: bool = True
    dynamic_task_ratio: float = 0.3             # fraction of initial tasks added as future tasks
    dynamic_spawn_t_min_frac: float = 0.05      # earliest spawn as fraction of max_time_h
    dynamic_spawn_t_max_frac: float = 0.70      # latest spawn
    dynamic_generate_online: bool = True        # generate geometry/attributes on spawn (not at reset)

    reward: RewardConfig = field(default_factory=RewardConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
