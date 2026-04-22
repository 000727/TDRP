# Modular Environment Design

This project keeps the Gymnasium environment as the owner of the core state
machine, while uncertainty and reward behavior are plugged in through small
modules.

## Core Responsibility

`TruckMultiDroneCleanEnv` owns:

- reset and step lifecycle
- action decoding and action masks
- event queue and event handlers
- truck/drone/task runtime state
- feasibility checks and terminal conditions

The core environment should not hard-code every research variant. Instead, it
calls uncertainty and reward hooks at stable insertion points.

`tdrp.envs.fleet` provides the first fleet-state skeleton for future multi-truck
support:

- `FleetSpec`: static truck/drone layout
- `TruckRuntimeState`: per-truck runtime state
- `DroneRuntimeState`: per-drone runtime state
- `FleetRuntimeState`: combined runtime container

`MultiTruckMultiDroneEnv` is the first Gymnasium-compatible multi-fleet
skeleton. It is intentionally smaller than the legacy event simulator, but it
now uses a staged event advance. A drone task action expands into:

- takeoff
- cruise to task entry
- task service or line traversal
- cruise to recovery stop
- landing to ground-wait state
- truck recovery and swap

The environment advances to the next completed event, and still-busy agents
remain visible through observation fields such as `truck_eta_h`,
`truck_target_stop`, `drone_phase`, `drone_task_id`, `drone_eta_h`, and
`task_claimed`. Planned recovery is visible through `drone_return_truck` and
`drone_return_stop`. Energy is integrated during each active phase, and cumulative
metrics such as `episode_energy_used`, `episode_risk_exposure`,
`episode_ground_wait_time_h`, and `episode_swap_count` are exposed in `info`.
Truck positions are interpolated continuously while moving, so `truck_xy` is a
spatial state rather than only the last reached stop.
Active drone phases also consider failure event times. If payload lifetime
expires or battery drops below the safety margin before the current phase ETA,
the sortie aborts at that intermediate time, the task claim is released, and
the drone is routed to the nearest recovery stop for truck pickup or immediate
swap if a truck is already there.
Future task spawn times are also first-class event candidates, so an idle
environment advances directly to the next scheduled task appearance instead of
polling by fixed wait steps. Line-task service follows the remaining polyline:
if a drone fails while traversing a line, the completed prefix is removed and
the unfinished suffix remains available for later reassignment.
Recovery planning can be configured with `recovery_policy_name`. The default
`nearest_stop` preserves the original behavior. `min_total_time` chooses the
truck/stop pair that minimizes recovery synchronization time across the fleet,
and `min_energy` chooses the lowest drone return-energy stop.
When `enable_recovery_stop_actions` is explicitly set to true, a drone that is still on a truck
can use a stop action to set `drone_planned_recovery_stop` before selecting a
task. The next task action consumes that planned stop as the sortie recovery
point, while the environment still chooses the recovery truck from the active
policy.
It already uses the same uncertainty and reward modules as the legacy
environment.

## Uncertainty Modules

Uncertainty modules inherit from `tdrp.uncertainty.UncertaintyModule`.

Available default modules:

- `DynamicWindModule`: time-varying and spatially non-uniform wind
- `DynamicTaskSpawnModule`: online task geometry generation when spawn events fire
- `BatteryAgingShockModule`: battery shocks with optional age-dependent probability
- `PayloadReliabilityModule`: payload lifetime and optional work-age failure probability

The environment calls these hooks:

- `on_reset(env)`
- `on_time_advance(env, dt_h)`
- `flight_time(env, from_xy, to_xy, base_speed_kmph, current_h)`
- `hover_power(env, xy, current_power_per_h)`
- `flight_power(env, from_xy, to_xy, base_power_per_h, current_power_per_h)`
- `on_takeoff(env, drone_id)`
- `on_sortie_start(env, drone_id)`
- `payload_failed(env, drone_id)`
- `on_new_task_spawn(env, task_id)`

To add a new uncertainty source, implement only the hooks it needs and pass the
module through `EnvConfig(uncertainty_modules=[...])`.

`DynamicWindModule` uses a 2D wind vector. For each flight segment, the wind is
decomposed into along-track and cross-track components. Tailwind shortens
flight time, headwind lengthens it and increases cruise power, and crosswind
both slightly reduces effective progress speed and increases cruise power.
Hover power depends on local wind magnitude.

## Reward Models

Reward models inherit from `tdrp.rewards.RewardModel`.

The default is `TimeCompletionReward`, which preserves the existing behavior:

- normalized time penalty
- weighted task completion reward
- deadline early/late adjustment
- rendezvous shaping from environment events

To use a different objective, pass either `EnvConfig(reward_model=...)` for a
custom object or `EnvConfig(reward_model_name=...)` for a built-in model.

Built-in reward model names:

- `time_completion`: time cost, completion bonus, deadline adjustment, rendezvous shaping
- `completion`: completion-only task reward
- `decay_value`: task completion value decays with task-specific waiting time and urgency
- `deadline`: time cost plus task/deadline reward
- `energy_aware`: default reward plus normalized energy penalty
- `risk_sensitive`: default reward plus risk exposure and failure penalties

The environments expose reward events such as `energy_used`, `risk_exposure`,
`payload_failure_count`, and `battery_depleted_count` through `step_events`.

`decay_value` implements the first project objective directly:

```text
value_i(t) = task_done_bonus * task_reward_weight_i
             * exp(-task_urgency_i * (t - task_spawn_time_i)) / ntasks
```

The value is clipped below by `reward.min_task_value_frac` to avoid completely
zeroing a task. The observation includes both `task_reward_weight` and
normalized `task_urgency`.

## Scenario Sources

Training can use either generated or imported scenarios.

- Random generation is the default through `ScenarioGenerator`.
- External data can be passed as `EnvConfig(scenario_data=...)`.
- External JSON can be passed as `EnvConfig(scenario_path="...json")`.
- `reset(options={"scenario": scenario})` still overrides both for direct
  experiment control.

The minimal imported schema is:

```json
{
  "stop_xy": [[0.0, 0.0], [1.0, 0.0]],
  "tasks": [
    {
      "type": "point",
      "xy": [0.4, 0.2],
      "service_h": 0.02,
      "spawn_time": 0.0,
      "reward_weight": 1.5,
      "urgency": 0.3
    },
    {
      "type": "line",
      "polyline_xy": [[0.2, 0.8], [0.8, 0.8]],
      "spawn_time": 0.5,
      "reward_weight": 2.0,
      "urgency": 0.6
    }
  ]
}
```

Dynamic task arrival is controlled by `dynamic_arrival_process`.
`uniform_count` preserves the previous behavior: a deterministic number of
future tasks is sampled uniformly in the configured time window. `poisson`
samples the number of future point and line tasks from Poisson distributions
whose means are controlled by `dynamic_task_ratio`; conditional on those
counts, arrival times are sorted uniform samples in the spawn window.

## Extension Rule

New research behavior should usually enter through one of these modules before
editing the core event state machine. Edit the core only when the physical
process itself changes, such as adding a new vehicle state, a new action type,
or a new event type.
