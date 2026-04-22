import numpy as np

from tdrp.envs import EnvConfig, MultiTruckMultiDroneEnv, TruckMultiDroneCleanEnv, scenario_from_dict
from tdrp.envs.configs import UncertaintyConfig
from tdrp.envs.constants import DS_ON_GROUND, DS_ON_TRUCK
from tdrp.envs.multi_env import PH_SERVICE
from tdrp.uncertainty import UncertaintyModule


class FixedBatteryShock(UncertaintyModule):
    name = "fixed_battery_shock"

    def __init__(self, amount):
        self.amount = float(amount)

    def on_takeoff(self, env, drone_id):
        env.drone_batt[drone_id] = max(0.0, float(env.drone_batt[drone_id]) - self.amount)
        return {"battery_shock": self.amount}


class ConstantWindField:
    n_kernels = 1

    def __init__(self, wind):
        self.wind = np.asarray(wind, dtype=np.float64)
        self.kernel_wind = self.wind.reshape(1, 2)
        self.centers = np.zeros((1, 2), dtype=np.float64)

    def get(self, xy):
        return self.wind.copy()

    def advance(self, dt):
        return None


def choose_masked_action(env, obs):
    """Simple deterministic policy for environment invariants, not performance."""
    mask = obs["action_mask"]
    ntrucks = getattr(env, "ntrucks", 1)
    any_drone_task = False
    for row_i in range(ntrucks, mask.shape[0]):
        legal = np.flatnonzero(mask[row_i])
        task_legal = legal[(legal >= env.act_offset_task) & (legal < env.AWAIT)]
        if task_legal.size:
            any_drone_task = True
            break

    action = []
    for row_i, row in enumerate(mask):
        legal = np.flatnonzero(row)
        assert legal.size > 0
        non_wait = legal[legal != env.AWAIT]
        if row_i < ntrucks:
            if any_drone_task and env.AWAIT in legal:
                action.append(int(env.AWAIT))
            elif non_wait.size:
                action.append(int(non_wait[0]))
            else:
                action.append(int(env.AWAIT))
            continue

        task_legal = legal[(legal >= env.act_offset_task) & (legal < env.AWAIT)]
        if task_legal.size:
            action.append(int(task_legal[0]))
        elif non_wait.size:
            action.append(int(non_wait[0]))
        else:
            action.append(int(env.AWAIT))
    return action


def run_invariant_rollout(env, seed=7, max_steps=120):
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs)
    last_t = float(info.get("time_h", 0.0))
    last_done = int(info.get("task_done_count", 0))
    last_spawned = int(info.get("task_spawned_count", 0))

    terminated = truncated = False
    for step in range(max_steps):
        obs, _, terminated, truncated, info = env.step(choose_masked_action(env, obs))
        assert env.observation_space.contains(obs)
        now_t = float(info.get("time_h", env.t))
        done = int(info.get("task_done_count", 0))
        spawned = int(info.get("task_spawned_count", 0))

        assert now_t + 1e-9 >= last_t
        assert done >= last_done
        assert spawned >= last_spawned
        assert np.all(env.drone_batt >= -env.cfg.eps_batt)

        last_t, last_done, last_spawned = now_t, done, spawned
        if terminated or truncated:
            break

    return info, terminated, truncated, step + 1


def test_legacy_static_instance_reaches_success():
    env = TruckMultiDroneCleanEnv(
        EnvConfig(
            fixed_scale="XS",
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )

    info, terminated, truncated, _ = run_invariant_rollout(env)

    assert terminated
    assert not truncated
    assert info["end_reason"] == "success"
    assert info["task_done_count"] == info["task_total"]


def test_legacy_all_uncertainties_instance_runs_to_time_limit():
    env = TruckMultiDroneCleanEnv(
        EnvConfig(
            fixed_scale="XS",
            allow_dynamic_tasks=True,
            dynamic_task_ratio=0.5,
            uncertainty=UncertaintyConfig(
                enabled=True,
                fixed_seed=123,
                wind_enabled=True,
                battery_enabled=True,
                payload_enabled=True,
                battery_shock_base_prob=1.0,
                payload_probabilistic_failure=True,
                payload_failure_base_prob=0.05,
            ),
        )
    )

    info, terminated, truncated, _ = run_invariant_rollout(env)

    assert not terminated
    assert truncated
    assert info["end_reason"] == "time_limit"
    assert info["task_spawned_count"] == info["task_total"]
    assert info["task_done_count"] > 0


def test_multi_truck_static_instance_runs_with_valid_state():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=2,
            ndrones=3,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )

    info, terminated, truncated, _ = run_invariant_rollout(env)

    assert terminated or truncated
    assert info["ntrucks"] == 2
    assert info["ndrones"] == 3
    assert info["task_done_count"] > 0


def test_multi_truck_all_uncertainties_instance_runs_with_valid_state():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=2,
            ndrones=3,
            allow_dynamic_tasks=True,
            dynamic_task_ratio=0.5,
            uncertainty=UncertaintyConfig(
                enabled=True,
                fixed_seed=123,
                wind_enabled=True,
                battery_enabled=True,
                payload_enabled=True,
                battery_shock_base_prob=1.0,
                payload_probabilistic_failure=True,
                payload_failure_base_prob=0.05,
            ),
        )
    )

    info, _, truncated, _ = run_invariant_rollout(env)

    assert truncated
    assert info["task_spawned_count"] == info["task_total"]
    assert info["task_done_count"] > 0


def test_multi_truck_dynamic_task_spawn_is_event_timed():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=True,
            dynamic_task_ratio=0.5,
            dynamic_spawn_t_min_frac=0.20,
            dynamic_spawn_t_max_frac=0.20,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    env.reset(seed=7)
    future_times = env.scenario.task_spawn_time[
        env.scenario.task_spawn_time > env.cfg.eps_time
    ]
    assert future_times.size > 0
    first_spawn = float(np.min(future_times))

    _, _, _, _, info = env.step([env.AWAIT] * env.nagents)

    assert abs(float(info["time_h"]) - first_spawn) <= 1e-6
    assert info["task_spawned_count"] > env.scenario.npoint_init + env.scenario.nline_init


def test_poisson_dynamic_arrival_process_generates_scheduled_tasks():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=True,
            dynamic_task_ratio=1.5,
            dynamic_arrival_process="poisson",
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    env.reset(seed=7)
    future_times = env.scenario.task_spawn_time[
        env.scenario.task_spawn_time > env.cfg.eps_time
    ]

    assert future_times.size > 0
    assert np.all(np.diff(np.sort(future_times)) >= 0.0)
    assert env.scenario.ntasks <= 128


def test_imported_scenario_data_can_reset_environment():
    data = {
        "scale": "tiny-import",
        "map_range": 2.0,
        "max_time_h": 1.0,
        "max_decisions": 20,
        "stop_xy": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "tasks": [
            {
                "type": "point",
                "xy": [0.4, 0.2],
                "service_h": 0.01,
                "spawn_time": 0.0,
                "reward_weight": 1.7,
                "urgency": 0.3,
            },
            {
                "type": "line",
                "polyline_xy": [[0.2, 0.8], [0.8, 0.8]],
                "spawn_time": 0.3,
                "reward_weight": 2.0,
                "urgency": 0.6,
            },
        ],
    }
    scenario = scenario_from_dict(data, EnvConfig())
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            ntrucks=1,
            ndrones=1,
            scenario_data=data,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    obs, info = env.reset(seed=7)

    assert env.scenario.scale == scenario.scale == "tiny-import"
    assert info["task_total"] == 2
    assert info["task_spawned_count"] == 1
    assert np.isclose(env.scenario.task_reward_weight[0], 1.7)
    assert np.isclose(env.scenario.task_urgency[1], 0.6)
    assert env.observation_space.contains(obs)


def test_multi_truck_env_keeps_concurrent_drone_sorties_active():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=2,
            ndrones=3,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    obs, _ = env.reset(seed=7)
    action = [env.AWAIT] * env.nagents
    used_tasks = set()
    for d in range(env.ndrones):
        row = obs["action_mask"][env.ntrucks + d]
        for a in np.flatnonzero(row):
            if not (env.act_offset_task <= int(a) < env.AWAIT):
                continue
            tid, _ = env._decode_task_id_and_side(int(a))
            if tid in used_tasks:
                continue
            action[env.ntrucks + d] = int(a)
            used_tasks.add(tid)
            break

    obs, _, _, _, info = env.step(action)

    assert info["task_done_count"] == 0
    assert info["active_drone_count"] == len(used_tasks)
    assert int(obs["task_claimed"].sum()) == len(used_tasks)
    assert np.any(obs["drone_eta_h"] > 0.0)
    assert np.all(obs["drone_phase"][: len(used_tasks)] >= 0)

    for _ in range(20):
        obs, _, _, _, info = env.step([env.AWAIT] * env.nagents)
        if info["ground_waiting_drone_count"] > 0:
            break

    assert info["task_done_count"] > 0
    assert info["ground_waiting_drone_count"] > 0


def test_multi_truck_position_interpolates_while_moving():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=2,
            ndrones=2,
            drones_per_truck=(1, 1),
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    obs, _ = env.reset(seed=7)
    start_xy = env._truck_xy[0].copy()
    target_stop = int(np.flatnonzero(obs["action_mask"][0])[0])
    target_xy = env.scenario.stop_xy[target_stop].copy()

    action = [env.AWAIT] * env.nagents
    action[0] = target_stop
    row = obs["action_mask"][env.ntrucks + 1]
    task_actions = [
        int(a)
        for a in np.flatnonzero(row)
        if env.act_offset_task <= int(a) < env.AWAIT
    ]
    assert task_actions
    action[env.ntrucks + 1] = task_actions[0]

    env.step(action)

    assert int(env.truck_busy[0]) == 1
    assert np.linalg.norm(env._truck_xy[0] - start_xy) > 0.0
    assert np.linalg.norm(env._truck_xy[0] - target_xy) > 0.0


def test_multi_truck_recovery_policy_can_assign_non_carrier_truck():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=2,
            ndrones=1,
            drones_per_truck=(1, 0),
            allow_dynamic_tasks=False,
            recovery_policy_name="min_total_time",
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    obs, _ = env.reset(seed=7)
    carrier = int(env.drone_carrier_truck[0])
    chosen_action = None
    chosen_stop = None
    row = obs["action_mask"][env.ntrucks]
    for a in np.flatnonzero(row):
        if not (env.act_offset_task <= int(a) < env.AWAIT):
            continue
        tid, side = env._decode_task_id_and_side(int(a))
        _, exit_xy = env._task_entry_exit_xy(tid, side)
        stop = env._nearest_stop_id(exit_xy)
        if stop != int(env.truck_stop[carrier]):
            chosen_action = int(a)
            chosen_stop = int(stop)
            break
    assert chosen_action is not None

    env.truck_stop[1] = int(chosen_stop)
    env.truck_target[1] = int(chosen_stop)
    env._truck_xy[1] = env.scenario.stop_xy[int(chosen_stop)]

    action = [env.AWAIT] * env.nagents
    action[env.ntrucks] = int(chosen_action)
    obs, _, _, _, _ = env.step(action)

    assert int(obs["drone_return_truck"][0]) == 1
    assert int(obs["drone_return_stop"][0]) == int(chosen_stop)


def test_multi_truck_drone_can_plan_recovery_stop_before_task():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=2,
            ndrones=1,
            drones_per_truck=(1, 0),
            allow_dynamic_tasks=False,
            recovery_policy_name="min_total_time",
            enable_recovery_stop_actions=True,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    obs, _ = env.reset(seed=7)
    planned_stop = int((int(env.truck_stop[0]) + 1) % env.scenario.nstops)
    row = obs["action_mask"][env.ntrucks]
    assert int(row[planned_stop]) == 1

    obs, _, _, _, _ = env.step([env.AWAIT, env.AWAIT, planned_stop])

    assert int(obs["drone_planned_recovery_stop"][0]) == planned_stop

    row = obs["action_mask"][env.ntrucks]
    task_action = int(
        next(a for a in np.flatnonzero(row) if env.act_offset_task <= int(a) < env.AWAIT)
    )
    obs, _, _, _, _ = env.step([env.AWAIT, env.AWAIT, task_action])

    assert int(obs["drone_return_stop"][0]) == planned_stop
    assert int(obs["drone_planned_recovery_stop"][0]) == -1


def test_multi_truck_payload_can_fail_mid_sortie():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(
                enabled=True,
                fixed_seed=123,
                wind_enabled=False,
                battery_enabled=False,
                payload_enabled=True,
                payload_intensity=1.0,
                payload_lifetime_h=0.01,
                payload_deg_std=0.0,
            ),
        )
    )
    obs, _ = env.reset(seed=7)
    action = [env.AWAIT] * env.nagents
    row = obs["action_mask"][env.ntrucks]
    action[env.ntrucks] = int(
        next(a for a in np.flatnonzero(row) if env.act_offset_task <= int(a) < env.AWAIT)
    )

    info = {}
    for _ in range(10):
        obs, _, _, _, info = env.step(action)
        action = [env.AWAIT] * env.nagents
        if info["episode_payload_failure_count"] > 0:
            break

    assert info["episode_payload_failure_count"] > 0
    assert info["task_done_count"] == 0


def test_multi_truck_battery_can_deplete_mid_sortie():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            safety_batt_margin=0.5,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(enabled=False),
        )
    )
    obs, _ = env.reset(seed=7)
    action = [env.AWAIT] * env.nagents
    row = obs["action_mask"][env.ntrucks]
    task_actions = [
        int(a)
        for a in np.flatnonzero(row)
        if env.act_offset_task <= int(a) < env.AWAIT
    ]
    assert task_actions
    action[env.ntrucks] = task_actions[0]

    obs, _, _, _, _ = env.step(action)
    assert int(obs["drone_status"][0]) != 0
    env.drone_batt[0] = 0.6

    saw_depletion = False
    for _ in range(10):
        obs, _, _, _, _ = env.step([env.AWAIT] * env.nagents)
        if float(env.step_events.get("battery_depleted_count", 0.0)) > 0.0:
            saw_depletion = True
            break

    assert saw_depletion
    assert int(obs["task_done"].sum()) == 0


def test_multi_truck_battery_shock_reduces_energy_without_direct_failure():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(enabled=False),
            uncertainty_modules=[FixedBatteryShock(amount=10.0)],
        )
    )
    obs, _ = env.reset(seed=7)
    action = [env.AWAIT] * env.nagents
    row = obs["action_mask"][env.ntrucks]
    action[env.ntrucks] = int(
        next(a for a in np.flatnonzero(row) if env.act_offset_task <= int(a) < env.AWAIT)
    )

    obs, _, _, _, info = env.step(action)

    assert info["episode_battery_shock_count"] == 1
    assert float(env.drone_batt[0]) < float(env.cfg.max_battery)
    assert float(env.drone_batt[0]) > float(env.cfg.safety_batt_margin)
    assert int(obs["drone_status"][0]) not in (DS_ON_TRUCK, DS_ON_GROUND)
    assert float(env.step_events.get("battery_depleted_count", 0.0)) == 0.0
    assert int(obs["task_claimed"].sum()) == 1


def test_wind_angle_changes_flight_time_and_power():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(
                enabled=True,
                fixed_seed=123,
                wind_enabled=True,
                battery_enabled=False,
                payload_enabled=False,
                wind_intensity=1.0,
                wind_crosswind_speed_k=0.2,
                wind_headwind_power_k=0.2,
                wind_crosswind_power_k=0.3,
            ),
        )
    )
    env.reset(seed=7)
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    base_time = np.linalg.norm(end - start) / env.cfg.drone_speed_kmph
    base_power = env.cfg.fly_power_per_h

    env._wind_field = ConstantWindField([10.0, 0.0])
    tail_time = env._effective_flight_time(start, end)
    tail_power = env._effective_flight_power(start, end)

    env._wind_field = ConstantWindField([-10.0, 0.0])
    head_time = env._effective_flight_time(start, end)
    head_power = env._effective_flight_power(start, end)

    env._wind_field = ConstantWindField([0.0, 10.0])
    cross_time = env._effective_flight_time(start, end)
    cross_power = env._effective_flight_power(start, end)

    assert tail_time < base_time
    assert head_time > base_time
    assert cross_time > base_time
    assert tail_power == base_power
    assert head_power > base_power
    assert cross_power > base_power


def test_multi_truck_payload_failure_releases_unfinished_task():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(
                enabled=True,
                fixed_seed=123,
                wind_enabled=False,
                battery_enabled=False,
                payload_enabled=True,
                payload_intensity=1.0,
                payload_lifetime_h=0.01,
                payload_deg_std=0.0,
            ),
        )
    )
    obs, _ = env.reset(seed=7)
    action = [env.AWAIT] * env.nagents
    row = obs["action_mask"][env.ntrucks]
    tid, _ = env._decode_task_id_and_side(
        int(next(a for a in np.flatnonzero(row) if env.act_offset_task <= int(a) < env.AWAIT))
    )
    action[env.ntrucks] = env.act_offset_task + int(
        next(
            k
            for k in range(env._task_action_count)
            if int(env._task_action_tid[k]) == tid
        )
    )

    info = {}
    for _ in range(10):
        obs, _, _, _, info = env.step(action)
        action = [env.AWAIT] * env.nagents
        if info["episode_payload_failure_count"] > 0:
            break

    assert info["episode_payload_failure_count"] > 0
    assert int(obs["task_done"][tid]) == 0
    assert int(obs["task_claimed"][tid]) == 0


def test_multi_truck_line_failure_preserves_remaining_progress():
    env = MultiTruckMultiDroneEnv(
        EnvConfig(
            fixed_scale="XS",
            ntrucks=1,
            ndrones=1,
            allow_dynamic_tasks=False,
            uncertainty=UncertaintyConfig(
                enabled=True,
                fixed_seed=123,
                wind_enabled=False,
                battery_enabled=False,
                payload_enabled=True,
                payload_intensity=1.0,
                payload_lifetime_h=10.0,
                payload_deg_std=0.0,
            ),
        )
    )
    obs, _ = env.reset(seed=7)
    line_action = None
    line_tid = None
    row = obs["action_mask"][env.ntrucks]
    for a in np.flatnonzero(row):
        if not (env.act_offset_task <= int(a) < env.AWAIT):
            continue
        tid, _ = env._decode_task_id_and_side(int(a))
        if int(env.scenario.task_type[tid]) == 1:
            line_action = int(a)
            line_tid = int(tid)
            break
    assert line_action is not None
    original_len = float(env._line_remain_len[line_tid])

    action = [env.AWAIT] * env.nagents
    action[env.ntrucks] = line_action
    for _ in range(20):
        obs, _, _, _, _ = env.step(action)
        action = [env.AWAIT] * env.nagents
        if int(env.drone_phase[0]) == PH_SERVICE:
            break

    assert int(env.drone_phase[0]) == PH_SERVICE
    fail_after = max(float(env._drone_service_h[0]) * 0.4, env.cfg.exec_check_interval_h)
    env._drone_payload_remain[0] = float(env.t) - float(env._drone_sortie_start_t[0]) + fail_after

    info = {}
    for _ in range(10):
        obs, _, _, _, info = env.step([env.AWAIT] * env.nagents)
        if info["episode_payload_failure_count"] > 0:
            break

    assert info["episode_payload_failure_count"] > 0
    assert int(obs["task_done"][line_tid]) == 0
    assert int(obs["task_claimed"][line_tid]) == 0
    assert 0.0 < float(env._line_remain_len[line_tid]) < original_len
