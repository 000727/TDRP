import numpy as np

from tdrp.envs import EnvConfig, MultiTruckMultiDroneEnv
from tdrp.envs.configs import RewardConfig, UncertaintyConfig


def first_task_action(env, obs):
    action = [env.AWAIT] * env.nagents
    row = obs["action_mask"][env.ntrucks]
    task_actions = [
        int(a)
        for a in np.flatnonzero(row)
        if env.act_offset_task <= int(a) < env.AWAIT
    ]
    assert task_actions
    action[env.ntrucks] = task_actions[0]
    return action


def rollout_one_reward(reward_model_name):
    cfg = EnvConfig(
        fixed_scale="XS",
        ntrucks=1,
        ndrones=1,
        allow_dynamic_tasks=False,
        reward_model_name=reward_model_name,
        reward=RewardConfig(energy_coef=10.0, risk_coef=10.0),
        uncertainty=UncertaintyConfig(enabled=False),
    )
    env = MultiTruckMultiDroneEnv(cfg)
    obs, _ = env.reset(seed=11)
    _, reward, _, _, info = env.step(first_task_action(env, obs))
    return float(reward), env.step_events.copy(), info


def test_reward_models_run_in_multi_env():
    for name in [
        "time_completion",
        "completion",
        "decay_value",
        "deadline",
        "energy_aware",
        "risk_sensitive",
    ]:
        reward, events, info = rollout_one_reward(name)

        assert np.isfinite(reward)
        assert info["decision_count"] == 1
        assert "energy_used" in events
        assert "risk_exposure" in events


def test_energy_and_risk_rewards_change_same_transition_reward():
    base_reward, _, _ = rollout_one_reward("time_completion")
    energy_reward, energy_events, _ = rollout_one_reward("energy_aware")
    risk_reward, risk_events, _ = rollout_one_reward("risk_sensitive")

    assert energy_events["energy_used"] > 0.0
    assert risk_events["risk_exposure"] > 0.0
    assert energy_reward < base_reward
    assert risk_reward < base_reward


def test_decaying_value_reward_decreases_with_waiting_time():
    cfg = EnvConfig(
        fixed_scale="XS",
        ntrucks=1,
        ndrones=1,
        allow_dynamic_tasks=False,
        reward_model_name="decay_value",
        uncertainty=UncertaintyConfig(enabled=False),
    )
    env = MultiTruckMultiDroneEnv(cfg)
    env.reset(seed=11)
    tid = 0
    env.scenario.task_reward_weight[tid] = 1.0
    env.scenario.task_urgency[tid] = 1.0
    env.scenario.task_spawn_time[tid] = 0.0

    early = env.reward_model.task_deadline_adjustment(env, tid, 0.1)
    late = env.reward_model.task_deadline_adjustment(env, tid, 1.0)

    assert early > late
    assert late > 0.0
