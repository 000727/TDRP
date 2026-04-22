from tdrp.envs import EnvConfig, FleetRuntimeState, FleetSpec
from tdrp.rewards import (
    CompletionReward,
    DecayingValueReward,
    DeadlineReward,
    EnergyAwareReward,
    RewardModel,
    RiskSensitiveReward,
    TimeCompletionReward,
    build_reward_model,
)
from tdrp.uncertainty import UncertaintyModule, build_uncertainty_modules


class CustomUncertainty(UncertaintyModule):
    name = "custom"


class CustomReward(RewardModel):
    name = "custom_reward"


def test_default_uncertainty_modules_are_registered():
    cfg = EnvConfig()

    modules = build_uncertainty_modules(cfg)
    names = {module.name for module in modules}

    assert {
        "dynamic_wind",
        "dynamic_task_spawn",
        "battery_aging_shock",
        "payload_reliability",
    }.issubset(names)


def test_custom_uncertainty_modules_override_defaults():
    custom = CustomUncertainty()
    cfg = EnvConfig(uncertainty_modules=[custom])

    assert build_uncertainty_modules(cfg) == [custom]


def test_default_and_custom_reward_models():
    assert isinstance(build_reward_model(EnvConfig()), TimeCompletionReward)

    custom = CustomReward()
    cfg = EnvConfig(reward_model=custom)

    assert build_reward_model(cfg) is custom


def test_named_reward_models_can_be_built():
    expected = {
        "time_completion": TimeCompletionReward,
        "completion": CompletionReward,
        "decay_value": DecayingValueReward,
        "deadline": DeadlineReward,
        "energy_aware": EnergyAwareReward,
        "risk_sensitive": RiskSensitiveReward,
    }

    for name, cls in expected.items():
        assert isinstance(build_reward_model(EnvConfig(reward_model_name=name)), cls)


def test_fleet_spec_maps_drones_to_carrier_trucks():
    spec = FleetSpec(ntrucks=2, drones_per_truck=(2, 1))

    assert spec.ndrones == 3
    assert [spec.carrier_for_drone(i) for i in range(3)] == [0, 0, 1]

    state = FleetRuntimeState.from_spec(spec, depot_stop_id=0, max_battery=100.0)

    assert len(state.trucks) == 2
    assert len(state.drones) == 3
    assert state.drones[2].carrier_truck_id == 1


def test_fleet_spec_can_be_derived_from_env_config():
    cfg = EnvConfig(ntrucks=2, ndrones=5)

    spec = FleetSpec.from_config(cfg)

    assert spec.ntrucks == 2
    assert spec.drones_per_truck == (3, 2)
    assert spec.ndrones == 5
