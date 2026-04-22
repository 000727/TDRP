from .configs import EnvConfig, RewardConfig, Scenario, UncertaintyConfig
from .fleet import FleetRuntimeState, FleetSpec, DroneRuntimeState, TruckRuntimeState
from .scenario_io import load_scenario_json, scenario_from_dict


def __getattr__(name):
    if name == "TruckMultiDroneCleanEnv":
        from .env import TruckMultiDroneCleanEnv

        return TruckMultiDroneCleanEnv
    if name == "MultiTruckMultiDroneEnv":
        from .multi_env import MultiTruckMultiDroneEnv

        return MultiTruckMultiDroneEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MultiTruckMultiDroneEnv",
    "TruckMultiDroneCleanEnv",
    "EnvConfig",
    "FleetRuntimeState",
    "FleetSpec",
    "DroneRuntimeState",
    "RewardConfig",
    "Scenario",
    "load_scenario_json",
    "scenario_from_dict",
    "TruckRuntimeState",
    "UncertaintyConfig",
]
