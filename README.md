# TDRP

TDRP is a research codebase for vehicle-UAV collaborative path planning under uncertainty. The project focuses on simulation environments and reinforcement learning algorithms for coordinated routing decisions when travel time, service time, task appearance, wind, battery behavior, and payload reliability are uncertain.

## Research Scope

- Vehicle-UAV collaborative routing and task allocation.
- Simulation environment design for coupled ground-air operations.
- Uncertainty modeling for wind, battery shock, payload degradation, dynamic tasks, deadlines, and heterogeneous task risk.
- Reinforcement learning algorithms for robust and adaptive decision-making.
- Benchmarking against heuristic, optimization, and learning-based baselines.

## Current Implementation

The repository now includes an initial simulation environment migrated from the local project:

- `TruckMultiDroneCleanEnv`: Gymnasium-compatible truck-multi-drone environment.
- `MultiTruckMultiDroneEnv`: modular multi-truck, multi-drone turn-based skeleton.
- Scenario generation for point and line tasks.
- Heterogeneous task priority, deadline, reward weight, service factor, and risk attributes.
- Uncertainty components for wind, battery shock, and payload degradation.
- Event-driven state transitions for truck movement, drone takeoff/landing, task execution, swaps, and dynamic task spawning.
- A Maskable PPO training script under `src/tdrp/algorithms/`.
- Pluggable uncertainty modules and reward models for modular environment maintenance.
- Built-in reward objectives for completion, deadline, energy-aware, and risk-sensitive studies.

## Repository Layout

```text
configs/                 Experiment and environment configuration files
docs/                    Project notes, design decisions, and research plans
src/tdrp/envs/           Simulation environment, configs, constants, scenarios, wind model
src/tdrp/uncertainty/    Pluggable wind, task-spawn, battery, and payload uncertainty modules
src/tdrp/rewards/        Replaceable reward models for different research objectives
src/tdrp/algorithms/     RL policies, trainers, and baselines
src/tdrp/evaluation/     Metrics, rollout evaluation, and comparisons
src/tdrp/utils/          Shared utilities
tests/                   Unit and smoke tests
```

## Install

```powershell
python -m pip install -e .[dev]
```

For RL training dependencies:

```powershell
python -m pip install -e .[dev,rl]
```

## Next Steps

1. Add smoke tests for `TruckMultiDroneCleanEnv.reset()` and action masks.
2. Refactor the PPO script into smaller trainer/model/rollout modules.
3. Create reproducible experiment entry points under `experiments/`.
4. Add baseline heuristic policies.
5. Add evaluation scripts for deterministic and uncertainty-shifted scenarios.

See `docs/MODULAR_ENVIRONMENT.md` for the environment extension points.
