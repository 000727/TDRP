# Roadmap

## Phase 0: Repository Foundation

- Define project scope and MVP problem.
- Establish package structure, config conventions, and docs.
- Add smoke tests for imports and configuration loading.

## Phase 1: Simulation Environment

- Implement node, vehicle, UAV, and task data structures.
- Implement deterministic small-instance simulator.
- Add stochastic travel-time and service-time models.
- Define reward and termination conditions.
- Add visualization or rollout trace export.

## Phase 2: Baselines

- Nearest-neighbor vehicle-only baseline.
- Greedy vehicle-UAV assignment baseline.
- Simple insertion or dispatch heuristic.
- Random policy baseline for RL sanity checks.

## Phase 3: Reinforcement Learning

- Wrap simulator in a Gymnasium-style API.
- Implement policy observation/action encoding.
- Train a first PPO/DQN-style baseline depending on action design.
- Compare under deterministic and uncertain scenarios.

## Phase 4: Robust and Adaptive Planning

- Add scenario sampling and domain randomization.
- Evaluate distribution shift.
- Add risk-sensitive objective terms.
- Explore hierarchical or multi-agent policy structures.

## Phase 5: Experiments and Paper Assets

- Freeze benchmark sets.
- Generate reproducible tables and plots.
- Document ablation studies.
- Package experiment scripts and configs.
