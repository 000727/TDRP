# TDRP

TDRP is a research codebase for vehicle-UAV collaborative path planning under uncertainty. The project focuses on building simulation environments and reinforcement learning algorithms for coordinated routing decisions when travel time, service time, demand, communication, or environment dynamics are uncertain.

## Research Scope

- Vehicle-UAV collaborative routing and task allocation.
- Simulation environment design for coupled ground-air operations.
- Uncertainty modeling for travel time, demand, availability, and disturbances.
- Reinforcement learning algorithms for robust and adaptive decision-making.
- Benchmarking against heuristic, optimization, and learning-based baselines.

## Initial Architecture

```text
configs/                 Experiment and environment configuration files
docs/                    Project notes, design decisions, and research plans
experiments/             Runnable experiment entry points and result scripts
src/tdrp/                Core Python package
tests/                   Unit and smoke tests
```

Core package layout:

```text
src/tdrp/envs/           Simulation environments and state transitions
src/tdrp/uncertainty/    Stochastic scenario generators and uncertainty models
src/tdrp/algorithms/     RL policies, trainers, and baselines
src/tdrp/evaluation/     Metrics, rollout evaluation, and comparisons
src/tdrp/utils/          Shared utilities
```

## Near-Term Goals

1. Define the first formal problem setting.
2. Implement a small vehicle-UAV simulation environment.
3. Add uncertainty models and reproducible scenario generation.
4. Build baseline dispatch/routing heuristics.
5. Add a first RL training loop and evaluation protocol.

## Development Status

This repository is at the project-initialization stage. The first priority is to stabilize the problem definition and simulator API before implementing complex training algorithms.
