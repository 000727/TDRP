# Project Charter

## Problem Statement

This project studies vehicle-UAV collaborative path planning under uncertainty. A ground vehicle and one or more UAVs must coordinate routing, launch/recovery, service assignment, and timing decisions while facing uncertain travel times, demands, task availability, weather, battery consumption, or communication conditions.

## Research Questions

1. How should a simulator represent coupled vehicle-UAV decisions while remaining efficient for reinforcement learning?
2. Which uncertainty sources matter most for route feasibility and system cost?
3. Can RL policies learn robust coordination strategies that outperform deterministic heuristics under distribution shift?
4. How should rollout reward balance travel cost, lateness, task completion, energy, and risk?

## MVP Problem Definition

Start with a small single-vehicle single-UAV setting:

- One depot.
- One ground vehicle.
- One UAV carried by the vehicle.
- A set of customer/task nodes.
- The UAV can be launched from the vehicle, serve one task, and rendezvous with the vehicle.
- Travel times are stochastic.
- Objective: minimize expected completion time plus penalties for infeasible or late service.

## Simulator Principles

- Deterministic seed control for reproducible experiments.
- Explicit state, action, transition, reward, and termination definitions.
- Separate deterministic geometry from stochastic uncertainty models.
- Keep a small smoke-test environment before scaling.

## Algorithm Principles

- Establish heuristic baselines before complex RL.
- Use simple RL baselines first, then add hierarchy or robustness.
- Evaluate policies under both training distribution and shifted uncertainty distributions.
- Log enough rollout traces to diagnose coordination failures.
