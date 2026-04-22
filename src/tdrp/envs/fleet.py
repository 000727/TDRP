from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DS_ON_TRUCK


@dataclass(frozen=True)
class FleetSpec:
    """Static fleet layout for multi-truck, multi-drone environments."""

    ntrucks: int = 1
    drones_per_truck: tuple[int, ...] = (1,)

    @classmethod
    def from_config(cls, cfg: object) -> "FleetSpec":
        ntrucks = int(getattr(cfg, "ntrucks", 1))
        layout = getattr(cfg, "drones_per_truck", None)
        if layout is None:
            ndrones = int(getattr(cfg, "ndrones", 1))
            base = ndrones // ntrucks
            rem = ndrones % ntrucks
            layout = tuple(base + (1 if i < rem else 0) for i in range(ntrucks))
        return cls(ntrucks=ntrucks, drones_per_truck=tuple(int(n) for n in layout))

    def __post_init__(self) -> None:
        if self.ntrucks <= 0:
            raise ValueError("ntrucks must be positive")
        if len(self.drones_per_truck) != self.ntrucks:
            raise ValueError("drones_per_truck must have one entry per truck")
        if any(int(n) < 0 for n in self.drones_per_truck):
            raise ValueError("drones_per_truck cannot contain negative counts")

    @property
    def ndrones(self) -> int:
        return int(sum(int(n) for n in self.drones_per_truck))

    def carrier_for_drone(self, drone_id: int) -> int:
        if not (0 <= int(drone_id) < self.ndrones):
            raise IndexError("drone_id out of range")
        offset = 0
        for truck_id, count in enumerate(self.drones_per_truck):
            next_offset = offset + int(count)
            if int(drone_id) < next_offset:
                return int(truck_id)
            offset = next_offset
        raise IndexError("drone_id out of range")


@dataclass
class TruckRuntimeState:
    truck_id: int
    stop_id: int
    target_stop_id: int
    busy: bool = False


@dataclass
class DroneRuntimeState:
    drone_id: int
    carrier_truck_id: int
    stop_id: int
    status: int = DS_ON_TRUCK
    task_id: int = -1
    battery: float = 100.0
    xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


@dataclass
class FleetRuntimeState:
    trucks: list[TruckRuntimeState]
    drones: list[DroneRuntimeState]

    @classmethod
    def from_spec(
        cls,
        spec: FleetSpec,
        depot_stop_id: int,
        max_battery: float,
    ) -> "FleetRuntimeState":
        trucks = [
            TruckRuntimeState(
                truck_id=i,
                stop_id=int(depot_stop_id),
                target_stop_id=int(depot_stop_id),
            )
            for i in range(spec.ntrucks)
        ]
        drones = [
            DroneRuntimeState(
                drone_id=d,
                carrier_truck_id=spec.carrier_for_drone(d),
                stop_id=int(depot_stop_id),
                battery=float(max_battery),
            )
            for d in range(spec.ndrones)
        ]
        return cls(trucks=trucks, drones=drones)
