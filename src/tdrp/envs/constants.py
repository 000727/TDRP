from __future__ import annotations

TASK_POINT, TASK_LINE = 0, 1

DS_ON_TRUCK, DS_ON_GROUND, DS_TAKING_OFF, DS_CRUISING, DS_LANDING, DS_EXECUTING, DS_WAITING = range(7)

(EV_TRUCK_ARRIVE_STOP, EV_DRONE_TAKEOFF_DONE, EV_DRONE_ARRIVE_STOP,
 EV_DRONE_LAND_DONE, EV_DRONE_ARRIVE_TASK, EV_DRONE_TASK_DONE,
 EV_SWAP_DONE, EV_NEW_TASK_SPAWN) = range(8)

SCALE_CONFIGS = {
    "XS": dict(map_range=5.0,  nstops=5,  npoint=6,  nline=4,  max_time_h=2.0,  max_decisions=100),
    "S":  dict(map_range=10.0, nstops=8,  npoint=20, nline=10, max_time_h=5.0,  max_decisions=200),
    "M":  dict(map_range=15.0, nstops=12, npoint=35, nline=15, max_time_h=8.0,  max_decisions=400),
    "L":  dict(map_range=20.0, nstops=16, npoint=50, nline=20, max_time_h=15.0, max_decisions=800),
    "XL": dict(map_range=25.0, nstops=20, npoint=65, nline=25, max_time_h=20.0, max_decisions=1200),
}

MAX_STOPS, MAX_TASKS, MAX_LINE_PTS = 20, 128, 8
MAX_TASK_ACTIONS = 2 * MAX_TASKS
N_WIND_KERNELS = 4

SCALE_TIME_REFS_H = {
    "XS": 0.62,
    "S":  2.70,
    "M":  6.20,
    "L":  9.53,
    "XL": 15.0,
}
