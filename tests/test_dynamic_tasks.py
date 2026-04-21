import numpy as np

from tdrp.envs import EnvConfig, TruckMultiDroneCleanEnv


def test_dynamic_tasks_can_be_pregenerated_offline():
    cfg = EnvConfig(
        fixed_scale="XS",
        allow_dynamic_tasks=True,
        dynamic_task_ratio=0.5,
        dynamic_generate_online=False,
    )
    env = TruckMultiDroneCleanEnv(cfg)
    env.reset(seed=0)

    sc = env.scenario
    future = np.flatnonzero(sc.task_spawn_time[: sc.ntasks] > cfg.eps_time)

    assert future.size > 0
    assert np.abs(sc.point_xy[future]).sum() > 0.0
    assert sc.task_priority[future].sum() > 0
    assert sc.task_deadline_h[future].sum() > 0.0
