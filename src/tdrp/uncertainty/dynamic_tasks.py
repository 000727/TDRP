from __future__ import annotations

from typing import Any, Optional

from .base import UncertaintyModule


class DynamicTaskSpawnModule(UncertaintyModule):
    """Generate future task geometry when a scheduled spawn event fires."""

    name = "dynamic_task_spawn"

    def on_new_task_spawn(self, env: Any, task_id: int) -> Optional[bool]:
        if not bool(env.cfg.dynamic_generate_online):
            return None
        result = env._scenario_gen.generate_task_slot(
            env.scenario, int(task_id), float(env.t), env.np_random
        )
        if result is None:
            return False
        env._apply_generated_task_slot(int(task_id), result)
        return True
