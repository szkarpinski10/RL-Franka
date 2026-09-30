from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

class task_success(ManagerTermBase):
    def __init__ (self,cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg,env)
        self._was_lifted = torch.zeros(
            env.num_envs,
            dtype = torch.bool,
            device = env.device,
        )

    def reset (self,env_ids):
        self._was_lifted[env_ids] = False

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            command_name: str,
            height_lift: float,
            at_destination_threshold: float,
            object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
            robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

    ) -> torch.Tensor:
        robot: RigidObject = env.scene [robot_cfg.name]
        obj: RigidObject = env.scene [object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w,_ = combine_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            command [:,:3],
        )

        object_pos_w = obj.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim = 1)

        is_lifted = object_pos_w[:,2] > height_lift
        self._was_lifted |= is_lifted

        return (self._was_lifted & (distance<at_destination_threshold))