# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import combine_frame_transforms
import warp as wp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

# function from IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/lift/mdp/rewards
class object_goal_distance(ManagerTermBase):
    """Reward the agent for tracking the object-to-goal pose using a tanh kernel.

    If ``success_threshold`` is provided in the term params, this also tracks per-episode
    success (sticky binary: object ever within ``success_threshold`` of the commanded goal
    while lifted above ``minimal_height``) and logs the mean across environments under
    ``Metrics/success_rate`` on reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        object_pos_w = obj.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
        is_lifted = object_pos_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1 - torch.tanh(distance / std))

# Reach reward
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w.torch
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


# Grasp reward
def grasp_reward(env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.03,
) -> torch.Tensor:

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    cube_pos_w = object.data.root_pos_w.torch
    grasp_point = ee_frame.data.target_pos_w.torch [:,0,:] 
    object_ee_distance = torch.linalg.norm(cube_pos_w - grasp_point, dim=1)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) >= 1, "Observations require at least one gripper joint"

    # Grasped: the end-effector is close to the object and every gripper joint has moved
    # away from the open position (i.e. the jaws have closed on the object).
    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
    grasped = object_ee_distance < diff_threshold
    for joint_id in gripper_joint_ids:
        grasped = torch.logical_and(
            grasped,
            torch.abs(robot.data.joint_pos.torch[:, joint_id] - open_val) > env.cfg.gripper_threshold,
        )


    return grasped.float() 


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)



class cube_at_destination(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs,dtype = torch.bool, device = env.device)

    def reset(self,env_ids:torch.Tensor):
        self._was_lifted [ env_ids] = False

    def __call__(
        self,
        env:ManagerBasedRLEnv,
        command_name: str,
        height_lift: float,
        at_destination_treshold: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        
    ) -> torch.Tensor:

        robot: RigidObject = env.scene [robot_cfg.name]
        obj: RigidObject = env.scene [ object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command [:,:3]
        )

        object_pos_w = obj.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim = 1)

        is_lifted = object_pos_w[:,2] > height_lift
        self._was_lifted |= is_lifted

        check_position = distance <at_destination_treshold

        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        cube_released = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for joint_id in gripper_joint_ids:
            cube_released &= torch.abs(robot.data.joint_pos.torch[:,joint_id]-open_val) <env.cfg.gripper_threshold

        success = self._was_lifted & check_position & cube_released

        return success.float()


def lifting_progress(
    env:ManagerBasedRLEnv,object_cfg:SceneEntityCfg = SceneEntityCfg("cube_1"),
    starting_height:float = 0.58,
    target_height:float = 0.62)-> torch.Tensor:

    obj: RigidObject = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w.torch[:,2]

    progress = (cube_z-starting_height)/(target_height - starting_height)

    return torch.clamp(progress, min = 0.0, max = 1.0)


def cube_near_robot(
    env:ManagerBasedRLEnv,
    area: float,
    object_cfg:SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg:SceneEntityCfg = SceneEntityCfg("robot"),
) ->torch.Tensor:

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    cube_x_y = object.data.root_pos_w.torch[:,[0,1]]
    robot_x_y = robot.data.root_pos_w.torch[:,[0,1]]

    diff = cube_x_y - robot_x_y
    distance_cube_to_robot = torch.linalg.norm(diff,dim=1)

    return torch.clamp((area-distance_cube_to_robot)/area,min = 0.0,max = 1.0)