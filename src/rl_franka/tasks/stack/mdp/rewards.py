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


### pomocnicze funkcje

#sprawdzenie czy kostka znajduje sie na target table
def cube_at_target_pos(
        env:ManagerBasedRLEnv,
        table_widht: float = 0.025,
        table_lenght: float = 0.025,
        z_threshold: float = 0.025,
        target_offset_z: float = 0.05,
        target_cfg : SceneEntityCfg = SceneEntityCfg("target_table"),
        object_cfg : SceneEntityCfg = SceneEntityCfg("cube_1"),


) -> torch.Tensor: 
    target: RigidObject = env.scene[target_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    target_table_pos = target.data.root_pos_w.torch
    cube_pos = object.data.root_pos_w.torch

    diff = cube_pos - target_table_pos

    check_x = torch.abs(diff[:,0])<table_lenght - 0.01
    check_y = torch.abs(diff[:,1])<table_widht - 0.01
    check_z = torch.abs(diff[:,2] - target_offset_z) < z_threshold

    return check_x & check_y & check_z

# sprawdzenie czy kostka jest w chwytaku albo na target table, prewencja podrzucania kostki do gory 
def cube_control(
    env: ManagerBasedRLEnv,
    hold_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos = cube.data.root_pos_w.torch
    ee_pos = ee_frame.data.target_pos_w.torch[:, 0, :]

    cube_in_gripper = torch.linalg.norm(cube_pos - ee_pos, dim=1) < hold_threshold


    return cube_in_gripper | cube_at_target_pos(env)



#funkcje - rewardy 
# (1) reaching zmieniony względem nvidii - reaching nie działa w momencie kostki w miejscu docelowym 

# nvidia 
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

# poprawiona 
def reaching_unless_at_target(
        env:ManagerBasedRLEnv,
        std:float , 
) ->torch.Tensor:
    reach = object_ee_distance(env,std)
    return torch.where(cube_at_target_pos(env),1.0,reach)

# (2) lift reward zmiana dodanie warunku, że kostka znajduje się w chwytaku - inaczej robot podrzuca kostke do góry
def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)

def cube_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[object_cfg.name]
    is_lifted = cube.data.root_pos_w.torch[:, 2] > minimal_height
    return (is_lifted & cube_control(env)).float()


# (3) naprowadzanie do celu 

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


class cube_target_distance(ManagerTermBase):
    def __init__(self, cfg:RewardTermCfg, env:ManagerBasedRLEnv):
        super().__init__(cfg,env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/cube_on_table"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            std: float,
            minimal_height: float,
            target_offset_z: float = 0.05,
            target_cfg : SceneEntityCfg = SceneEntityCfg("target_table"),
            object_cfg : SceneEntityCfg = SceneEntityCfg("cube_1"),
            success_threshold: float | None = None,
        ) -> torch.Tensor:
        target: RigidObject = env.scene[target_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]

        table_pos = target.data.root_pos_w.torch
        target_pos = table_pos + torch.tensor([0.0,0.0,target_offset_z],device=env.device)

        cube_pos = object.data.root_pos_w.torch
        distance = torch.linalg.norm(target_pos - cube_pos, dim =1)

        is_lifted = (cube_pos[:, 2] > minimal_height) & cube_control(env)

        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)

        return torch.where(is_lifted,1-torch.tanh(distance/std),0.0)

# (4) kostka na stole docelowym 
def cube_released(
    env: ManagerBasedRLEnv, 
    speed_std:float = 0.05,
    open_threshold: float = 0.035,
    object_cfg :SceneEntityCfg = SceneEntityCfg("cube_1"),
    ) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger.*")
    gripper_open = (robot.data.joint_pos.torch[:, finger_ids] > open_threshold).all(dim=1)
    released = cube_at_target_pos(env)& gripper_open

    cube_speed= torch.linalg.norm(object.data.root_lin_vel_w.torch,dim=1)
    cube_is_still = 1- torch.tanh(cube_speed/speed_std)


    return released.float() * cube_is_still

# (5) gdy wszystkie kostki ze sceny są na stole docelowym - powrót do ułożena bazowego 
def arm_home_return(
        env:ManagerBasedRLEnv,
        std: float,
        open_threshold: float = 0.035,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
)->torch.Tensor:
    robot:Articulation = env.scene[robot_cfg.name]

    finger_ids, _ = robot.find_joints("panda_finger.*")
    gripper_open = (robot.data.joint_pos.torch[:, finger_ids] > open_threshold).all(dim=1)
    released = cube_at_target_pos(env) & gripper_open

    arm_ids, _ = robot.find_joints("panda_joint.*")
    joint_diff = robot.data.joint_pos.torch[:, arm_ids] - robot.data.default_joint_pos.torch[:, arm_ids]
    distance = torch.linalg.norm(joint_diff, dim=1)

    return torch.where(released, 1 - torch.tanh(distance / std), 0.0)


def arm_still_at_home(
    env: ManagerBasedRLEnv,
    home_threshold: float = 0.4,
    speed_std: float = 0.5,
    open_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    finger_ids, _ = robot.find_joints("panda_finger.*")
    gripper_open = (robot.data.joint_pos.torch[:, finger_ids] > open_threshold).all(dim=1)
    released = cube_at_target_pos(env) & gripper_open

    arm_ids, _ = robot.find_joints("panda_joint.*")
    joint_diff = robot.data.joint_pos.torch[:, arm_ids] - robot.data.default_joint_pos.torch[:, arm_ids]
    at_home = released & (torch.linalg.norm(joint_diff, dim=1) < home_threshold)

    arm_speed = torch.linalg.norm(robot.data.joint_vel.torch[:, arm_ids], dim=1)
    return torch.where(at_home, 1 - torch.tanh(arm_speed / speed_std), 0.0)



def open_when_on_target(
    env: ManagerBasedRLEnv,
    grasp_width: float = 0.025,   
    open_width: float = 0.04,     
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    finger_ids, _ = robot.find_joints("panda_finger.*")
    finger = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1)

    
    opening = torch.clamp((finger - grasp_width) / (open_width - grasp_width), 0.0, 1.0)

    return torch.where(cube_at_target_pos(env), opening, 0.0)