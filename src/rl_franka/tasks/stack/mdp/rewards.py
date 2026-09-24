# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
import warp as wp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

# # Function from IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/mdp/observations.py
# def object_grasped(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg,
#     ee_frame_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg,
#     diff_threshold: float = 0.06,
# ) -> torch.Tensor:
#     """Check if an object is grasped by the specified robot."""

#     robot: Articulation = env.scene[robot_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]

#     object_pos = object.data.root_pos_w.torch
#     end_effector_pos = ee_frame.data.target_pos_w.torch[:, 0, :]
#     pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

#     if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
#         surface_gripper = env.scene.surface_grippers["surface_gripper"]
#         suction_cup_status = wp.to_torch(surface_gripper.state).view(-1, 1)  # 1: closed, 0: closing, -1: open
#         suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
#         grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

#     else:
#         if hasattr(env.cfg, "gripper_joint_names"):
#             gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
#             assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

#             grasped = torch.logical_and(
#                 pose_diff < diff_threshold,
#                 torch.abs(
#                     robot.data.joint_pos.torch[:, gripper_joint_ids[0]]
#                     - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
#                 )
#                 > env.cfg.gripper_threshold,
#             )
#             grasped = torch.logical_and(
#                 grasped,
#                 torch.abs(
#                     robot.data.joint_pos.torch[:, gripper_joint_ids[1]]
#                     - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
#                 )
#                 > env.cfg.gripper_threshold,
#             )

#     return grasped

# Function from IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py
# Reward for lifting the cube 
def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


# Function from IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py
# Reward for reaching 
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



# def grasp_reward(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     reward_value_grasped : float,
# ) -> torch.Tensor:
#     return 0
