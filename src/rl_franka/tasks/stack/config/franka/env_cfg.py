# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp

##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# @configclass
# class StackPhysicsCfg(PresetCfg):
#     """Physics presets for the generated cart-pole environment."""

#     isaacsim_physx: PhysxCfg = PhysxCfg()
#     ovphysx: OvPhysxCfg = OvPhysxCfg()
#     physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
#     newton_mjwarp: NewtonCfg = NewtonCfg(
#         solver_cfg=MJWarpSolverCfg(
#             njmax=5,
#             nconmax=3,
#             cone="pyramidal",
#             impratio=1,
#             integrator="implicitfast",
#         ),
#         num_substeps=1,
#         debug_mode=False,
#         use_cuda_graph=True,
#     )
#     newton_kamino: NewtonCfg = NewtonCfg(
#         solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
#         debug_mode=False,
#         use_cuda_graph=True,
#     )
#     default: NewtonCfg = newton_mjwarp


##
# Scene definition
##


_FRANKA_STACK_IK_REL_INIT_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.0400,
}


@configclass
class StackSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path = "{ENV_REGEX_NS}/Robot",
        init_state = ArticulationCfg.InitialStateCfg(
            joint_pos = _FRANKA_STACK_IK_REL_INIT_JOINT_POS,
            pos = [0.0,0.0,0.55],
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


    # table
    table = AssetBaseCfg(
        prim_path = "{ENV_REGEX_NS}/Table",
        init_state = AssetBaseCfg.InitialStateCfg(
            pos = [0.5,0,0.3], rot = [1.0,0,0.0,0]
        ),
        spawn = sim_utils.CuboidCfg(
            size = (1.6, 1.2, 0.5),
            collision_props = sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
    )
    
    # cube 
    cube_1 = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/Cube_1",
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = [0.45, 0.0, 0.58],
        ),
        spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.45,
                ),
        ),

    )

    # end effector marker
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame = FrameTransformerCfg(
        prim_path = "{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis = False,
        visualizer_cfg = marker_cfg,
        target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path = "{ENV_REGEX_NS}/Robot/panda_hand",
                name = "end_effector",
                offset = OffsetCfg(pos = [0.0,0.0,0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ]
    )




##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )

    gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # joint 
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
        # actions 
        actions = ObsTerm(func=mdp.last_action)

        # ee 
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # cube 
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # joints offset 
    randomize_franka_joint_state = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # cube random pos
    randomize_cube1_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0, 0),
                "roll": (0, 0), "pitch": (0, 0), "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reach = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.3, "object_cfg": SceneEntityCfg("cube_1"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0,
    )

    lift = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.63, "object_cfg": SceneEntityCfg("cube_1")},
        weight=15.0,
    )
    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cube dropped 
    cube_1_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.52, "asset_cfg": SceneEntityCfg("cube_1")},
    )

    # (3) Cube lifted - success
    # success = 



@configclass
class CurriculumCfg:
    pass

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )
##
# Environment configuration
##


@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the generated cart-pole environment."""

    # Scene settings
    scene: StackSceneCfg = StackSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # general settings
        self.decimation = 2
        self.episode_length_s = 15
        
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
        
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
