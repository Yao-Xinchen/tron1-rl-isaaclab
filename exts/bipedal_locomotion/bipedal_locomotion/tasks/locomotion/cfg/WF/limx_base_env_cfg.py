import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
import isaaclab.sim as sim_utils
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.WF.terrains_cfg import ROUGH_TERRAINS_CFG
from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_ARM_CFG

##################
# Scene Definition
##################


@configclass
class WFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene"""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # bipedal robot
    robot: ArticulationCfg = WHEELFOOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # height sensors
    height_scanner: RayCasterCfg = MISSING

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, update_period=0.0
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    base_twist = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2),  # min max [m/s]
            lin_vel_y=(-0.0, 0.0),  # min max [m/s]
            ang_vel_z=(-0.3, 0.3),  # min max [rad/s]
            # lin_vel_x=(-2.0, 2.0),  # min max [m/s]
            # lin_vel_y=(-1.0, 1.0),  # min max [m/s]
            # ang_vel_z=(-0.5, 0.5),  # min max [rad/s]
        ),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"],
                                           scale=0.5, use_default_offset=True)
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["wheel_[RL]_Joint"], 
                                           scale=10.0, use_default_offset=True)


@configclass
class ObservarionsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements exclude wheel pos
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"]
            )},
        )  # 6

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint","wheel_[RL]_Joint"]
            )},
            scale=0.1,
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
      
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class HistoryObsCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements exclude wheel pos
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"]
            )},
        )  # 6

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint","wheel_[RL]_Joint"]
            )},
            scale=0.1,
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 15
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # robot base measurements
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0),scale=1.0,)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,clip=(-100.0, 100.0),scale=1.0,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity,clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"]
            )},
        )  # 6
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=0.1,)

        # last action
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0,)

        # velocity command
        # vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # heights scan
        heights = ObsTerm(func=mdp.height_scan,params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        
        # Privileged observation
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_twist"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    prepare_quantity_for_tron1_piper = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # "mass_distribution_params": (-5.0, 5.0),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )

    # actuator gains randomization
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # center of mass randomization
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )

    # # set arm links to zero mass and inertia
    # zero_arm_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="link[1-7]"),
    #         "mass_distribution_params": (0.000001, 0.000001),
    #         "operation": "abs",
    #         "recompute_inertia": True,
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # -- safety
    safety_exp = RewTerm(
        func=mdp.safety_reward_exp, weight=1.0, params={"base_height_target": 0.9, "std": math.sqrt(0.5)}
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_[RL]_Link","hip_[RL]_Link"]), "threshold": 1.0},
    )

    # -- task
    track_base_linear_velocity_exp = RewTerm(
        func=mdp.track_base_linear_velocity_exp,
        weight=2.0,
        params={
            "command_name": "base_twist",
            "std": math.sqrt(0.5),
        },
    )
    track_base_angular_velocity_exp = RewTerm(
        func=mdp.track_base_angular_velocity_exp,
        weight=1.5,
        params={
            "command_name": "base_twist",
            "std": math.sqrt(0.5),
        },
    )
    # track_base_penalty_cb = RewTerm(func=mdp.tracking_base_cb_penalty_l1, weight=-1.0)
    # track_base_pb = RewTerm(func=mdp.track_base_pb, weight=15.0)

    track_base_velocity_exp = RewTerm(
        func=mdp.track_base_velocity_exp,
        weight=1.0,
        params={"command_name": "base_twist", "std": math.sqrt(0.5)},
    )

    # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_weighted_torques_l2 = RewTerm(
        func=mdp.weighted_joint_torques_l2,
        weight=-4.0e-5,
        params={
            "torque_weight": {
                "abad_L_Joint": 0.2,
                "hip_L_Joint": 0.2,
                "knee_L_Joint": 0.2,
                # "foot_L_Joint": 0.2,
                "abad_R_Joint": 0.2,
                "hip_R_Joint": 0.2,
                "knee_R_Joint": 0.2,
                # "foot_R_Joint": 0.2,
                "wheel_L_Joint": 3.0,
                "wheel_R_Joint": 3.0,
            }
        },
    )

    dof_weighted_power_l1 = RewTerm(
        func=mdp.weighted_joint_power_l1,
        weight=-2.5e-4,
        params={
            "power_weight": {
                "abad_L_Joint": 1.0,
                "hip_L_Joint": 1.0,
                "knee_L_Joint": 1.0,
                # "foot_L_Joint": 1.0,
                "abad_R_Joint": 1.0,
                "hip_R_Joint": 1.0,
                "knee_R_Joint": 1.0,
                # "foot_R_Joint": 1.0,
                "wheel_L_Joint": 1.0,
                "wheel_R_Joint": 1.0,
            }
        },
    )

    # dof_acc_l2 = RewTerm(
    #     func=mdp.joint_acc_l2, weight=-2.0e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    # )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.0004)
    # -- optional penalties
    dof_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")}
    )
    dof_vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    # dof_power_l1 = RewTerm(func=mdp.dof_power_l1, weight=0.0)
    dof_non_wheel_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    # joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=0.0)
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=0.0, params={"target_height": 0.3})

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000)
    # alive = RewTerm(func=mdp.stay_alive, weight=2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )
    
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_stochastic,
        params={
            "limit_angle": math.pi * 0.4,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )

    bad_height = DoneTerm(
        func=mdp.bad_height_stochastic,
        params={
            "limit_height": 0.4,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    velocity_commands_ranges_level = CurrTerm(
        func=mdp.velocity_commands_ranges_level,  # type: ignore
        params={
            "max_range": {"lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.5, 1.5), "ang_vel_z": (-2.0, 2.0)},
            "update_interval": 80 * 24,  # 80 iterations * 24 steps per iteration
            "command_name": "base_twist",
        },
    )

########################
# Environment definition
########################


@configclass
class WFEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: WFSceneCfg = WFSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.005
        self.seed = 42
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class WFEnvCfg_PLAY(WFEnvCfg):
    """Configuration for the play environment"""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        if hasattr(self.events, 'push_robot'):
            del self.events.push_robot
