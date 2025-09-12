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
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.WF.terrains_cfg import BLIND_ROUGH_TERRAINS_CFG

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
        terrain_generator=BLIND_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # bipedal robot
    robot: ArticulationCfg = MISSING

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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=True,
        resampling_time_range=(3.0, 15.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=0.25,
        use_default_offset=True,
    )
    
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_L_Joint", "wheel_R_Joint"],
        scale=1.0, # 10
    )


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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_exclude_wheel,
                            params={"wheel_joints_name": ["wheel_[RL]_Joint"]}, 
                            noise=GaussianNoise(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05,)

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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_exclude_wheel,
                            params={"wheel_joints_name": ["wheel_[RL]_Joint"]}, 
                            noise=GaussianNoise(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05,)

        # last action
        last_action = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # robot base measurements
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0),scale=1.0,)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,clip=(-100.0, 100.0),scale=1.0,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity,clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0,)

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
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

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

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),
            "damping_distribution_params": (2.0, 3.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.075, 0.075), (-0.075, 0.075), (-0.075, 0.075)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "force_range": {
                "x": (-500.0, 500.0),
                "y": (-500.0, 500.0),
                "z": (-0.0, 0.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""
    # termination related rewards
    keep_balance = RewTerm(
        func=mdp.stay_alive,
        weight=1.0
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-5.0
    )

    # rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # rew_nominal_foot_position = RewTerm(
    #     func=mdp.nominal_foot_position,
    #     weight=1.0, # 4.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"), 
    #             "base_height_target": 0.80,
    #             "command_name": "base_velocity",
    #             "std": math.sqrt(0.5),},
    # )
    rew_leg_symmetry = RewTerm(
        func=mdp.leg_symmetry,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"), "std": math.sqrt(0.5)},
    )
    rew_same_foot_x_position = RewTerm(
        func=mdp.same_feet_x_position,
        weight=-50, # 0.1, # changed to penalty mode
        params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
    )

    # penalizations
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00016)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.3) # -0.03
    pen_non_wheel_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]), "threshold": 10.0},
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.03)
    pen_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100,
        params={"min_feet_distance": 0.32,
                "max_feet_distance": 0.35,
                "feet_links_name": ["wheel_[RL]_Link"]}
    )
    # pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_base_height = RewTerm(func=mdp.base_com_height, params={"target_height": 0.80}, weight=-30.0)
    
    
    pen_joint_power_l1 = RewTerm(func=mdp.joint_powers_l1, weight=-2e-5)
    pen_joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")}
    )
    pen_vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.03, # -5e-5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    
    # pen_wheel_slide = RewTerm(
    #         func=mdp.wheel_slide_penalty,
    #         weight=-0.5,
    #         params={
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
    #             "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
    #             "radius": 0.128,
    #             "axis_idx": 1,
    #         },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # lin_vel_curriculum = CurrTerm(
    #         func=mdp.lin_vel_curriculum,
    #         params={
    #             "command_name": "base_velocity",
    #             "rwd_threshold": 0.7,
    #             "time_step": 2e-4 / 24,
    #             "max_lin_vel_x": (-1.0, 1.0),
    #             "max_lin_vel_y": (-0.75, 0.75),
    #         },
    # )

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
