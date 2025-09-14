import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/WF_TRON1A/WF_TRON1A.usd")

WHEELFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8+0.166),
        joint_pos={
            ".*_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"],
            effort_limit=80.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.8,
            friction=0.0
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[RL]_Joint"],
            effort_limit=40.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.33
        ),
    },
)

usd_path_arm = os.path.join(current_dir, "../usd/WF_TRON1A_ARM/WF_TRON1A_ARM.usd")

WHEELFOOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path_arm,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
            # arm
            "J1": 0.0,  # [rad]
            "J2": 0.0,  # [rad]
            "J3": 0.0,  # [rad]
            "J4": 0.0,  # [rad]
            "J5": 0.0,  # [rad]
            "J6": 0.0,  # [rad]
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"],
            effort_limit=80.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.8,
            friction=0.0
        ),
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[RL]_Joint"],
            effort_limit=40.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.33

        ),
        "arm_former_three": ImplicitActuatorCfg(
            joint_names_expr=["J1", "J2", "J3"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=18.0,
            damping=1.0,
            friction=0.0,
        ),
        "arm_later_three": ImplicitActuatorCfg(
            joint_names_expr=["J4", "J5", "J6"],
            effort_limit=3.0,
            velocity_limit=3.9,
            stiffness=4.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
