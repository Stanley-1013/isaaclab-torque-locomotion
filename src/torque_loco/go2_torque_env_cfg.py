# src/torque_loco/go2_torque_env_cfg.py
"""Go2 velocity task, converted from joint-position to joint-effort (torque) control.

Subclasses the stock ``UnitreeGo2FlatEnvCfg`` and, in ``__post_init__`` (after the
parent runs), (1) swaps the position action term for a joint-effort term and
(2) replaces the stock ``DCMotorCfg`` ``"base_legs"`` actuator with a zero-gain
``IdealPDActuatorCfg`` so the policy's commanded torque passes straight to the joint.

API grounded against the installed Isaac Lab 5.1 build (see docs/operations.md):
  - env cfg ......... isaaclab_tasks...velocity.config.go2.flat_env_cfg:UnitreeGo2FlatEnvCfg
  - effort action ... isaaclab.envs.mdp.JointEffortActionCfg (JointActionCfg: joint_names, scale)
  - actuator ........ isaaclab.actuators.IdealPDActuatorCfg (ActuatorBaseCfg fields)
  - stock actuator .. DCMotorCfg(effort_limit=23.5, velocity_limit=30.0, stiffness=25, damping=0.5)
"""

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

# Go2 joint torque limit: SATA used a 23.5 N·m sim clip (real Go2 peak ~45 N·m).
# This matches the stock DCMotor effort_limit, keeping the cross-engine story aligned.
GO2_EFFORT_LIMIT = 23.5
# Stock Go2 actuator velocity limit (carried over from DCMotorCfg).
GO2_VELOCITY_LIMIT = 30.0


@configclass
class Go2TorqueEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat Go2 velocity task driven by direct joint torques."""

    def __post_init__(self):
        super().__post_init__()

        # 1) Policy outputs joint torques directly (the paradigm migration).
        #    ActionManager reads cfg.__dict__ and skips None terms, so nulling
        #    joint_pos drops it and the dynamically-added joint_effort is picked up.
        self.actions.joint_pos = None
        self.actions.joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=GO2_EFFORT_LIMIT,  # action in [-1, 1] -> torque in N·m
        )

        # 2) Make the actuator pass effort straight through (zero PD gains) so the
        #    policy's torque is what reaches the joint. effort_limit is the hard clip.
        self.scene.robot.actuators["base_legs"] = IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=GO2_EFFORT_LIMIT,
            velocity_limit=GO2_VELOCITY_LIMIT,
        )


@configclass
class Go2TorqueEnvCfg_PLAY(Go2TorqueEnvCfg):
    """Smaller, deterministic scene for rendering/eval rollouts."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
