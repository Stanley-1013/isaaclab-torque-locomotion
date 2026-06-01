# src/torque_loco/bio_actuator.py
"""BioActuator: applies the SATA biomechanical pipeline over an effort passthrough.

Subclasses IdealPDActuator (zero PD gains -> feed-forward effort passthrough) and replaces the
torque with the activation(tanh-EMA)+Hill+fatigue output of bio_constraints.apply_bio. Holds
per-env/joint activation + fatigue buffers, reset on env reset. Reads the growth scalar from the
owning env (env._G) so the torque ceiling grows during training; rear legs stay at the max.
The envelope is enforced by tanh (not a hard clip) -> _clip_effort is NOT called.
"""
import re
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .bio_constraints import apply_bio, BioState, BioCfg
from .growth import torque_limit_scale


class BioActuator(IdealPDActuator):
    cfg: "BioActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        e, j = self._num_envs, self.num_joints
        self._biocfg = BioCfg(
            kappa_scale=cfg.kappa_scale, gamma=cfg.gamma, beta=cfg.beta,
            activation_process=cfg.activation_process, hill_model=cfg.hill_model,
            motor_fatigue=cfg.motor_fatigue,
        )
        self.activation = torch.zeros(e, j, device=self._device)
        self.motor_fatigue = torch.zeros(e, j, device=self._device)
        self._vel_limit = torch.full((e, j), cfg.vel_limit_hip_thigh, device=self._device)
        for k, name in enumerate(self.joint_names):
            if "calf" in name:
                self._vel_limit[:, k] = cfg.vel_limit_calf
        self._front_mask = torch.tensor(
            [bool(re.match(r"^F[LR]_", n)) for n in self.joint_names],
            device=self._device,
        )
        self._dt = None
        self._env = None

    def set_runtime(self, dt, env):
        self._dt = dt
        self._env = env

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self.activation[env_ids] = 0.0
        g = float(getattr(self._env, "_G", 1.0)) if self._env is not None else 1.0
        hi = 0.2 * g
        if self._biocfg.motor_fatigue and hi > 0.0:
            self.motor_fatigue[env_ids] = torch.rand_like(self.motor_fatigue[env_ids]) * hi
        else:
            self.motor_fatigue[env_ids] = 0.0

    def _current_torque_limit(self):
        g = float(getattr(self._env, "_G", 1.0)) if self._env is not None else 1.0
        front = torque_limit_scale(g, self.cfg.tau_start, self.cfg.tau_end)
        tl = torch.full_like(self.activation, self.cfg.tau_end)
        tl[:, self._front_mask] = front
        return tl

    def compute(self, control_action: ArticulationActions, joint_pos, joint_vel):
        action = control_action.joint_efforts
        tau_limit = self._current_torque_limit()
        dt = self._dt if self._dt is not None else 0.005
        state = BioState(activation=self.activation, fatigue=self.motor_fatigue)
        torque, state = apply_bio(action, joint_vel, tau_limit, self._vel_limit, dt, state, self._biocfg)
        self.activation, self.motor_fatigue = state.activation, state.fatigue
        self.computed_effort = torque
        self.applied_effort = torque
        control_action.joint_efforts = torque
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class BioActuatorCfg(IdealPDActuatorCfg):
    class_type: type = BioActuator
    kappa_scale: float = 5.0
    gamma: float = 0.6
    beta: float = 0.9
    activation_process: bool = True
    hill_model: bool = True
    motor_fatigue: bool = True
    tau_start: float = 7.05
    tau_end: float = 23.5
    vel_limit_hip_thigh: float = 30.1
    vel_limit_calf: float = 20.07
