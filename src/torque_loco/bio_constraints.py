# src/torque_loco/bio_constraints.py
"""Pure-torch SATA biomechanical pipeline (sim-free, unit-tested).

Faithful to SATA (arXiv:2502.12674, Eqs 1-4) and the legged_gym reference:
  activation:  alpha_current = tanh(action * kappa_scale / tau_limit)
               alpha_t       = alpha_current*gamma + alpha_{t-1}*(1-gamma)   (gamma = new-weight)
  Hill:        tau = tau_limit * alpha_t * (1 - sign(alpha_t) * q_dot / q_dot_limit)
  fatigue:     zeta_t = (zeta_{t-1} + |tau|*dt) * beta
Flags mirror SATA's control cfg; the migration runs all-on (reference).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class BioCfg:
    kappa_scale: float = 5.0      # action -> torque-space scale (SATA action_scale)
    gamma: float = 0.6            # EMA NEW-weight (alpha_current*gamma + prev*(1-gamma))
    beta: float = 0.9             # fatigue recovery factor (multiplicative decay/step)
    activation_process: bool = True
    hill_model: bool = True
    motor_fatigue: bool = True


@dataclass(frozen=True)
class BioState:
    activation: torch.Tensor      # (E, J) EMA activation sign, in (-1, 1) when activation on
    fatigue: torch.Tensor         # (E, J) leaky-integrator fatigue >= 0


def apply_bio(
    action: torch.Tensor,
    joint_vel: torch.Tensor,
    torque_limit: torch.Tensor,
    vel_limit: torch.Tensor,
    dt: float,
    state: BioState,
    cfg: BioCfg,
) -> tuple[torch.Tensor, BioState]:
    """Map a raw policy action to applied joint torque + updated bio state.

    Args (all (E, J) tensors unless noted):
        action: raw policy output a_s (pre-scale).
        joint_vel: current joint velocity q_dot.
        torque_limit: current per-joint tau_limit (already grown by the curriculum).
        vel_limit: per-joint q_dot_limit (Hill denominator).
        dt: physics timestep (float).
        state: BioState (activation, fatigue).
        cfg: BioCfg.
    Returns:
        (torque, new_state). torque is (E, J) applied joint torque (NOT hard-clipped).
        With activation on, tanh bounds |alpha| < 1; but the Hill term can AMPLIFY torque
        up to ~2x tau_limit for opposing joint velocity (eccentric contraction). The final
        sim safety net is the actuator's effort_limit, not this function.
    """
    a_s = action * cfg.kappa_scale
    if cfg.activation_process:
        alpha_current = torch.tanh(a_s / torque_limit)
        alpha = alpha_current * cfg.gamma + state.activation * (1.0 - cfg.gamma)
    else:
        alpha = a_s / torque_limit
    if cfg.hill_model:
        torque = torque_limit * alpha * (1.0 - torch.sign(alpha) * joint_vel / vel_limit)
    else:
        torque = alpha * torque_limit
    if cfg.motor_fatigue:
        fatigue = (state.fatigue + torque.abs() * dt) * cfg.beta
    else:
        fatigue = torch.zeros_like(state.fatigue)
    return torque, BioState(activation=alpha, fatigue=fatigue)
