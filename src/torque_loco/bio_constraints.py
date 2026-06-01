# src/torque_loco/bio_constraints.py
from dataclasses import dataclass
import torch

@dataclass
class BioCfg:
    alpha: float          # low-pass coefficient (0,1]; higher = faster
    effort_limit: float   # N·m, per joint
    fatigue_rate: float   # capacity loss per step at full overload
    recovery_rate: float  # capacity regained per step when unloaded
    fatigue_onset: float  # |torque|/limit above which fatigue accrues

@dataclass
class BioState:
    activation: torch.Tensor  # (num_envs, num_joints) filtered torque
    capacity: torch.Tensor    # (num_envs, num_joints) in (0,1]

def apply_bio_constraints(cmd_torque, state, cfg):
    act = state.activation + cfg.alpha * (cmd_torque - state.activation)
    load = act.abs() / cfg.effort_limit
    overload = torch.clamp(load - cfg.fatigue_onset, min=0.0)
    cap = state.capacity - cfg.fatigue_rate * overload
    cap = cap + cfg.recovery_rate * (1.0 - load).clamp(min=0.0)
    cap = cap.clamp(0.05, 1.0)
    limit = cap * cfg.effort_limit
    out = act.clamp(-limit, limit)
    return out, BioState(activation=act, capacity=cap)
