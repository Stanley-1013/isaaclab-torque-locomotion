# src/torque_loco/metrics.py
import torch

def peak_torque(tau):           # tau: (T, J)
    return tau.abs().max().item()

def action_jerk(actions):       # actions: (T, J); jerk ~ 2nd difference
    if actions.shape[0] < 3:
        return 0.0
    d2 = actions[2:] - 2 * actions[1:-1] + actions[:-2]
    return d2.abs().mean().item()

def mech_energy(tau, vel, dt):  # sum |tau*vel| dt over the rollout
    return (tau * vel).abs().sum().item() * dt

def sata_peak_torque(tau):                  # tau: (T, J) -> scalar
    """Peak |torque| over all steps & joints (matches eval_under_conditions)."""
    return tau.abs().max().item()

def sata_energy_per_step(tau, vel, dt):     # (T, J),(T, J) -> scalar
    """Mean per-step mechanical energy: mean_t( sum_j |tau*vel| ) * dt."""
    power = (tau * vel).abs().sum(dim=1)    # (T,)
    return (power.mean() * dt).item()

def sata_mean_jerk(actions, dt):            # (T, J) -> scalar
    """Mean action jerk: mean_t( sum_j |a_t - a_{t-1}| ) / dt (first difference)."""
    if actions.shape[0] < 2:
        return 0.0
    d1 = (actions[1:] - actions[:-1]).abs().sum(dim=1)   # (T-1,)
    return (d1.mean() / dt).item()
