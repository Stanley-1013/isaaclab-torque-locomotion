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
