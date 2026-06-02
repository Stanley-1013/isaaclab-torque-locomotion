# src/torque_loco/growth.py
"""SATA Gompertz growth curriculum (paper Eqs 5-7), pure-torch/float, sim-free.

G(t) = exp(-exp(-k*(t - x0))) drives torque-ceiling and control-frequency growth.
Driven by cumulative env-step count t. Defaults from paper Table III / code.
"""
import math

K = 3e-5
X0 = 24000.0


def gompertz(step, k=K, x0=X0):
    """Gompertz developmental scalar G(t) in (0, 1)."""
    return math.exp(-math.exp(-k * (float(step) - x0)))


def torque_limit_scale(g, tau_start, tau_end):
    """Interpolate the torque ceiling: tau_start + (tau_end - tau_start) * G."""
    return tau_start + (tau_end - tau_start) * g


def control_freq(g, f_start=100.0, f_end=200.0):
    """Interpolate the control frequency 100 -> 200 Hz."""
    return f_start + (f_end - f_start) * g
