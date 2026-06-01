# tests/test_metrics.py
import torch
from torque_loco.metrics import peak_torque, action_jerk, mech_energy

def test_peak_torque_is_max_abs():
    tau = torch.tensor([[1.0, -5.0], [3.0, 2.0]])  # (T, J)
    assert peak_torque(tau) == 5.0

def test_jerk_zero_for_constant_action():
    a = torch.ones(10, 4)
    assert action_jerk(a) == 0.0

def test_energy_nonnegative_and_scales():
    tau = torch.ones(5, 2); vel = torch.ones(5, 2)
    e = mech_energy(tau, vel, dt=0.02)
    assert e > 0
    assert torch.isclose(torch.tensor(mech_energy(2*tau, vel, 0.02)),
                         torch.tensor(2*e), atol=1e-5)
