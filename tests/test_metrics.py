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

from torque_loco.metrics import sata_peak_torque, sata_energy_per_step, sata_mean_jerk

def test_sata_peak_torque_is_max_abs_over_all():
    tau = torch.tensor([[1.0, -5.0], [3.0, 2.0]])     # (T, J)
    assert sata_peak_torque(tau) == 5.0

def test_sata_energy_per_step_matches_formula():
    tau = torch.ones(5, 2); vel = torch.ones(5, 2)    # |tau*vel| sum_j = 2 per step
    e = sata_energy_per_step(tau, vel, dt=0.02)        # mean_t( sum_j|tau*vel| ) * dt = 2*0.02
    assert abs(e - 0.04) < 1e-6

def test_sata_mean_jerk_first_difference_over_dt():
    a = torch.tensor([[0.0], [1.0], [1.0]])           # diffs: |1-0|=1, |1-1|=0 ; mean=0.5 ; /dt
    assert abs(sata_mean_jerk(a, dt=0.5) - (0.5 / 0.5)) < 1e-6

def test_sata_mean_jerk_zero_for_constant():
    assert sata_mean_jerk(torch.ones(10, 4), dt=0.005) == 0.0
