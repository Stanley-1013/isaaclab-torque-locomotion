# tests/test_bio_constraints.py
import torch
from torque_loco.bio_constraints import apply_bio_constraints, BioState, BioCfg

CFG = BioCfg(alpha=0.2, effort_limit=23.5, fatigue_rate=0.05,
             recovery_rate=0.02, fatigue_onset=0.7)

def _state(n):
    return BioState(activation=torch.zeros(1, n), capacity=torch.ones(1, n))

def test_lowpass_lags_then_converges():
    s = _state(1)
    cmd = torch.full((1, 1), 10.0)
    out, s = apply_bio_constraints(cmd, s, CFG)
    assert out.item() < 10.0
    for _ in range(200):
        out, s = apply_bio_constraints(cmd, s, CFG)
    assert abs(out.item() - 10.0) < 1e-2

def test_never_exceeds_effort_limit():
    s = _state(4)
    cmd = torch.full((1, 4), 1000.0)
    for _ in range(50):
        out, s = apply_bio_constraints(cmd, s, CFG)
    assert torch.all(out.abs() <= CFG.effort_limit + 1e-4)

def test_sustained_high_torque_fatigues_capacity():
    s = _state(1)
    cmd = torch.full((1, 1), 23.5)
    for _ in range(100):
        _, s = apply_bio_constraints(cmd, s, CFG)
    assert s.capacity.item() < 0.95

def test_capacity_recovers_when_idle():
    s = BioState(activation=torch.zeros(1, 1), capacity=torch.full((1, 1), 0.5))
    cmd = torch.zeros(1, 1)
    for _ in range(100):
        _, s = apply_bio_constraints(cmd, s, CFG)
    assert s.capacity.item() > 0.5
