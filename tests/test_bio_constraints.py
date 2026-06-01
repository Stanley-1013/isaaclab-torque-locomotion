import torch
from torque_loco.bio_constraints import apply_bio, BioState, BioCfg

CFG = BioCfg()  # SATA defaults: kappa_scale=5, gamma=0.6, beta=0.9, all flags True

def _state(e, j):
    return BioState(activation=torch.zeros(e, j), fatigue=torch.zeros(e, j))

def _limits(e, j, tau=23.5, vlim=30.0):
    return torch.full((e, j), tau), torch.full((e, j), vlim)

def test_activation_ema_lags_then_converges():
    s = _state(1, 1); tau, vlim = _limits(1, 1)
    action = torch.full((1, 1), 1.0)          # a_s = 5.0; target = tanh(5/23.5)=0.2098
    jv = torch.zeros(1, 1)
    out1, s = apply_bio(action, jv, tau, vlim, 0.005, s, CFG)
    target = torch.tanh(torch.tensor(5.0 / 23.5)).item()
    # step 1 EMA: alpha = target*0.6 + 0*0.4 = 0.6*target ; torque = alpha*tau (jv=0 -> hill=1)
    assert abs(out1.item() - 0.6 * target * 23.5) < 1e-3
    for _ in range(200):
        out, s = apply_bio(action, jv, tau, vlim, 0.005, s, CFG)
    assert abs(s.activation.item() - target) < 1e-3          # EMA converged to tanh target

def test_no_activation_is_linear_and_can_exceed_limit():
    cfg = BioCfg(activation_process=False, hill_model=False, motor_fatigue=False)
    s = _state(1, 1); tau, vlim = _limits(1, 1)
    action = torch.full((1, 1), 10.0)         # a_s = 50 ; alpha = 50/23.5 ; torque = alpha*tau = 50
    out, s = apply_bio(action, torch.zeros(1, 1), tau, vlim, 0.005, s, cfg)
    assert abs(out.item() - 50.0) < 1e-3                     # linear, unbounded (no tanh, no clip)

def test_hill_reduces_torque_when_velocity_same_direction():
    s = _state(1, 1); tau, vlim = _limits(1, 1)
    action = torch.full((1, 1), 5.0)          # positive activation
    # same-direction velocity (positive) -> torque reduced; opposing -> increased
    out_same, _ = apply_bio(action, torch.full((1, 1), 15.0), tau, vlim, 0.005, _state(1, 1), CFG)
    out_opp,  _ = apply_bio(action, torch.full((1, 1), -15.0), tau, vlim, 0.005, _state(1, 1), CFG)
    assert out_same.item() < out_opp.item()

def test_hill_off_is_plain_activation_times_limit():
    cfg = BioCfg(hill_model=False)
    s = _state(1, 1); tau, vlim = _limits(1, 1)
    out, s = apply_bio(torch.full((1, 1), 5.0), torch.full((1, 1), 20.0), tau, vlim, 0.005, s, cfg)
    assert abs(out.item() - s.activation.item() * 23.5) < 1e-4   # velocity ignored

def test_fatigue_accumulates_under_load_and_decays_when_idle():
    s = _state(1, 1); tau, vlim = _limits(1, 1)
    for _ in range(50):
        _, s = apply_bio(torch.full((1, 1), 5.0), torch.zeros(1, 1), tau, vlim, 0.005, s, CFG)
    loaded = s.fatigue.item(); assert loaded > 0.0
    for _ in range(50):
        _, s = apply_bio(torch.zeros(1, 1), torch.zeros(1, 1), tau, vlim, 0.005, s, CFG)
    assert s.fatigue.item() < loaded                          # decays toward 0 (×0.9/step)

def test_fatigue_off_is_zero():
    cfg = BioCfg(motor_fatigue=False)
    s = _state(1, 1); tau, vlim = _limits(1, 1)
    _, s = apply_bio(torch.full((1, 1), 5.0), torch.zeros(1, 1), tau, vlim, 0.005, s, cfg)
    assert torch.all(s.fatigue == 0.0)

def test_activation_soft_bounds_torque_for_slow_joint():
    s = _state(1, 4); tau, vlim = _limits(1, 4)
    action = torch.full((1, 4), 1000.0)       # absurd; tanh saturates alpha -> |torque|<=tau
    for _ in range(20):
        out, s = apply_bio(action, torch.zeros(1, 4), tau, vlim, 0.005, s, CFG)
    assert torch.all(out.abs() <= 23.5 + 1e-3)
