import math
from torque_loco.growth import gompertz, torque_limit_scale, control_freq

def test_gompertz_at_inflection_is_exp_minus_1():
    # G(x0) = exp(-exp(0)) = exp(-1) ~ 0.3679
    assert abs(gompertz(24000) - math.exp(-1)) < 1e-6

def test_gompertz_starts_near_zero_and_saturates():
    assert gompertz(0) < 0.15                 # early training, body barely unlocked
    assert gompertz(200000) > 0.99            # asymptotes to 1

def test_gompertz_is_monotonic_increasing():
    vals = [gompertz(s) for s in range(0, 60000, 5000)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))

def test_torque_limit_scale_interpolates_start_to_end():
    assert abs(torque_limit_scale(0.0, 7.05, 23.5) - 7.05) < 1e-6
    assert abs(torque_limit_scale(1.0, 7.05, 23.5) - 23.5) < 1e-6
    assert abs(torque_limit_scale(0.5, 7.05, 23.5) - 15.275) < 1e-6

def test_control_freq_interpolates_100_to_200():
    assert abs(control_freq(0.0) - 100.0) < 1e-6
    assert abs(control_freq(1.0) - 200.0) < 1e-6
