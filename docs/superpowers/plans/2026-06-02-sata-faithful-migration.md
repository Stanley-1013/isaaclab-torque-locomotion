# Faithful SATA → Isaac Lab Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the full SATA bio-inspired torque-control method (activation low-pass + Hill + fatigue + Gompertz growth curriculum) on the Unitree Go2 in Isaac Lab, and check the full-bio reference matches the Isaac Gym reference cross-engine (per-step reward terms, peak torque / energy / jerk) — a fidelity check, not a claim about what the bio layers "are".

**Architecture:** Manager-based. A `Go2SataEnv(ManagerBasedRLEnv)` overrides `step()` to run SATA's variable-frequency accumulator loop (the one growth component that doesn't fit Isaac Lab's fixed decimation); a `BioActuator(IdealPDActuator)` applies the tanh-EMA activation + Hill + fatigue pipeline; the growth scalar `G(t)` is computed each step into `env._G` and read live by the actuator, the SATA reward terms, and the push-DR event. Tier-2 trains only the full-bio reference and validates it against the reproduction repo's ground-truth envelope numbers — no no-bio ablation.

**Tech Stack:** Isaac Sim 5.1.0, Isaac Lab (main), Python 3.11, PyTorch 2.7.0+cu128, rsl_rl PPO, gymnasium. Sim-free math is pure-torch and TDD'd in the `sata` conda env (no Isaac Sim needed).

**Design spec:** `docs/superpowers/specs/2026-06-02-sata-faithful-migration-design.md` (read it first).

---

## Conventions (READ FIRST)

- **Sim-free tests** (Phases A) run with: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/<file> -q`. The `sata` env has torch; no Isaac Sim import.
- **Sim runs** (Phases B–C) use the Isaac Lab launcher. Canonical preamble:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
  export OMNI_KIT_ACCEPT_EULA=YES CMAKE_POLICY_VERSION_MINIMUM=3.5
  cd ~/workspace/IsaacLab
  CUDA_VISIBLE_DEVICES=<idle_gpu> ./isaaclab.sh -p \
    ~/workspace/isaaclab-torque-locomotion/scripts/train_go2.py --task <id> --headless ...
  ```
  Confirm an idle GPU first (`nvidia-smi`); shared box — lab-citizenship.
- **Commit author:** `git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "..."`.
- **SATA constants** (paper Table III + code, for reference everywhere below):
  `kappa_scale=5.0, gamma=0.6 (EMA new-weight), beta=0.9, k=3e-5, x0=24000,`
  `tau_start_front=7.05, tau_end=23.5, rear_tau=23.5 (constant), f_start=100, f_end=200, dt=0.005`.
- **Joint order gotcha:** SATA's URDF orders joints FL,FR,RL,RR×(hip,thigh,calf). Isaac Lab orders by its own scheme and addresses joints **by name regex**. NEVER assume positional indices. "Front legs" (torque-ceiling growth) = joints whose name matches `^F[LR]_` ; resolve indices by name at runtime. Per-joint velocity limits: hip/thigh = 30.1, calf = 20.07 rad/s (match by name).
- **SATA Go2 defaults to replicate:** base init `pos z = 0.10` (low crouch); default joint angles `hip ±0.1 (L +0.1 / R −0.1), thigh 1.45, calf −2.5`. These differ from Isaac Lab's stock Go2 (z=0.4, thigh 0.8/1.0, calf −1.5) and MUST be overridden.

---

## File Structure

```
src/torque_loco/
  bio_constraints.py   # REWRITE: pure-torch activation(tanh-EMA)+Hill+fatigue (was capacity-clip)
  growth.py            # NEW: pure-torch Gompertz G(t) + torque/freq schedules
  metrics.py           # EXTEND: add SATA-aligned reducers (peak τ, energy, first-diff jerk)
  bio_actuator.py      # NEW: BioActuator(IdealPDActuator) + BioActuatorCfg
  sata_mdp.py          # NEW: obs terms (torque, fatigue), 9 reward terms (G-modulated),
                       #      push-DR event (G-scaled), random-fatigue reset event, G(t) updater
  go2_sata_env.py      # NEW: Go2SataEnv(ManagerBasedRLEnv) step() override + Go2SataEnvCfg(+_PLAY)
  __register__.py      # MODIFY: add Isaac-Velocity-Flat-Go2-Sata-v0 (+ -Play-v0)
tests/
  test_bio_constraints.py  # REWRITE for the new API
  test_growth.py           # NEW
  test_metrics.py          # EXTEND
scripts/
  train_go2.py         # DONE (register-then-delegate launcher) — reused as-is
  eval_metrics.py      # NEW: roll out a checkpoint, dump per-step torque/vel/action CSV
results/               # logs (gitignored), CSVs/plots (committed)
```

Two files carry the real complexity (`bio_constraints.py`, `go2_sata_env.py`); the rest is thin glue.

---

## Phase A — Sim-free math (TDD, no GPU)

### Task A1: Rewrite `bio_constraints.py` (activation tanh-EMA + Hill + fatigue)

**Files:**
- Modify (rewrite): `src/torque_loco/bio_constraints.py`
- Test (rewrite): `tests/test_bio_constraints.py`

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_bio_constraints.py` with:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/test_bio_constraints.py -q`
Expected: FAIL — `apply_bio` / new signature not defined (old module had `apply_bio_constraints`).

- [ ] **Step 3: Rewrite the implementation**

Replace the entire contents of `src/torque_loco/bio_constraints.py` with:

```python
# src/torque_loco/bio_constraints.py
"""Pure-torch SATA biomechanical pipeline (sim-free, unit-tested).

Faithful to SATA (arXiv:2502.12674, Eqs 1-4) and the legged_gym reference:
  activation:  alpha_current = tanh(action * kappa_scale / tau_limit)
               alpha_t       = alpha_current*gamma + alpha_{t-1}*(1-gamma)   (gamma = new-weight)
  Hill:        tau = tau_limit * alpha_t * (1 - sign(alpha_t) * q_dot / q_dot_limit)
  fatigue:     zeta_t = (zeta_{t-1} + |tau|*dt) * beta
Flags mirror SATA's control cfg; the migration runs all-on (reference).
"""
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


@dataclass
class BioState:
    activation: torch.Tensor      # (E, J) EMA activation sign, in (-1, 1) when activation on
    fatigue: torch.Tensor         # (E, J) leaky-integrator fatigue >= 0


def apply_bio(action, joint_vel, torque_limit, vel_limit, dt, state, cfg):
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/test_bio_constraints.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/bio_constraints.py tests/test_bio_constraints.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: rewrite bio_constraints to faithful SATA (tanh-EMA activation + Hill + fatigue)"
```

### Task A2: `growth.py` — Gompertz G(t) + schedules

**Files:**
- Create: `src/torque_loco/growth.py`
- Test: `tests/test_growth.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_growth.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

Run: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/test_growth.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `src/torque_loco/growth.py`:

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/test_growth.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/growth.py tests/test_growth.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: Gompertz growth curriculum (G(t) + torque/freq schedules) + tests"
```

### Task A3: Extend `metrics.py` with SATA-aligned reducers

**Files:**
- Modify: `src/torque_loco/metrics.py`
- Test: `tests/test_metrics.py`

The existing `metrics.py` has `peak_torque`, `action_jerk` (2nd-difference), `mech_energy`. The reproduction's `eval_under_conditions.py` defines envelope metrics differently: **jerk = mean over t of Σ_j|a_t − a_{t-1}| / dt** (first difference). Add SATA-aligned reducers so the cross-engine comparison uses the same definitions; keep the old ones.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

Run: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/test_metrics.py -q`
Expected: FAIL — new names not defined (old tests still pass).

- [ ] **Step 3: Implement**

Append to `src/torque_loco/metrics.py`:

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest tests/ -q`
Expected: all pass (old + new).

- [ ] **Step 5: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/metrics.py tests/test_metrics.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: SATA-aligned envelope reducers (peak torque, energy/step, first-diff jerk)"
```

---

## Phase B — Isaac Lab integration

These touch Isaac Sim; logic is already covered by Phase A tests. Validation is by smoke-train.

### Task B1: `BioActuator` + `BioActuatorCfg`

**Files:**
- Create: `src/torque_loco/bio_actuator.py`

The actuator receives the feed-forward effort from a `JointEffortActionCfg(scale=1.0)` (we keep the
×kappa_scale inside `apply_bio`, so the action term passes the raw action through as `joint_efforts`).
It reads the growth scalar from `env._G` (set on the env each step — Task B3) via a back-reference
set at env construction; if absent it defaults to 1.0 (full capacity). Front-leg torque ceiling
grows with G; rear stays at 23.5. dt is injected after construction.

- [ ] **Step 1: Implement the actuator**

Create `src/torque_loco/bio_actuator.py`:

```python
# src/torque_loco/bio_actuator.py
"""BioActuator: applies the SATA biomechanical pipeline over an effort passthrough.

Subclasses IdealPDActuator (zero PD gains -> feed-forward effort passthrough) and replaces the
torque with the activation(tanh-EMA)+Hill+fatigue output of bio_constraints.apply_bio. Holds
per-env/joint activation + fatigue buffers, reset on env reset. Reads the growth scalar from the
owning env (env._G) so the torque ceiling grows during training; rear legs stay at the max.
The envelope is enforced by tanh (not a hard clip) -> _clip_effort is NOT called.
"""
import re
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .bio_constraints import apply_bio, BioState, BioCfg
from .growth import torque_limit_scale


class BioActuator(IdealPDActuator):
    cfg: "BioActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        e, j = self._num_envs, self.num_joints
        self._biocfg = BioCfg(
            kappa_scale=cfg.kappa_scale, gamma=cfg.gamma, beta=cfg.beta,
            activation_process=cfg.activation_process, hill_model=cfg.hill_model,
            motor_fatigue=cfg.motor_fatigue,
        )
        self.activation = torch.zeros(e, j, device=self._device)
        self.motor_fatigue = torch.zeros(e, j, device=self._device)
        # per-joint velocity limit tensor for the Hill term (hip/thigh 30.1, calf 20.07)
        self._vel_limit = torch.full((e, j), cfg.vel_limit_hip_thigh, device=self._device)
        for k, name in enumerate(self.joint_names):
            if "calf" in name:
                self._vel_limit[:, k] = cfg.vel_limit_calf
        # front-leg mask (name starts FL_/FR_) for the growing torque ceiling
        self._front_mask = torch.tensor(
            [bool(re.match(r"^F[LR]_", n)) for n in self.joint_names],
            device=self._device,
        )
        self._dt = None          # injected by the env after construction
        self._env = None         # back-reference to read env._G; set by the env

    def set_runtime(self, dt, env):
        self._dt = dt
        self._env = env

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self.activation[env_ids] = 0.0
        # random initial fatigue U(0, 0.2*G) per SATA _reset_dofs (scaled by growth)
        g = float(getattr(self._env, "_G", 1.0)) if self._env is not None else 1.0
        hi = 0.2 * g
        if self._biocfg.motor_fatigue and hi > 0.0:
            self.motor_fatigue[env_ids] = torch.rand_like(self.motor_fatigue[env_ids]) * hi
        else:
            self.motor_fatigue[env_ids] = 0.0

    def _current_torque_limit(self):
        g = float(getattr(self._env, "_G", 1.0)) if self._env is not None else 1.0
        front = torque_limit_scale(g, self.cfg.tau_start, self.cfg.tau_end)   # 7.05 -> 23.5
        tl = torch.full_like(self.activation, self.cfg.tau_end)               # rear constant 23.5
        tl[:, self._front_mask] = front
        return tl

    def compute(self, control_action: ArticulationActions, joint_pos, joint_vel):
        action = control_action.joint_efforts                  # raw policy action (scale=1 action term)
        tau_limit = self._current_torque_limit()
        dt = self._dt if self._dt is not None else 0.005
        state = BioState(activation=self.activation, fatigue=self.motor_fatigue)
        torque, state = apply_bio(action, joint_vel, tau_limit, self._vel_limit, dt, state, self._biocfg)
        self.activation, self.motor_fatigue = state.activation, state.fatigue
        self.computed_effort = torque
        self.applied_effort = torque                           # NO hard clip; tanh is the envelope
        control_action.joint_efforts = torque
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class BioActuatorCfg(IdealPDActuatorCfg):
    class_type: type = BioActuator
    kappa_scale: float = 5.0
    gamma: float = 0.6
    beta: float = 0.9
    activation_process: bool = True
    hill_model: bool = True
    motor_fatigue: bool = True
    tau_start: float = 7.05        # front-leg torque ceiling at G=0
    tau_end: float = 23.5          # torque ceiling at G=1 (and rear-leg constant)
    vel_limit_hip_thigh: float = 30.1
    vel_limit_calf: float = 20.07
```

- [ ] **Step 2: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/bio_actuator.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: BioActuator (tanh-EMA activation + Hill + fatigue, growing torque ceiling)"
```

### Task B2: `sata_mdp.py` — obs / reward / event terms

**Files:**
- Create: `src/torque_loco/sata_mdp.py`

Implements custom manager terms. All read `env._G` for growth modulation. `phi(x)=exp(-4|x|)`.

- [ ] **Step 1: Implement**

Create `src/torque_loco/sata_mdp.py`:

```python
# src/torque_loco/sata_mdp.py
"""Custom SATA manager terms: observations (torque, fatigue), the 9 SATA reward terms
(growth-modulated via env._G), the G-scaled push event, and a random-fatigue reset.
phi(x) = exp(-4|x|) (SATA Gaussian-shaped tracking kernel; == exp(-|x|/0.25))."""
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _phi(x):
    return torch.exp(-4.0 * x.abs())

def _G(env):
    return float(getattr(env, "_G", 1.0))

def _actuator(env, name="base_legs"):
    return env.scene["robot"].actuators[name]

# ---- observations ----
def applied_torque(env, asset_cfg=SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]

def motor_fatigue(env, asset_cfg=SceneEntityCfg("robot")):
    return _actuator(env).motor_fatigue

# ---- reward terms (return (num_envs,)) ----
def track_x(env, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command(command_name)
    g = _G(env)
    rng = env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x
    mid = 0.5 * (rng[0] + rng[1])
    return _phi(vx - mid) * (1.0 - g) + _phi(vx - cmd[:, 0]) * (1.0 + g)

def track_y(env, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _phi(asset.data.root_lin_vel_b[:, 1] - cmd[:, 1]) * _G(env)

def track_yaw(env, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _phi(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2]) * _G(env)

def base_height(env, target_height=0.3, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    g = _G(env)
    h = torch.clamp(asset.data.root_pos_w[:, 2], max=target_height)
    gx = asset.data.projected_gravity_b[:, 0]
    head_up = torch.maximum(gx, -torch.clamp(0.2 * (1.5 - 2.0 * g) * torch.ones_like(gx), max=0.0))
    return h * (1.0 + g) - head_up

def roll_penalty(env, asset_cfg=SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.projected_gravity_b[:, 1].abs()

def lin_vel_z(env, asset_cfg=SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.root_lin_vel_b[:, 2] ** 2

def soft_dof_pos_limits(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    lo = asset.data.soft_joint_pos_limits[..., 0]
    hi = asset.data.soft_joint_pos_limits[..., 1]
    out = -(q - lo).clamp(max=0.0) + (q - hi).clamp(min=0.0)
    return out.sum(dim=1)

def fatigue_penalty(env, kappa_scale=5.0, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    fatigue = _actuator(env).motor_fatigue
    action_scaled = (asset.data.joint_effort_target * kappa_scale).abs()
    return (fatigue * action_scaled).sum(dim=1)

def joint_acc_l2(env, asset_cfg=SceneEntityCfg("robot")):
    return (env.scene[asset_cfg.name].data.joint_acc ** 2).sum(dim=1)

# ---- events ----
def push_scaled_by_growth(env, env_ids, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    """Push by setting base velocity, magnitude scaled by env._G (SATA max_push_vel*general_scale)."""
    asset: Articulation = env.scene[asset_cfg.name]
    g = _G(env)
    vel = torch.zeros((len(env_ids), 6), device=env.device)
    for i, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        if key in velocity_range:
            lo, hi = velocity_range[key]
            vel[:, i] = (torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo) * g
    root = asset.data.root_state_w[env_ids].clone()
    root[:, 7:13] += vel
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
```

(Note: verify exact data-field names — `root_lin_vel_b`, `root_ang_vel_b`, `projected_gravity_b`,
`root_pos_w`, `soft_joint_pos_limits`, `joint_effort_target`, `joint_acc` — against the installed
`ArticulationData` during Step 2; adjust if a name differs. The random-fatigue reset is handled in
`BioActuator.reset`, so no separate reset event is needed for fatigue; the base-pose/dof jitter reset
is configured via stock `mdp.reset_root_state_uniform` + `mdp.reset_joints_by_scale` in Task B3.)

- [ ] **Step 2: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/sata_mdp.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: SATA mdp terms (torque/fatigue obs, 9 G-modulated rewards, G-scaled push)"
```

### Task B3: `Go2SataEnv` (variable-freq step) + `Go2SataEnvCfg`

**Files:**
- Create: `src/torque_loco/go2_sata_env.py`

The env subclass: (1) on construction wire `dt`/back-ref into the BioActuator; (2) override `step()`
to compute `G(t)` from `common_step_counter` into `self._G` and run SATA's variable-frequency
accumulator (decimation determined by `current_freq` rather than a fixed `cfg.decimation`).

- [ ] **Step 1: Read the stock step() to copy faithfully**

Run: `sed -n '150,230p' ~/workspace/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
Expected: see the exact body of `step()` (action processing, the `for _ in range(self.cfg.decimation)` physics loop, `sim.step`, counters, obs/reward/done/extras). Copy its structure into the override below, replacing ONLY the fixed decimation loop with the accumulator. Record the real method body in `docs/operations.md`.

- [ ] **Step 2: Implement env + cfg**

Create `src/torque_loco/go2_sata_env.py`:

```python
# src/torque_loco/go2_sata_env.py
"""Go2SataEnv: manager-based Go2 velocity env with SATA's full bio stack.

Overrides step() to (a) update the Gompertz growth scalar env._G each step and (b) run a
variable-frequency physics loop (control 100->200 Hz over training) instead of fixed decimation.
All other managers are stock. The cfg installs the BioActuator, SATA reward/obs/event terms,
SATA command ranges, SATA defaults (base z=0.10, thigh 1.45, calf -2.5), and 200 Hz physics.
"""
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from . import sata_mdp
from .bio_actuator import BioActuatorCfg
from .growth import gompertz, control_freq

PHYSICS_HZ = 200.0
PHYSICS_DT = 1.0 / PHYSICS_HZ      # 0.005


class Go2SataEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._G = 1.0
        act = self.scene["robot"].actuators["base_legs"]
        if hasattr(act, "set_runtime"):
            act.set_runtime(PHYSICS_DT, self)
        self._control_accum = 0.0

    def step(self, action):
        # Update growth scalar from cumulative env steps (drives actuator/reward/event live).
        self._G = gompertz(self.common_step_counter)
        # Variable control frequency: number of physics sub-steps this env step.
        freq = control_freq(self._G)                       # 100 -> 200 Hz
        # SATA accumulator: run sub-steps while accumulated control-time < one control period.
        n_sub = 0
        self._control_accum = 0.0
        while self._control_accum * freq < 1.0:
            n_sub += 1
            self._control_accum += PHYSICS_DT
        n_sub = max(1, n_sub)
        # Delegate to the stock step() body but with a dynamic decimation.
        # IMPLEMENTATION NOTE (Task B3 Step 1): paste the stock step() body here, replacing
        # `for _ in range(self.cfg.decimation):` with `for _ in range(n_sub):`. Everything else
        # (action processing, sim.step, counters, manager compute, obs/reward/done/extras) stays.
        return self._stepped(action, n_sub)

    def _stepped(self, action, n_sub):
        raise NotImplementedError(
            "Paste stock ManagerBasedRLEnv.step() body here with the decimation loop replaced by "
            "`for _ in range(n_sub)`; see Task B3 Step 1 for the exact source to copy."
        )


@configclass
class Go2SataEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # --- 200 Hz physics; control freq grows via the env step() override ---
        self.sim.dt = PHYSICS_DT
        self.decimation = 1                  # baseline; the env overrides the loop count
        self.sim.render_interval = 4
        self.episode_length_s = 10.0
        # --- SATA Go2 defaults (low crouch + folded legs) ---
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.10)
        self.scene.robot.init_state.joint_pos = {
            "FL_hip_joint": 0.1, "RL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
            "FL_thigh_joint": 1.45, "RL_thigh_joint": 1.45, "FR_thigh_joint": 1.45, "RR_thigh_joint": 1.45,
            "FL_calf_joint": -2.5, "RL_calf_joint": -2.5, "FR_calf_joint": -2.5, "RR_calf_joint": -2.5,
        }
        # --- action: raw torque passthrough (scale=1; *kappa_scale lives in the actuator) ---
        self.actions.joint_pos = None
        self.actions.joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0,
        )
        # --- BioActuator replaces the stock DCMotor group ---
        self.scene.robot.actuators["base_legs"] = BioActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
            effort_limit=1000.0, velocity_limit=30.0,    # no hard clip; envelope is the tanh
        )
        # --- SATA command ranges (fixed) ---
        cr = self.commands.base_velocity.ranges
        cr.lin_vel_x = (-0.5, 1.5); cr.lin_vel_y = (-0.5, 0.5); cr.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        # --- observations: SATA 60-dim (replace policy group) ---
        p = self.observations.policy
        p.base_lin_vel.scale = 2.0; p.base_ang_vel.scale = 0.25
        p.joint_pos.scale = 1.0; p.joint_vel.scale = 0.05
        p.height_scan = None
        p.actions = None                                  # SATA uses torque + fatigue, not last_action
        p.applied_torque = ObservationTermCfg(func=sata_mdp.applied_torque)
        p.motor_fatigue = ObservationTermCfg(func=sata_mdp.motor_fatigue)
        # --- rewards: replace with SATA's 9 terms (weights = scale*dt; preserve ratios) ---
        R = self.rewards
        for name in list(vars(R)):
            setattr(R, name, None)                        # clear stock terms
        DT = PHYSICS_DT
        R.track_x = RewardTermCfg(func=sata_mdp.track_x, weight=10.0 * DT)
        R.track_y = RewardTermCfg(func=sata_mdp.track_y, weight=5.0 * DT)
        R.track_yaw = RewardTermCfg(func=sata_mdp.track_yaw, weight=5.0 * DT)
        R.base_height = RewardTermCfg(func=sata_mdp.base_height, weight=5.0 * DT)
        R.roll = RewardTermCfg(func=sata_mdp.roll_penalty, weight=-5.0 * DT)
        R.lin_vel_z = RewardTermCfg(func=sata_mdp.lin_vel_z, weight=-5.0 * DT)
        R.joint_limits = RewardTermCfg(func=sata_mdp.soft_dof_pos_limits, weight=-5.0 * DT)
        R.fatigue = RewardTermCfg(func=sata_mdp.fatigue_penalty, weight=-0.05 * DT)
        R.joint_acc = RewardTermCfg(func=sata_mdp.joint_acc_l2, weight=-1e-6 * DT)
        # --- events: SATA DR ---
        E = self.events
        E.push_robot = EventTermCfg(
            func=sata_mdp.push_scaled_by_growth, mode="interval", interval_range_s=(4.0, 4.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5),
                                       "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}},
        )
        if hasattr(E, "add_base_mass") and E.add_base_mass is not None:
            E.add_base_mass.params["mass_distribution_params"] = (-1.0, 5.0)
        # friction / COM randomization: keep stock ranges if present; otherwise add per spec.


@configclass
class Go2SataEnvCfg_PLAY(Go2SataEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
```

- [ ] **Step 3: Fill in `_stepped` from the real source**

Using the body captured in Step 1, replace `_stepped`'s `NotImplementedError` with the stock
`step()` body, changing only the physics loop to `for _ in range(n_sub):`. Keep `common_step_counter`
and `_sim_step_counter` updates identical to stock so logging/curriculum cadence is unchanged.

- [ ] **Step 4: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/go2_sata_env.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: Go2SataEnv (variable-freq step override) + full SATA env cfg"
```

### Task B4: Register + smoke-train

**Files:**
- Modify: `src/torque_loco/__register__.py`

- [ ] **Step 1: Add registration**

Append to `src/torque_loco/__register__.py`:

```python
gym.register(
    id="Isaac-Velocity-Flat-Go2-Sata-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO,
    },
)
gym.register(
    id="Isaac-Velocity-Flat-Go2-Sata-Play-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO,
    },
)
```

(Note: `entry_point` is the custom env class, not `ManagerBasedRLEnv` — gym.make will instantiate
`Go2SataEnv`. Verify the rsl_rl train.py passes `cfg=` to this entry point; it does, via hydra.)

- [ ] **Step 2: Smoke-train 10 iterations**

Confirm an idle GPU (`nvidia-smi`), then:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES CMAKE_POLICY_VERSION_MINIMUM=3.5
cd ~/workspace/IsaacLab
CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p \
  ~/workspace/isaaclab-torque-locomotion/scripts/train_go2.py \
  --task Isaac-Velocity-Flat-Go2-Sata-v0 --headless --num_envs 1024 --max_iterations 10 \
  > ~/workspace/isaaclab-torque-locomotion/results/sata_smoke.log 2>&1
```
Expected: exit 0; `Learning iteration 0..9`; finite Mean reward (no NaN); no shape/term errors.
If a `sata_mdp` data-field name is wrong, the traceback names it — fix and re-run.
If reward is NaN/explodes immediately, check the variable-freq loop and the tanh path (early G≈0
makes torque ceiling 7.05 and freq 100 Hz — both should be stabilizing, not destabilizing).

- [ ] **Step 3: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add src/torque_loco/__register__.py docs/operations.md
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: register Go2-Sata task + smoke-train passes (full bio stack)"
```

---

## Phase C — Train the reference + validate the envelope

### Task C1: Train the full-bio reference

- [ ] **Step 1: Launch seeds (detached, idle GPUs)**

Reuse `scripts/dispatch_seeds.sh` (built earlier):
```bash
cd ~/workspace/isaaclab-torque-locomotion
nohup setsid scripts/dispatch_seeds.sh <gpu> Isaac-Velocity-Flat-Go2-Sata-v0 go2_sata <seed...> \
  </dev/null > results/dispatch_sata_gpu<gpu>.log 2>&1 & disown
```
Use 3000 max_iterations (SATA) — set `MAX_ITER=3000` env var for the dispatcher. **Seed count:
≥8 seeds (user directive 2026-06-02)** — matches SATA's 8-seed ground-truth reference (104±16), so
the cross-engine reproducibility claim carries mean±std. Distribute across the ~3 idle GPUs via the
dispatcher (≈3 seeds/GPU sequentially); time is not a constraint. Report mean±std final reward.

- [ ] **Step 2: Confirm it walks**

After training: `play.py`-style rollout via the launcher with `--task Isaac-Velocity-Flat-Go2-Sata-Play-v0
--load_run <run> --headless --video --video_length 300`. Watch the clip: sustained forward locomotion,
tracks (vx,vy,yaw), does not fall. Record final mean reward per seed.

- [ ] **Step 3: Commit clip + curve + log**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add results docs/operations.md
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "results: full-SATA Go2 reference walks in Isaac Lab"
```

### Task C2: Roll out + envelope metrics vs ground truth

**Files:**
- Create: `scripts/eval_metrics.py`

- [ ] **Step 1: Write the rollout dumper**

Create `scripts/eval_metrics.py` modeled on Isaac Lab's `play.py`: load the env (`*-Sata-Play-v0`) +
the trained policy, run N episodes headless, and each step append `robot.data.applied_torque`,
`robot.data.joint_vel`, and the raw policy action to buffers; write a CSV `results/metrics_sata.csv`
with per-step rows. Reuse the `train_go2.py` register-then-delegate pattern (import
`torque_loco.__register__`, then run the eval body) so the task id resolves. Confirm the data field
names against the smoke run.

- [ ] **Step 2: Compute the envelope + plot**

Create `scripts/plot_envelope.py`: load `results/metrics_sata.csv`, apply
`metrics.sata_peak_torque / sata_energy_per_step / sata_mean_jerk`, and bar-plot them with the
**23.5 N·m sim clip** and **45 N·m real limit** as reference lines. Expected (ground truth from
`bio-inspired-adaptive-locomotion/results/phase3-bio-claims-and-robustness`): reference **peak τ ≈ 22.5
N·m** (inside the envelope), energy/jerk in the reference's nominal band.

- [ ] **Step 3: Record the cross-engine comparison**

In `docs/operations.md`, tabulate Isaac-Lab reference vs SATA ground-truth (peak τ / energy/step /
jerk). State explicitly: cross-engine **reward** is not directly comparable; the **envelope metrics**
are the reproducibility claim. Note seed count and that results are preliminary/sim-only.

- [ ] **Step 4: Commit**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add scripts/eval_metrics.py scripts/plot_envelope.py results docs/operations.md
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "results: Tier-2 envelope reproduces (full SATA reference, Isaac Lab vs ground truth)"
```

**✅ Tier 2 secured: the full-SATA reference reproduces the feasibility-envelope finding cross-engine.**

---

## Self-review notes (author)

- **Spec coverage:** §3.1 step override → B3; §3.2 G(t)+5 schedules → growth A2 (G,τ,f), B1 (τ ceiling
  in actuator), B2 (push DR), B3 (freq loop, reward G-mod); §3.3 BioActuator → B1 (+ math A1); §3.4
  obs(60) → B2/B3; §3.5 reward(9) → B2/B3; §3.6 commands+DR+reset → B3 (commands/DR) + B1 (random
  fatigue reset); §3.7 constants → B3; §4 sim-free TDD → A1/A2/A3; §5 ground-truth validation → C2.
- **Placeholder scan:** the only deferred bodies are explicit "paste the stock step() source"
  (B3 Step 1/3, with the exact source command) and the `eval_metrics.py`/`plot_envelope.py` bodies
  (C2, modeled on the named existing script) — flagged build steps against real references, not hidden
  TODOs. All math/actuator/cfg/mdp code is complete and literal.
- **Type consistency:** `apply_bio(action, joint_vel, torque_limit, vel_limit, dt, state, cfg) ->
  (torque, BioState)`; `BioState(activation, fatigue)`; `BioCfg(kappa_scale, gamma, beta,
  activation_process, hill_model, motor_fatigue)`; `BioActuatorCfg(... tau_start, tau_end,
  vel_limit_hip_thigh, vel_limit_calf)`; `gompertz(step)`, `torque_limit_scale(g, tau_start, tau_end)`,
  `control_freq(g)`; metrics `sata_peak_torque/sata_energy_per_step/sata_mean_jerk` — consistent across
  tasks. `env._G` read by actuator + all reward/event terms.
- **Open verifications (flagged in-task):** ArticulationData field names (B2 Step 2 note); the stock
  `step()` body (B3 Step 1); USD effort-limit clamp not biting (effort_limit=1000 in cfg mitigates);
  whether Isaac Lab RewardManager multiplies weight by step_dt (B3 — adjust the `*DT` if double-scaled).
```
