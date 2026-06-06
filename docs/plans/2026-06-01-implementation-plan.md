# Isaac Lab Torque-Locomotion Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port SATA's torque-control + bio feasibility-envelope idea from the deprecated Isaac Gym to Isaac Lab on the Unitree Go2, check whether the prior repo's peak-torque behaviour also appears across engines, and (stretch) test cross-engine zero-shot transfer as a sim2real proxy — all within a one-week course-report window.

**Architecture:** Build on Isaac Lab's stock manager-based Go2 velocity task. Swap the action term from joint-position to **joint-effort** (torque control). The bio layer (activation low-pass + fatigue capacity) lives in a custom `BioActuator` whose constraint math is factored into a **pure-torch, sim-free function** so it can be unit-tested without launching the simulator. Compare Go2 trained with vs without the bio actuator on training reward AND actuator-feasibility metrics (peak torque, jerk, energy), mirroring the SATA repo's Phase-3 analysis.

**Tech Stack:** Isaac Sim 5.1.0, Isaac Lab (main), Python 3.11, PyTorch 2.7.0+cu128, rsl_rl PPO, conda. Lab: 4×A6000, K8s pod, NFS `$HOME`, no sudo, existing TigerVNC/noVNC (port 6080).

---

## ⚠️ Pre-flight risk gates (read before Task 0)

This plan is **gated on environment bootstrap (Phase 0)**. Two container-specific risks can block everything; they are front-loaded so failure is found on Day 0, not Day 5:

- **R1a — no sudo.** The official install says `sudo apt install cmake build-essential`. You have no sudo. Mitigation in Task 0.2 (install cmake via conda).
- **R1b — CUDA 12.8.** Isaac Sim 5.1 wants torch cu128; the SATA env is cu117. The container driver must support the CUDA 12.8 runtime. Task 0.1 checks `nvidia-smi` driver version FIRST; if it can't support 12.8, STOP and escalate to the user before sinking more time.
- **R2 — obs/action contract** (Phase 3 only): `legged_gym` obs ordering ≠ Isaac Lab manager ordering. Detailed when we reach it.

**Tiered fallback (each tier is a complete, presentable result):** Phase 1 = Tier 1 (migration working). Phase 2 = Tier 2 (peak-torque behaviour also appears). Phase 3 = Tier 3 (transfer). Phase 5 = Tier 3b (new robot). If a tier is cut, the talk still stands on the tier below.

**API caveat (not a placeholder license):** code blocks below are grounded in the Isaac Lab `main` docs as of 2026-06-01 but the exact API of the *installed* 5.1 build must be confirmed. Where a task says "verify the API," that verification IS the first step — do not assume.

---

## File structure (new repo `isaaclab-torque-locomotion`)

```
src/torque_loco/
  __init__.py
  bio_constraints.py      # pure-torch sim-free math: activation low-pass + fatigue
  bio_actuator.py         # BioActuator(IdealPDActuator) wrapping bio_constraints
  go2_torque_env_cfg.py   # Go2 velocity cfg subclass: JointEffortAction + actuator swap
  metrics.py              # peak torque / jerk / energy from logged rollouts (sim-free)
  __register__.py         # gym.register the new task ids
tests/
  test_bio_constraints.py # TDD target for the bio math
  test_metrics.py         # TDD target for the metric reducers
scripts/
  train_go2.sh            # thin wrappers around Isaac Lab's rsl_rl train/play
  eval_metrics.py         # roll out a checkpoint, dump per-step torque/vel CSV
docs/
  specs/2026-06-01-design.md
  plans/2026-06-01-implementation-plan.md
  operations.md           # running log of what actually happened (env quirks etc.)
results/                  # CSVs, plots, clips (committed; logs/ gitignored)
```

Two files carry the real complexity (`bio_constraints.py`, `go2_torque_env_cfg.py`); everything else is thin glue. Files that change together live together under `src/torque_loco/`.

---

## Phase 0 — Environment bootstrap (D0) · Tier-0 gate

No TDD here — this is discovery with hard verification gates. Log every quirk to `docs/operations.md` as you go.

### Task 0.1: Confirm the driver can run CUDA 12.8

- [ ] **Step 1: Check the driver and idle GPUs**

Run: `nvidia-smi`
Expected: note the "CUDA Version:" in the top-right (driver's max supported runtime) and which GPUs are idle. **CUDA Version must be ≥ 12.8.** Also note SATA jobs / other tenants — pick a confirmed-idle GPU (lab-citizenship rule).

- [ ] **Step 2: Decision gate**

If `CUDA Version` < 12.8 → **STOP. Escalate to the user** (driver upgrade needs cluster admin; not solvable in-session). Record the blocker in `docs/operations.md`. Do not proceed.
If ≥ 12.8 → continue.

### Task 0.2: Create the Isaac Lab conda env (no sudo)

- [ ] **Step 1: Create env and install Isaac Sim**

```bash
conda create -y -n isaaclab python=3.11
conda activate isaaclab
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

- [ ] **Step 2: Install PyTorch (cu128)**

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

- [ ] **Step 3: Get cmake WITHOUT sudo (R1a mitigation)**

```bash
conda install -y -c conda-forge cmake make cxx-compiler
```
Expected: `cmake --version` ≥ 3.22 resolves inside the env. (This replaces `sudo apt install cmake build-essential`.)

- [ ] **Step 4: Clone and install Isaac Lab**

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```
Expected: completes without a fatal error. If it fails on a missing system lib, record the exact lib in `docs/operations.md` and try the conda-forge equivalent before escalating.

### Task 0.3: Verify headless sim + render path

- [ ] **Step 1: Headless smoke test**

Run: `cd ~/workspace/IsaacLab && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless`
Expected: exits cleanly (no GL/driver crash). A non-headless run needs the VNC display; headless is what training uses.

- [ ] **Step 2: Confirm the stock Go2 task trains for a few iterations**

Run:
```bash
cd ~/workspace/IsaacLab
CUDA_VISIBLE_DEVICES=<idle_gpu> ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --num_envs 1024 --max_iterations 10
```
Expected: rsl_rl prints rising mean reward for ~10 iterations; a run dir appears under `logs/rsl_rl/...`. **This is the Tier-0 success gate — Isaac Lab works on this box.**

- [ ] **Step 3: Confirm render-to-video works**

Run the same `play.py` with `--headless --video --video_length 200` on a trained-or-random checkpoint; confirm an mp4 is written. (If `--video` needs the VNC EGL context, reuse the SATA VNC setup; log the working invocation.)

- [ ] **Step 4: Commit the ops log**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add docs/operations.md
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "ops: Isaac Lab 5.1 bootstrap log (env, driver, smoke tests)"
```

---

## Phase 1 — Go2 torque-control baseline (D1) · Tier 1

Convert the stock Go2 task from position to **effort** control and train a walking policy. This alone demonstrates the platform + paradigm migration.

### Task 1.1: Scaffold the package

- [ ] **Step 1: Create the package skeleton**

```bash
cd ~/workspace/isaaclab-torque-locomotion
mkdir -p src/torque_loco tests scripts results
touch src/torque_loco/__init__.py
```

- [ ] **Step 2: Add a pytest sanity test**

```python
# tests/test_smoke.py
def test_import():
    import torque_loco  # noqa: F401
```

- [ ] **Step 3: Run it**

Run: `conda activate isaaclab && cd ~/workspace/isaaclab-torque-locomotion && PYTHONPATH=src pytest tests/test_smoke.py -q`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add src tests && git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: package skeleton + smoke test"
```

### Task 1.2: Go2 effort-control env cfg

**Files:** Create `src/torque_loco/go2_torque_env_cfg.py`, `src/torque_loco/__register__.py`

- [ ] **Step 1: Verify the API surface first**

Open the installed task's source to confirm the exact class names/import paths for THIS build:
```bash
python -c "import isaaclab_tasks.manager_based.locomotion.velocity.config.unitree_go2 as m; print(m.__file__)"
```
Read that file and its `rough_env_cfg` / `flat_env_cfg` to confirm: the env cfg class name, the `actions` term type (expect `JointPositionActionCfg`), and the robot articulation cfg import. Record the real names in `docs/operations.md`.

- [ ] **Step 2: Write the effort-control cfg**

```python
# src/torque_loco/go2_torque_env_cfg.py
from isaaclab.utils import configclass
from isaaclab.actuators import IdealPDActuatorCfg
import isaaclab.envs.mdp as mdp
# NOTE: confirm these import paths against Task 1.1 Step 1 output before running.
from isaaclab_tasks.manager_based.locomotion.velocity.config.unitree_go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

# Go2 joint torque limit: SATA used a 23.5 N·m sim clip (real Go2 peak 45 N·m).
GO2_EFFORT_LIMIT = 23.5

@configclass
class Go2TorqueEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 1) Policy outputs joint torques directly (paradigm migration).
        self.actions.joint_pos = None  # drop the position action term
        self.actions.joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=GO2_EFFORT_LIMIT,  # action in [-1,1] -> torque in N·m
        )
        # 2) Make the actuator pass effort straight through (zero PD gains)
        #    so the policy's torque is what reaches the joint.
        self.scene.robot.actuators["base_legs"] = IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=GO2_EFFORT_LIMIT,
            velocity_limit=30.0,
        )
```
(The actuator group key — here `"base_legs"` — must match the real Go2 cfg from Step 1; fix if different.)

- [ ] **Step 3: Register the task id**

```python
# src/torque_loco/__register__.py
import gymnasium as gym
from .go2_torque_env_cfg import Go2TorqueEnvCfg

gym.register(
    id="Isaac-Velocity-Flat-Go2-Torque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2TorqueEnvCfg,
            "rsl_rl_cfg_entry_point":
            "isaaclab_tasks.manager_based.locomotion.velocity.config.unitree_go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg"},
)
```
(Confirm the PPO runner cfg entry point string against Step 1's file listing.)

- [ ] **Step 4: Smoke-train 10 iterations**

```bash
cd ~/workspace/IsaacLab
CUDA_VISIBLE_DEVICES=<idle_gpu> ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Go2-Torque-v0 --headless --num_envs 1024 --max_iterations 10
```
Expected: trains without shape/term errors; reward is finite (not NaN). If the policy explodes immediately, halve `scale` and note it. (Use `--task` discovery: ensure `__register__` is imported — add `import torque_loco.__register__` via an `__init__` import or `--task` plugin hook; record the working mechanism.)

- [ ] **Step 5: Commit**

```bash
git add src && git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: Go2 effort-control (torque) env cfg + task registration"
```

### Task 1.3: Train the baseline to walking + validate vs SATA

- [ ] **Step 1: Full training run(s)**

```bash
cd ~/workspace/IsaacLab
CUDA_VISIBLE_DEVICES=<idle_gpu> nohup setsid ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Go2-Torque-v0 --headless --num_envs 4096 --max_iterations 1500 \
  </dev/null > ~/workspace/isaaclab-torque-locomotion/results/go2_torque_s1.log 2>&1 & disown
```
Launch 1–2 seeds first (vary an `--seed` flag if the script exposes one; else vary run name). Add more seeds across idle GPUs only if per-run wall-clock is cheap (resource-gated, per the spec). Detached so it survives disconnects.

- [ ] **Step 2: Verify it learned to walk**

After the run: `play.py --task ... --num_envs 32 --load_run <run> --headless --video --video_length 300`. Watch the clip; confirm sustained forward locomotion. Record final mean reward.

- [ ] **Step 3: Sanity-check against SATA ground truth**

The SATA Isaac-Gym Go2 reference scored reward 104±16 on its own scale — **the scales are NOT comparable** (different reward terms/engine). The valid sanity check is qualitative: does it walk and track the velocity command without falling? Note explicitly in `docs/operations.md` that cross-engine reward magnitudes are not directly comparable (anti-over-claim).

- [ ] **Step 4: Commit the clip + curve + log**

```bash
cd ~/workspace/isaaclab-torque-locomotion
git add results docs/operations.md
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "results: Tier-1 Go2 torque-control baseline walks in Isaac Lab"
```

**✅ Tier 1 secured: a new-engine torque-control Go2 walking policy + clip + curve.**

---

## Phase 2 — Bio actuator + envelope reproduction (D2–D3) · Tier 2

The technical core. TDD the bio math (sim-free), wrap it in an actuator, retrain with/without, compare feasibility metrics.

### Task 2.1: Bio-constraint math (pure torch, TDD)

**Files:** Create `src/torque_loco/bio_constraints.py`, Test `tests/test_bio_constraints.py`

The bio layer, mirroring SATA: (a) **activation low-pass** — the realised torque follows the commanded torque through a 1st-order filter `a_t = a_{t-1} + alpha*(cmd - a_{t-1})`; (b) **fatigue capacity** — a per-joint capacity `c_t in (0,1]` that *drops* when |torque| is a large fraction of the limit and *recovers* otherwise, then *clips* the torque to `c_t * effort_limit`.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert out.item() < 10.0                      # lags on step 1
    for _ in range(200):
        out, s = apply_bio_constraints(cmd, s, CFG)
    assert abs(out.item() - 10.0) < 1e-2          # converges to command

def test_never_exceeds_effort_limit():
    s = _state(4)
    cmd = torch.full((1, 4), 1000.0)              # absurd command
    for _ in range(50):
        out, s = apply_bio_constraints(cmd, s, CFG)
    assert torch.all(out.abs() <= CFG.effort_limit + 1e-4)

def test_sustained_high_torque_fatigues_capacity():
    s = _state(1)
    cmd = torch.full((1, 1), 23.5)                # at the limit
    for _ in range(100):
        _, s = apply_bio_constraints(cmd, s, CFG)
    assert s.capacity.item() < 0.95               # capacity dropped

def test_capacity_recovers_when_idle():
    s = BioState(activation=torch.zeros(1, 1), capacity=torch.full((1, 1), 0.5))
    cmd = torch.zeros(1, 1)
    for _ in range(100):
        _, s = apply_bio_constraints(cmd, s, CFG)
    assert s.capacity.item() > 0.5                # recovered
```

- [ ] **Step 2: Run, verify they fail**

Run: `PYTHONPATH=src pytest tests/test_bio_constraints.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the math**

```python
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
    # 1) activation low-pass
    act = state.activation + cfg.alpha * (cmd_torque - state.activation)
    # 2) fatigue: load fraction drives capacity down past onset, else recover
    load = act.abs() / cfg.effort_limit
    overload = torch.clamp(load - cfg.fatigue_onset, min=0.0)
    cap = state.capacity - cfg.fatigue_rate * overload
    cap = cap + cfg.recovery_rate * (1.0 - load).clamp(min=0.0)
    cap = cap.clamp(0.05, 1.0)
    # 3) clip torque to the fatigued capacity envelope
    limit = cap * cfg.effort_limit
    out = act.clamp(-limit, limit)
    return out, BioState(activation=act, capacity=cap)
```

- [ ] **Step 4: Run, verify pass**

Run: `PYTHONPATH=src pytest tests/test_bio_constraints.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/torque_loco/bio_constraints.py tests/test_bio_constraints.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: bio-constraint math (activation low-pass + fatigue capacity) + tests"
```

### Task 2.2: BioActuator wrapping the math

**Files:** Create `src/torque_loco/bio_actuator.py`

- [ ] **Step 1: Verify the ActuatorBase API**

```bash
python -c "import isaaclab.actuators as a; print(a.IdealPDActuator.compute.__doc__); print([x for x in dir(a)])"
```
Confirm: `IdealPDActuator(ActuatorBase)`, `compute(control_action, joint_pos, joint_vel) -> ArticulationActions`, and that `control_action.joint_efforts` carries the feed-forward effort. Record signatures.

- [ ] **Step 2: Implement the actuator**

```python
# src/torque_loco/bio_actuator.py
import torch
from dataclasses import MISSING
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from .bio_constraints import apply_bio_constraints, BioState, BioCfg

class BioActuator(IdealPDActuator):
    """Effort-passthrough actuator that applies SATA's bio constraints
    (activation low-pass + fatigue capacity) to the commanded torque."""
    cfg: "BioActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        n_env, n_j = self._joint_indices_shape()  # confirm helper in Step 1
        self._bio_cfg = BioCfg(alpha=cfg.alpha, effort_limit=cfg.effort_limit_value,
                               fatigue_rate=cfg.fatigue_rate, recovery_rate=cfg.recovery_rate,
                               fatigue_onset=cfg.fatigue_onset)
        self._state = BioState(
            activation=torch.zeros(n_env, n_j, device=self._device),
            capacity=torch.ones(n_env, n_j, device=self._device))

    def reset(self, env_ids):
        self._state.activation[env_ids] = 0.0
        self._state.capacity[env_ids] = 1.0

    def compute(self, control_action, joint_pos, joint_vel):
        # zero PD gains -> base returns the feed-forward effort unchanged
        action = super().compute(control_action, joint_pos, joint_vel)
        if self.cfg.bio_enabled:
            out, self._state = apply_bio_constraints(action.joint_efforts, self._state, self._bio_cfg)
            action.joint_efforts = out
        return action

@configclass
class BioActuatorCfg(IdealPDActuatorCfg):
    class_type: type = BioActuator
    bio_enabled: bool = True
    alpha: float = 0.2
    effort_limit_value: float = 23.5
    fatigue_rate: float = 0.05
    recovery_rate: float = 0.02
    fatigue_onset: float = 0.7
```
(`_joint_indices_shape` / `_device` are placeholders for the real ActuatorBase attributes confirmed in Step 1 — substitute the actual `self.num_envs`, `self.num_joints`, `self._device`.)

- [ ] **Step 3: Commit**

```bash
git add src/torque_loco/bio_actuator.py
git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "feat: BioActuator applying bio constraints over effort passthrough"
```

### Task 2.3: Wire the bio actuator into a cfg variant + train with/without

**Files:** Modify `src/torque_loco/go2_torque_env_cfg.py`, `src/torque_loco/__register__.py`

- [ ] **Step 1: Add a bio cfg subclass**

```python
# append to go2_torque_env_cfg.py
from .bio_actuator import BioActuatorCfg

@configclass
class Go2TorqueBioEnvCfg(Go2TorqueEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["base_legs"] = BioActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
            effort_limit=GO2_EFFORT_LIMIT, velocity_limit=30.0,
            bio_enabled=True, effort_limit_value=GO2_EFFORT_LIMIT,
        )
```
Register it as `Isaac-Velocity-Flat-Go2-Torque-Bio-v0` (mirror Task 1.2 Step 3).

- [ ] **Step 2: Train the bio variant (same seeds/iters as baseline)**

Reuse the Task 1.3 launch command with `--task Isaac-Velocity-Flat-Go2-Torque-Bio-v0`, logging to `results/go2_torque_bio_s1.log`. Pin to an idle GPU.

- [ ] **Step 3: Confirm both variants walk**

Render a clip of each; confirm the bio variant also achieves locomotion (it should — the envelope only bites under stress). Note any reward delta with the spec's framing: a reward cost is the *expected* sign of a working constraint, not failure.

- [ ] **Step 4: Commit logs/clips**

```bash
git add results && git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "results: bio vs no-bio Go2 torque policies trained"
```

---

## Phase 3 — Feasibility metrics + the envelope comparison (D4) · Tier 2 payoff

### Task 3.1: Metric reducers (TDD, sim-free)

**Files:** Create `src/torque_loco/metrics.py`, Test `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run, verify fail.** `PYTHONPATH=src pytest tests/test_metrics.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run, verify pass.** → 3 passed.

- [ ] **Step 5: Commit.** `git add src/torque_loco/metrics.py tests/test_metrics.py && git -c user.email=... commit -m "feat: feasibility metrics (peak torque, jerk, energy) + tests"`

### Task 3.2: Roll out checkpoints and dump per-step torque/vel

**Files:** Create `scripts/eval_metrics.py`

- [ ] **Step 1: Write the rollout dumper**

A script that loads a checkpoint, runs N episodes headless, and writes a CSV of per-step joint torque + joint velocity. Reuse Isaac Lab's `play.py` structure (load env + policy) but add a logging hook on `env.step` capturing `robot.data.applied_torque` and `robot.data.joint_vel`. Confirm those data field names in Step 1's API check; record them.

```bash
# scripts/eval_metrics.py — invoked via Isaac Lab's python:
#   ./isaaclab.sh -p ~/workspace/isaaclab-torque-locomotion/scripts/eval_metrics.py \
#     --task Isaac-Velocity-Flat-Go2-Torque-v0 --load_run <run> --episodes 32 \
#     --out results/metrics_nobio.csv
```
(Full script body written during execution against the confirmed `play.py` API — flagged as a build step, not a placeholder claim.)

- [ ] **Step 2: Dump both variants**

Run for `Go2-Torque-v0` → `results/metrics_nobio.csv` and `Go2-Torque-Bio-v0` → `results/metrics_bio.csv`.

- [ ] **Step 3: Compute + plot the comparison**

A small `scripts/plot_envelope.py` that loads both CSVs, applies `metrics.py`, and bar-plots peak torque / jerk / energy (bio vs no-bio), with the Go2 23.5 sim-clip and 45 N·m real limit drawn as reference lines. Expected story (matching SATA Phase 3): bio keeps peak torque at/under the envelope; no-bio drifts higher.

- [ ] **Step 4: Commit results**

```bash
git add results scripts && git -c user.email=han.li@chainsea.com.tw -c user.name=han commit -m "results: Tier-2 peak-torque comparison (bio vs no-bio) measured in Isaac Lab"
```

**✅ Tier 2: the bio layers' peak-torque behaviour observed in the prior repo also appears here (sim-only, preliminary).**

---

## Phase 4 — Course deck (D5)

### Task 4.1: Assemble the report

- [ ] **Step 1:** Adapt the SATA repo's `final-presentation.html` aesthetic into a short deck: (1) why migrate (Isaac Gym deprecated), (2) torque-control paradigm in Isaac Lab, (3) Tier-1 walking clip, (4) Tier-2 envelope bar chart, (5) honest limits (sim-only, low-seed, cross-engine reward not comparable) + next phase (full bio stack, zero-shot transfer, new robot).
- [ ] **Step 2:** Commit the deck under `docs/`. Push the repo to GitHub (create the remote) once the user approves making it public.

---

## Phase 5 — STRETCH (gated; re-plan after Tier 2 lands)

These tiers are **outlined, not fully detailed** — their exact steps depend on Tier-2 outcomes and on R2 discovery. When you reach them, run a short `writing-plans` pass to detail the chosen one. Detailing them now would be fabrication.

### Tier 3 — Cross-engine zero-shot transfer (the headline)
- Load an existing **Isaac-Gym SATA reference checkpoint** (rsl_rl ActorCritic) into an Isaac Lab eval env.
- **R2 work:** write an observation adapter mapping Isaac Lab's manager-based obs vector to the exact order/scaling the SATA policy expects (read SATA's `LeggedRobotCfg` obs concatenation; unit-test the adapter on a known obs vector). This is the single riskiest piece — budget a full day; if the contract can't be matched cleanly, fall back to "native bio-vs-no-bio under a domain shift" instead.
- Run zero-shot; measure engine gap (reward proxy, fall rate, peak torque); compare full-bio vs ablated for gap survival.

### Tier 3b — New-robot generalization (cherry)
- Repeat Phase 1–3 with `Isaac-Velocity-Flat-Anymal-C-v0` (effort-control variant). Anymal-C has a learned actuator net by default — replacing it with the effort/bio actuator is the main extra step. Clearly labelled preliminary.

---

## Self-review notes (done by author)

- **Spec coverage:** §4.2 Tier 1 → Phase 1; §4.3 Tier 2 → Phase 2–3; §4.4 Tier 3 → Phase 5; §4.5 Tier 3b → Phase 5; §4.6 seeds → Tasks 1.3/2.2 (resource-gated, stated); §3 risks R1/R2 → Phase 0 gates + Phase 5 note; §8 honesty guardrails → Tasks 1.3 Step 3, 2.3 Step 3, 4.1 Step 1.
- **Placeholder scan:** the only non-literal blocks are explicitly flagged as "verify API against installed build" or "stretch, re-plan" — honest discovery gates, not hidden TODOs. The bio math, metrics, env cfg, and actuator are fully specified.
- **Type consistency:** `apply_bio_constraints(cmd, state, cfg) -> (Tensor, BioState)`, `BioState{activation, capacity}`, `BioCfg{alpha, effort_limit, fatigue_rate, recovery_rate, fatigue_onset}`, `BioActuatorCfg.effort_limit_value` used consistently across Tasks 2.1–2.3. Metric names `peak_torque/action_jerk/mech_energy` consistent across Task 3.1–3.2.
