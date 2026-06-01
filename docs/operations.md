# Operations log — Isaac Lab torque-locomotion migration

Running record of what actually happened during setup/execution. Read this first
if you open a fresh session in this repo.

## Environment (Phase 0) — DONE 2026-06-02

- **conda env `isaaclab`** (Python 3.11) at `~/miniconda3/envs/isaaclab`.
  Isaac Sim **5.1.0**, torch **2.7.0+cu128**. Driver 595.58.03 / CUDA 13.2 (≥12.8 OK).
- **Isaac Lab cloned at `~/workspace/IsaacLab`** (installed via `./isaaclab.sh --install`).
- GPUs: 4×A6000, GPU 0/1/2 idle at setup. Pin with `CUDA_VISIBLE_DEVICES`
  (shared box — only use confirmed-idle GPUs).

### Three install gotchas (all fixed — keep these as run conventions)

1. **conda Terms of Service** blocked `conda create`. One-time fix:
   ```bash
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   ```
2. **cmake 4.x breaks `egl_probe`** (its CMakeLists needs `cmake_minimum_required`
   < 3.5, which CMake 4 refuses). Fix: pin cmake < 4 in the env and set a policy floor.
   ```bash
   conda install -y -n isaaclab -c conda-forge "cmake<4"
   export CMAKE_POLICY_VERSION_MINIMUM=3.5
   ```
3. **Every run needs these or it fails**:
   - `conda activate isaaclab` BEFORE calling `./isaaclab.sh` — otherwise
     `isaaclab.sh -p` picks the wrong python → `ModuleNotFoundError: No module named 'isaaclab'`.
   - `export OMNI_KIT_ACCEPT_EULA=YES` — else Isaac Sim hangs on an interactive
     EULA prompt (EOF in headless).
   - Pin `CUDA_VISIBLE_DEVICES=<idle_gpu>`.

   Pip dependency warnings (`click 8.4.1` vs 8.1.7, `psutil 7.2.2` vs 5.9.8) are
   **non-fatal** — `import isaacsim` / `import isaaclab` both succeed.

### Canonical run preamble
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES CMAKE_POLICY_VERSION_MINIMUM=3.5
cd ~/workspace/IsaacLab
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --num_envs 4096 --max_iterations 1500
```
Logs land in `~/workspace/IsaacLab/logs/rsl_rl/<task>/<date-time>/`.

## Code done so far (sim-free, TDD'd in the `sata` env)
- `src/torque_loco/bio_constraints.py` — activation low-pass + fatigue capacity
  (4 tests, commit `28e1cfb`).
- `src/torque_loco/metrics.py` — peak torque / jerk / energy (3 tests, `e8e55c3`).
- Run tests: `PYTHONPATH=src ~/miniconda3/envs/sata/bin/python -m pytest -q`
  (pure-torch, no Isaac Sim needed).

## Current status (2026-06-02)
- **Tier-0 gate PASSED ✅** — stock `Isaac-Velocity-Flat-Unitree-Go2-v0`
  smoke-train ran headless on GPU 0 (64 envs, 5 iters): rsl_rl printed
  `Learning iteration 2/5…4/5` with updating Mean reward, exit code 0. First
  Isaac Sim launch took ~2 min (shader compile + Go2 USD pull); subsequent
  launches are faster. **Phase 0 fully cleared — Isaac Lab works on this box.**

## Verified API (Task 1.2 pre-work) — DONE 2026-06-02

Read from the installed source. **Corrections to the plan's guesses:**

| Plan guessed | Reality |
|---|---|
| module `...config.unitree_go2` | **`...config.go2`** (dir is `go2`, not `unitree_go2`) ✗ |
| env-cfg class `UnitreeGo2FlatEnvCfg` | ✅ correct |
| actuator group key `"base_legs"` | ✅ correct |
| actuator class `IdealPDActuator` | actually **`DCMotorCfg`** (subclasses `IdealPDActuatorCfg`, so subclassing still works) |

Concrete values:
- **Env cfg:** `isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg:UnitreeGo2FlatEnvCfg`
  (parent `rough_env_cfg.py:UnitreeGo2RoughEnvCfg` ← `velocity_env_cfg.LocomotionVelocityRoughEnvCfg`).
- **Action term:** `self.actions.joint_pos` (a `JointPositionActionCfg`); rough cfg sets `joint_pos.scale = 0.25`.
- **Effort action class:** `mdp.JointEffortActionCfg` (`import isaaclab.envs.mdp as mdp`), in
  `isaaclab/envs/mdp/actions/actions_cfg.py`.
- **Stock actuator** (group `"base_legs"`, in `isaaclab_assets/robots/unitree.py:UNITREE_GO2_CFG`):
  `DCMotorCfg(joint_names_expr=[".*_hip_joint",".*_thigh_joint",".*_calf_joint"], effort_limit=23.5,
  saturation_effort=23.5, velocity_limit=30.0, stiffness=25.0, damping=0.5)`.
  **Note: stock effort_limit 23.5 N·m == SATA's sim torque clip 23.5** — nice alignment for the
  cross-engine reproducibility story.
- **Actuator cfgs in `isaaclab.actuators`:** `ImplicitActuatorCfg`, `IdealPDActuatorCfg`,
  `DCMotorCfg(IdealPDActuatorCfg)`.
- **rsl_rl runner entry point:** `...config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg`.

**Implication for Task 1.2 (effort control):** subclass `UnitreeGo2FlatEnvCfg`; in `__post_init__`
after `super()`, set `self.actions.joint_pos = None`, add
`self.actions.joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=23.5)`,
and replace the `"base_legs"` actuator with an effort-passthrough
`IdealPDActuatorCfg(stiffness=0.0, damping=0.0, effort_limit=23.5, velocity_limit=30.0)`.
For Task 2.2 the `BioActuator` may subclass `IdealPDActuator` (clean passthrough) **or** `DCMotor`
(keeps the realistic torque-speed saturation as part of the envelope) — decide then.

## Task 1.2 — Go2 effort-control env cfg — DONE 2026-06-02

Implemented and smoke-tested. Files (in this repo):
- `src/torque_loco/go2_torque_env_cfg.py` — `Go2TorqueEnvCfg(UnitreeGo2FlatEnvCfg)`:
  `__post_init__` sets `self.actions.joint_pos = None` and adds
  `self.actions.joint_effort = mdp.JointEffortActionCfg(joint_names=[".*"], scale=23.5)`,
  then replaces the `"base_legs"` actuator with a zero-gain
  `IdealPDActuatorCfg(stiffness=0, damping=0, effort_limit=23.5, velocity_limit=30.0)`.
  Also a `Go2TorqueEnvCfg_PLAY` variant (50 envs, no corruption/pushes) for eval/render.
- `src/torque_loco/__register__.py` — registers `Isaac-Velocity-Flat-Go2-Torque-v0`
  (+ `-Play-v0`), reusing the stock `UnitreeGo2FlatPPORunnerCfg`.
- `scripts/train_go2.py` — launcher (see mechanism below).

### Two corrections to the plan (keep these)
1. **Use STRING entry points in `gym.register`, not class objects.** The plan passed
   `Go2TorqueEnvCfg` directly, but importing the cfg eagerly pulls in `isaaclab → pxr`,
   which is only importable *after* the Omniverse app launches. String entry points
   (`"torque_loco.go2_torque_env_cfg:Go2TorqueEnvCfg"`) defer that import to `gym.make`.
2. **Task discovery needs a register-then-delegate launcher (the "working mechanism").**
   Isaac Lab's `train.py` resolves the task cfg via a hydra decorator at *module import*,
   and the gym registry is process-global. So `scripts/train_go2.py`: adds `src/` to
   `sys.path` → `import torque_loco.__register__` (pure `gym.register`, no pxr, safe
   pre-launch) → `runpy.run_path(<IsaacLab>/scripts/.../rsl_rl/train.py, "__main__")`
   (its dir prepended so `import cli_args` resolves). CLI args flow through `sys.argv`.
   IsaacLab left unmodified. Override clone path with `ISAACLAB_PATH` env var if needed.

   Run preamble (GPU 0, all 4 idle at the time):
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
   export OMNI_KIT_ACCEPT_EULA=YES CMAKE_POLICY_VERSION_MINIMUM=3.5
   cd ~/workspace/IsaacLab
   CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
     ~/workspace/isaaclab-torque-locomotion/scripts/train_go2.py \
     --task Isaac-Velocity-Flat-Go2-Torque-v0 --headless --num_envs 1024 --max_iterations 10
   ```

### API confirmations (vs the pre-work guesses)
- `ActionManager._prepare_terms` iterates `self.cfg.__dict__.items()` and `continue`s on
  `None` → nulling `joint_pos` drops it, a dynamically-added `joint_effort` is picked up. ✓
- No reward term references the position action specifically; `action_rate_l2` /
  `dof_torques_l2` operate on the 12-dim action / applied torque generically. ✓

### Smoke result (Step 4) ✅
1024 envs, 10 iters, GPU 0, **exit 0**. Mean reward finite throughout (-2.8 → -3.6 …
ending ~-2.9, **no NaN, no explosion**); `error_vel_xy` even drifted down 0.018→0.024.
`scale=23.5` is stable — no need to halve. Log: `results/go2_torque_smoke.log`.
(Reward is negative because an untrained pure-torque policy falls — expected; the gate
is "finite, no shape errors", which passed.)

## Next steps (resume here)
1. **Task 1.3** — full training run(s) to walking (`--num_envs 4096 --max_iterations 1500`,
   detached), render a `play.py` clip to confirm forward locomotion, record final mean
   reward. Note: cross-engine reward magnitudes vs SATA are NOT comparable (anti-over-claim).
2. Continue plan tasks 2.2 → 2.3 → 3.2 (Tier 2: BioActuator + envelope comparison).
