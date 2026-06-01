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

## Task 1.3 — overnight multi-seed baseline (IN PROGRESS, launched 2026-06-02 ~02:00)

Running 8 seeds of the no-bio torque baseline (`Isaac-Velocity-Flat-Go2-Torque-v0`,
4096 envs, 1500 iters) across all 4 idle A6000s, mirroring SATA's multi-seed rigor
(SATA used 3, later 8). Seed 1 launched standalone on GPU 0; `scripts/dispatch_seeds.sh`
runs the rest sequentially per GPU, waiting for each GPU to free before starting:
- GPU 0: seed 1 (standalone) → 5 ; GPU 1: 2 → 6 ; GPU 2: 3 → 7 ; GPU 3: 4 → 8.
- Per-seed logs `results/go2_torque_s<N>.log`; dispatcher logs `results/dispatch_gpu<G>.log`.
- Dispatcher is idempotent (skips a seed whose log shows completion) and reusable for
  the bio variant later: `dispatch_seeds.sh <gpu> <task_id> <log_prefix> <seed...>`.
- Early health (seed 1): reward -3.0 → -0.66 by iter ~150, no NaN. Walking-quality
  check + `play.py` clip still TODO once a seed finishes.
- **Anti-over-claim:** cross-engine reward magnitudes vs SATA are NOT comparable
  (different terms/engine); validation is qualitative (does it walk + track command).

## SATA ground truth (read from sibling repos 2026-06-02) — RECONCILE BEFORE Task 2

Read `~/workspace/SATA` (original) + `~/workspace/bio-inspired-adaptive-locomotion`
(reproduction). **Our `bio_constraints.py` is a plausible but DIFFERENT formulation
from SATA's actual mechanism — must decide whether to realign for faithful Tier-2.**

SATA's real `go2_torque` (`SATA/legged_gym/.../go2/go2_torque/go2_torque.py`):
- **control_type `'TG'`** (torque + growth curriculum), `action_scale=5`, `decimation=1`
  (200 Hz, sim dt 0.005). No PD in torque mode.
- **Activation low-pass on the SIGN, not the torque:** `act = (tanh(τ/τlim) - act)*0.6 + act`
  (α=0.6). i.e. it filters `tanh(τ/limit)`, then torque = `act * τlim`.
- **Hill model** force-velocity shaping: `τ = act * τlim * (1 - sign(act)*ω/ω_max)`.
- **Fatigue = accumulator used as a REWARD PENALTY, not a hard clip:**
  `motor_fatigue += |τ|*dt; motor_fatigue *= 0.9`; reward term
  `_reward_motor_fatigue = Σ(motor_fatigue * |τ_action|)`, weight **-0.05**. ← our
  version instead CLIPS torque by a capacity in (0,1]; SATA never clips via fatigue.
- **Torque clip 23.5 N·m** (`go2_torque.py:315`); real Go2 peak **45 N·m** (`plot_phase3.py`).
- **Obs = 60** = lin_vel*2 + ang_vel*0.25 + proj_grav + (dof_pos-default)*1 + dof_vel*0.05
  + cmd*[2,2,0.25] + **torques (raw) + motor_fatigue (raw)**. (Adds torque+fatigue to obs.)
- Reward scales: forward 10, head_height 5, moving_y 5, moving_yaw 5, soft_dof_pos_limits
  -5, motor_fatigue -0.05, dof_acc -1e-6, roll -5, lin_vel_z -5. (No action-rate term —
  smoothness comes from the activation filter.)

Phase-3 headline (reproduction `results/phase3-bio-claims.../README.md`), what Tier-2
must reproduce qualitatively: reference peak τ 22.5±0.3; **no_activation → 42.5±4.4**
(jerk collapses); **no_fatigue → energy 1.69 (2.5×), jerk 26,998 (35×)**. Phase-1
reference reward **104±16 (8 seeds)** / 114±6 (3 seeds). Core claim: bio = **feasibility
envelope, not a reward device** (see [[feedback-research-rigor]]).

**Open decision for Task 2.2:** keep our clean hard-clip capacity model (simpler, sim-free
testable, "envelope" reading is literal) OR realign to SATA's penalty+Hill+sign-filter
formulation (faithful reproduction, but fatigue becomes a reward term not an actuator clip,
which muddies the "actuator envelope" framing and needs obs to carry torque+fatigue).
Decide with the user before implementing the BioActuator.

> **SUPERSEDED 2026-06-02:** scope pivoted to FULL faithful SATA (incl. growth curriculum).
> The Task-1.3 no-bio baseline above was abandoned (misaligned with SATA — stock IsaacLab
> rewards/linear scale); those seeds were killed. The bio formulation question was resolved
> by the user: realign fully to SATA. See the spec/plan below. Branch: `feat/sata-faithful-migration`.

## Faithful SATA migration — execution log (2026-06-02, branch feat/sata-faithful-migration)

Spec `docs/superpowers/specs/2026-06-02-sata-faithful-migration-design.md`; plan
`docs/superpowers/plans/2026-06-02-sata-faithful-migration.md`. Subagent-driven execution.

**Phase A (sim-free TDD, `sata` env) — DONE.** 19 tests pass.
- `bio_constraints.py` rewritten: `apply_bio` = tanh-EMA activation (γ=0.6) + Hill + fatigue
  leaky-integrator (β=0.9); frozen `BioState`. (replaces the old capacity-clip model)
- `growth.py`: Gompertz `gompertz(step)` (k=3e-5, x0=24000) + torque/freq schedules.
- `metrics.py`: + SATA-aligned reducers `sata_peak_torque / sata_energy_per_step / sata_mean_jerk`
  (first-diff jerk, matching the reproduction's `eval_under_conditions.py`).

**Phase B (Isaac Lab integration) — DONE + smoke-passed + reviewed.**
- `bio_actuator.py` `BioActuator(IdealPDActuator)`: tanh-EMA+Hill+fatigue, front-leg torque
  ceiling grows 7.05→23.5 via `env._G` (rear constant 23.5), per-joint vel limit (hip/thigh
  30.1, calf 20.07), no hard clip (envelope = tanh). Init-time fatigue seeds from `U(0,0.2·G(0))`.
- `sata_mdp.py`: obs `applied_torque`+`motor_fatigue`; 9 SATA rewards (G-modulated Table II:
  forward-target blend, moving_y/yaw ×G, base_height ×(1+G)); G-scaled push event.
- `go2_sata_env.py` `Go2SataEnv(ManagerBasedRLEnv)`: **step() override** mirrors stock
  `ManagerBasedRLEnv.step()` (lines 153-241) with the fixed decimation loop replaced by SATA's
  variable-frequency accumulator (`while accum*freq<1`, freq=100→200 via G); `env._G` set each
  step. Cfg: BioActuator, SATA obs(60)/rewards/commands(fixed ranges)/DR/defaults (base z=0.10,
  thigh 1.45, calf -2.5), 200 Hz physics. Task `Isaac-Velocity-Flat-Go2-Sata-v0` (+Play).

### Three bugs the smoke-train caught (platform-migration findings — keep these)
1. **Double dt-scaling.** Isaac Lab `RewardManager.compute` already does `func*weight*dt`
   (`reward_manager.py:150`). Our cfg weights were `scale*dt` → rewards dt²-scaled (~1e-4,
   "Mean reward -0.00"). Fix: weights = **raw SATA scales** (10,5,-5,…); the manager applies dt.
2. **base_contact termination kills the prone start.** Robot starts prone (z=0.10); the stock
   `base_contact` (illegal trunk contact) terminated at ~11 steps. SATA terminates on flip-over /
   joint-limit, NOT base contact (paper §IV-B). Fix: drop `base_contact`, add `bad_orientation`
   (limit_angle 1.4).
3. **joint_pos_out_of_limit uses SOFT (0.9-scaled) limits.** SATA's folded start (calf -2.5)
   sits at the soft-limit edge → instant joint_limit termination (~2 steps). Isaac Lab's term
   tests soft limits, not hard. Fix: drop the joint-limit termination; rely on the
   `soft_dof_pos_limits` REWARD penalty (-5) + flip-over. (platform diff vs SATA's hard-limit reset)

**Smoke result (post-fix, 30 iters, 1024 envs, GPU 1):** exit 0, no NaN; episode length climbs
10→590 (learns to survive from prone), all `time_out`/`bad_orientation 0`; reward breakdown sane
(track_x +1.82, base_height +0.16; net negative dominated by joint_acc −1.32 early — expected,
G still ~0.15). 22k steps/s. Logs `results/sata_smoke{,2,3}.log`.

## Task C1 — 8-seed reference training (IN PROGRESS, launched 2026-06-02 ~04:13)
`Isaac-Velocity-Flat-Go2-Sata-v0`, 4096 envs, **3000 iters, 8 seeds** (user: ≥8 for mean±std
matching SATA's 8-seed 104±16). GPU 0 busy (other tenant) → GPUs 1/2/3 only:
GPU1 seeds 1,4,7 ; GPU2 2,5,8 ; GPU3 3,6 (sequential via `dispatch_seeds.sh`, `MAX_ITER=3000`).
Logs `results/go2_sata_s<N>.log`, dispatcher `results/dispatch_sata_gpu<G>.log`.

## Next steps (resume here)
1. **Finish C1** — wait for 8 seeds; record final mean±std reward; render a
   `...-Go2-Sata-Play-v0` clip; confirm walking + velocity tracking.
2. **Task C2** — `scripts/eval_metrics.py` rollout dump → `scripts/plot_envelope.py`; compare
   peak τ / energy / jerk to ground truth (reference peak τ≈22.5 inside 23.5/45 envelope). Record
   the cross-engine envelope table. Anti-over-claim: reward not comparable cross-engine; envelope is.
3. Final code review → `finishing-a-development-branch` (merge `feat/sata-faithful-migration`→main).

## Task B3 — Go2SataEnv (variable-freq step override) + full SATA env cfg

Created `src/torque_loco/go2_sata_env.py`.

**step() source mirrored:** `Go2SataEnv._stepped` reproduces the body of
`isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv.step`
(IsaacLab `source/isaaclab/isaaclab/envs/manager_based_rl_env.py`, lines 153-240) verbatim,
with the SINGLE change `for _ in range(self.cfg.decimation):` -> `for _ in range(n_sub):`.
All `recorder_manager` calls, `_sim_step_counter` increment, render gating, counter increments,
termination/reward/reset/command/interval-event flow and the 5-tuple return are preserved.
Constructor mirrors `ManagerBasedRLEnv.__init__(self, cfg, render_mode=None, **kwargs)`
(same file line 65); `common_step_counter` is initialised to 0 there and incremented once per
`step()` after the physics loop (line 202) — `step()` reads it BEFORE incrementing, so the
Gompertz scalar uses the pre-increment count.

**Render gating choice:** kept stock `self._sim_step_counter % self.cfg.sim.render_interval == 0`.
With `decimation=1` / `render_interval=4` it renders every 4th physics step regardless of how
many sub-steps a given control step ran, which is correct; under headless training `is_rendering`
is False so the whole branch is skipped.

**API verified against installed source (no fixes needed beyond skeleton):**
- `UnitreeGo2FlatEnvCfg`: `.../config/go2/flat_env_cfg.py` (inherits `rough_env_cfg.py`,
  inherits `velocity_env_cfg.py::LocomotionVelocityRoughEnvCfg`).
- `mdp.JointEffortActionCfg`: defined `isaaclab/envs/mdp/actions/actions_cfg.py:95`, re-exported
  via `actions/__init__.py` -> `isaaclab.envs.mdp`.
- `ObservationTermCfg.scale`: `isaaclab/managers/manager_term_cfg.py:176` (field exists).
- obs terms `base_lin_vel/base_ang_vel/joint_pos/joint_vel/actions/height_scan`: all present in
  `velocity_env_cfg.py::ObservationsCfg.PolicyCfg`.
- actuator key `base_legs`: `isaaclab_assets/robots/unitree.py:170` (DCMotorCfg). Replaced with
  BioActuatorCfg(joint_names_expr=[".*"]) — Go2 only has the 12 leg joints so `.*` is equivalent
  to the stock `.*_hip/_thigh/_calf` exprs.
- events `push_robot` (interval) + `add_base_mass` (startup): present in
  `velocity_env_cfg.py::EventCfg`. NOTE: go2 `rough_env_cfg.py:34` already sets
  `push_robot = None`; we re-create it as the growth-scaled push. `add_base_mass` mass range
  overridden (-1,5).
- `num_rerenders_on_reset`, `self.extras`, `recorder_manager`, `step_dt`, `physics_dt`: all
  confirmed in `manager_based_env.py` / cfg.

**Reward clearing:** `for name in list(vars(R)): setattr(R, name, None)` — `@configclass`
instances expose terms as instance attrs, so this drops all stock terms before adding SATA terms.

py_compile: PASS. Runtime smoke = Task B4.
