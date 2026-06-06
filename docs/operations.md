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
  **Note: stock effort_limit 23.5 N·m == SATA's 23.5 sim torque clip** — a convenient match,
  one less variable to reconcile.
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
(jerk rises sharply); **no_fatigue → energy 1.69 (2.5×), jerk 26,998 (35×)**. Phase-1
reference reward **103.6±16.0 (the SATA repo's rounded headline 104±16, recomputed here from tfevents) (8 seeds)** / 114±6 (3 seeds). (The prior repo read the
bio layers as feasibility constraints rather than reward devices — its own
interpretation, not load-bearing here; this repo is about cross-engine fidelity.
See [[feedback-research-rigor]].)

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

**Concurrency race found (keep this gotcha):** Isaac Lab names each run dir by
`datetime.now().strftime("%Y-%m-%d_%H-%M-%S")` (timestamp-to-the-second). When two
concurrently-launched seeds initialize rsl_rl in the SAME wall-clock second, they pick the
same dir and the loser dies with `FileExistsError: .../<ts>/git/IsaacLab.diff`. Seed 6 hit this
(exit 1) at 05:34:41. **Fix:** `dispatch_seeds.sh` now passes `--run_name <prefix>_s<seed>`
(train.py appends it → `<ts>_<run_name>`, unique per seed). Seed 6 re-run on GPU3 with the fix.
Self-healing: any seed that loses the race is re-run solo (no concurrent init → no collision).
First-batch result: seeds 1,2,3 trained (final reward 75 / 24 / 71 — seed 2 a laggard, expected;
reward NOT comparable cross-engine, envelope metrics are the claim).

**WORSE than one crash — silent checkpoint corruption (keep this):** seeds **1,2,3** launched
simultaneously (04:13:10) ALL initialized at the same second and got the SAME run dir
`2026-06-02_04-14-08` (logs confirm all three print that dir). They did NOT crash — they wrote
checkpoints into the SAME directory, last-writer-wins, so that dir holds only ONE usable final
policy, not three. (The later collisions of seeds 6 and 8 instead crashed on `git/IsaacLab.diff`.)
So a same-second race can either crash a seed OR silently merge checkpoints. **Recovery:** void
`04-14-08`; re-run seeds 1,2,3 with the `--run_name`-fixed dispatcher (distinct dirs). Old per-seed
reward logs preserved as `results/go2_sata_s{1,2,3}.corrupt.log`. Clean 8-seed set = re-run
{1,2,3} + {4,5,6,7,8} (each a distinct `<ts>[_go2_sata_sN]` dir). Lesson: always pass a unique
`--run_name` for ANY concurrent rsl_rl launches — the timestamp dir name is not collision-safe.

**eval_metrics.py bug (fixed):** it redefined `--load_run`/`--checkpoint` AND called
`cli_args.add_rsl_rl_args` → argparse conflict. Removed the dupes; select a checkpoint via
`--load_run <dir>` (latest model_*.pt auto-picked); `--checkpoint` expects a full path. (28e9bf1)

## Task C1/C2 — RESULT (8 clean seeds, 2026-06-02 ~09:45)

All 8 seeds trained 3000 iters (clean, distinct dirs after the race fix). Walking confirmed
via training telemetry: every seed reaches full-length episodes (2000 steps, ~all time_out,
flip-over≈0) — the robot rises from the prone start and tracks the velocity command. Final
training rewards 24–75 (seed 2 reproducibly the laggard at 24; reward NOT comparable
cross-engine — SATA's 104±16 is a different engine). Eval: `scripts/eval_metrics.py` rolled out
each seed (32 envs ×1000 steps, headless) → `results/metrics_sata_s<N>.csv`;
`scripts/aggregate_envelope.py` → `results/envelope_summary.{csv,png}`.

**Cross-engine feasibility-envelope reproduction — SUPERSEDED (v1, flat terrain).**
> These peak-torque numbers are from the early v1 flat-terrain run and are **no longer the headline**.
> They are kept for history. The finalised result is the 8-seed SATA-rough reward reproduction +
> per-step decomposition in `REPRODUCTION_NOTES.md` (Isaac Lab 76.8 ± 16.2 vs Gym 103.6 ± 16.0). A
> G=1 peak-torque envelope eval on the rough terrain was not re-run (the deck/report lead on reward).

| metric | Isaac Lab (8-seed full SATA, v1 FLAT — superseded) | SATA Isaac-Gym ref |
|---|---|---|
| Peak \|torque\| | **23.86 ± 2.45 N·m** | 22.5 ± 0.3 |
| Energy / step | 1.82 ± 0.37 J | (reference band) |
| Action jerk | 1961 ± 478 | (reference band) |

All 8 seeds keep peak joint torque **inside the 45 N·m hardware envelope** (max 28.3),
matching SATA's ~22.5 reference. Wider spread than SATA's ±0.3 traces to two less-converged
seeds (s2 28.3, s3 26.1); the other six cluster ~22.75 ± 1.4. **→ the v1 flat-terrain peak-torque
numbers were consistent with the prior repo's peak-torque observation (envelope framing is the
prior repo's own interpretation, not load-bearing here; and our OWN envelope reproduction was
superseded by the reward result as THIS repo's headline; see the rough-terrain reward result).** Anti-over-claim: sim-only;
"within envelope" = within the rated torque number, not hardware-validated; reward magnitudes
are not cross-engine-comparable (envelope metrics are).

### Eval gotchas (keep)
- Eval MUST pass `--headless` — without it the RTX renderer segfaults on this display-less box
  (`librtx.scenedb` crash). Metrics need no rendering.
- Raw per-step CSVs + training/eval logs are gitignored (regenerable, bulky); the committed
  artifacts are `envelope_summary.{csv,png}` + per-seed `envelope_s*.png`.

## Remaining / next steps
1. **Walking clip** (deck nice-to-have, NOT blocking): `play.py --video` needs the VNC/EGL render
   path (headless RTX segfaults). Render via the Phase-0 VNC display when assembling the deck.
2. Final code review → `finishing-a-development-branch` (merge `feat/sata-faithful-migration`→main).
3. Deck (Phase 4 of the original plan): why-migrate, torque paradigm, envelope reproduction chart.

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

## Post-merge fidelity investigation (2026-06-02) — the migrated policy LOW-CRAWLED, not walked

The merged reproduction trained + "reproduced" the envelope number, but the rendered clip showed a
low forward crawl (base height ~0.10 m), not SATA's upright walk. User (who reproduced SATA cleanly
in Isaac Gym) correctly pushed: a faithful PORT should walk → we hadn't controlled the variables.
Systematic audit (one variable at a time):

1. **Deploy capacity G** — eval ran at G≈0.13 (infant body: front τ 9 N·m, 100 Hz) instead of SATA's
   "restore full capacity at deployment" (paper §IV-B). Fixed: `Go2SataEnvCfg_PLAY.growth_deploy_scale=1.0`.
   Real bug, but didn't fix the crawl.
2. **PPO network** — Isaac Lab's *Flat* runner shrinks the net to [128,128,128]; SATA uses [512,256,128].
   Fixed: point Sata task at Isaac Lab's *Rough* runner (= SATA's net + matched hyperparams; no
   invented values). Retrained 3000 iters → STILL crawled (base_z 0.106). Disconfirmed as the cause.
3. **Robot model** — deep-checked Isaac Lab Go2 USD vs SATA URDF: total mass 15.019 kg / base 6.921 kg
   IDENTICAL, hip/calf limits + default pose match; only deltas = thigh pos-limit range + calf vel-limit
   (minor). NOT the confound.
4. **ROOT CAUSE — reward fidelity (forward reward 2× over-scaled).** Read SATA's reference training
   tfevents (`logs/SATA/May25_*_ref_s1`): SATA balances `forward 8.2` with `head_height 2.6`. Ours had
   `forward 10.7` (higher) + `head_height 0.8` (3.3× lower) → policy over-collected forward by crawling.
   Cause: `track_x` implemented the PAPER Table-II two-term form `φ(vx−mid)(1−G)+φ(vx−cmd)(1+G)`
   (ceiling ~2.0), but SATA's CODE `_reward_forward` is a SINGLE exp of a blended target (ceiling 1.0).
   2× over-scaled forward dominated head_height → low crawl. **Fix (c0d89a1): match SATA code** —
   single-exp blended target; also aligned `base_height` head_up to SATA's exact `-(grav_x.clip(min=m))`.
   **Trust the code, not the paper's simplified Table.**

**CONFIRMED:** reward-fixed, at iter 1000 the robot rises prone→~0.20 m and walks forward (was 0.106
crawl); reward balance now matches SATA (forward 8.5, head_height 2.0, dof_acc −1.05, mean reward 102).
Full 8-seed reward-fixed retrain launched. The low-crawl `results/go2_sata_FAILURE_lowcrawl.gif` is kept
as the documented "before/failure" artifact for the deck. Net & G fixes were correct fidelity
improvements (kept), just not the gait cause.

**Lessons (keep):** (a) an envelope NUMBER reproducing ≠ faithful reproduction — check the actual gait;
(b) for faithful repro, control EVERY variable vs the reference + verify via the reference's own
training logs (per-term rewards), not just final behavior; (c) when paper Table ≠ code, the CODE is
authoritative (it produced the working policy).

## Render gotcha (keep): Isaac Gym replay needs flip_visual_attachments=True
The kinematic-replay clip first rendered the Go2 "disassembled" (mis-attached limb meshes) even
though the joint STATES were correct. Cause: `render_replay_isaacgym.py` loaded the URDF with raw
AssetOptions; the Unitree .dae meshes are y-up and MUST be flipped to z-up. Fix: set
`asset_opts.flip_visual_attachments = True` (+ `replace_cylinder_with_capsule = True`) to match
legged_gym's Go2 load. This is a RENDER-only bug — the trained policy was always fine (base 0.30 m
confirmed numerically). Deck clips re-rendered: go2_sata_FAILURE_lowcrawl.gif (crawl 0.10 m) /
go2_sata_FIXED_walk.gif (upright walk 0.30 m).

---

## 2026-06-03 — Rough-terrain control-variable fix, NaN crash fix, 4-agent faithfulness audit + maximal-fidelity rework

**Context:** user noticed the render's ground was flat and flagged that SATA's reproduction trains
on ROUGH terrain, not flat. This kicked off a terrain fix → a NaN-crash debug → a full faithfulness
audit → a maximal-fidelity rework. All training runs to date (flat + first rough attempts) are
SUPERSEDED; the authoritative runs are the 16 launched 2026-06-03 02:28.

### A. Terrain control-variable fix (commit 7e728f2)
SATA `go2_torque_config.py`: `mesh_type='trimesh'`, `terrain_proportions=[0.2 smooth slope, 0.8
rough slope, 0,0,0]`, `curriculum=False`, `measure_heights=True`. Our migration had inherited
`UnitreeGo2FlatEnvCfg` → FLAT, which UNDER-estimates the torque envelope (terrain-dependent). Fix:
`sata_terrain.py` (smooth slope + custom `rough_slope` = gentle pyramid slope + ±0.06 m uniform
noise in one cell, mirroring legged_gym), terrain-relative `base_height` via the height scanner,
`Go2SataRoughEnvCfg` (variant A = SATA slopes) + `Go2SataDefaultRoughEnvCfg` (variant B = Isaac Lab
default rough). `verify_terrain.py` confirmed A = gentle slopes+roughness (std .083/.024 m,
curriculum off), B = steep (std .598 m), flat = no scanner. obs was already SATA-faithful.

### B. NaN-divergence crash (commit 75aa16c)
All SATA-rough seeds crashed in a tight band (iter 619–885): `RuntimeError: normal expects all
elements of std >= 0.0` (PPO action std → NaN). Reward was healthy (~38) until a SUDDEN NaN → a
rare non-finite injected into one rollout (clusters at the iteration where the policy first walks
well enough to traverse the terrain). Two differences from SATA, both fixed to MATCH SATA:
1. height-scanner RayCaster returns ±inf for rays missing the mesh → -inf `base_height` reward →
   NaN. SATA's `_get_heights` is always finite. Fix: average only finite rays (fall back to root-z).
2. SATA clips obs to ±`clip_observations`(=100) every step; ours were unclipped. Fix: clip ±100 on
   every policy obs term.
SATA does NOT clip the Hill torque (raw → `set_dof_actuation_force_tensor`), so our no-clip
actuator is correct. CONFIRMED: canary SR1 reward 86.9, cleared the crash band, 0 crashes.

### C. 4-agent faithfulness audit (paper + SATA legged_gym + repro repo)
PPO hyperparams byte-for-byte match; robot mass identical (15.019 kg); all 9 reward terms, bio
activation/Hill/fatigue, Gompertz growth, obs composition+scales+clip ±100, command ranges, init
state, terrain type/proportions all MATCH. Found 1 real bug + several gaps.

### D. Maximal-fidelity rework (commit 712343b) — verified against SATA source, smoke-clean, 19 tests pass
- **REAL BUG — rear-leg torque growth:** SATA (`go2_torque.py:315,223-224`) sets `torque_limits=23.5`
  for ALL joints × `current_torque_limit_scale`(0.3→1.0); the `r_leg_scaled` rear knob is
  start=max=1.0 (a DISABLED no-op). So SATA rear legs grow 7.05→23.5 IDENTICALLY to front — NOT
  constant 23.5 as we (and the docs) had assumed. Old code pinned rear at 23.5 → rear over-powered
  early, biasing the growth curriculum + envelope. Fixed: all 12 joints grow; removed front mask.
- **Per-substep cadence:** SATA's `step_count` / reward / episode clock increment per PHYSICS
  SUBSTEP (inside the variable-frequency loop). n_sub=2 for the WHOLE growth phase (drops to 1 only
  at G=1/200 Hz), so our per-env-step counters made G ~2× too slow, reward ~½, episodes ~2× long.
  Fixed in `go2_sata_env.step`: growth counter `+= n_sub`; reward/command/event `dt = n_sub*0.005`;
  `episode_length_buf += n_sub` (→ correct 10 s sim-time horizon + reward magnitude).
- heading_command=False + rel_standing_envs=0 (SATA samples ang_vel_yaw directly, no standing envs).
- `GrowthVelocityCommand` (sata_mdp): command ranges scale by G — lin_vel_x narrows to its midpoint
  early (full by G=0.5), vy/yaw → 0 early (`go2_torque.py:337-357`).
- friction randomization U[0.5,1.25] (was fixed 0.8/0.6); reset joints = default×U(0.95,1.05);
  reset base shifted ±1 m xy (no yaw/vel rand); base_com randomization x±0.2, y/z±0.1 + base mass
  U[-1,5].
- hard-limit termination ±0.05 + soft_dof_pos_limits reward BOTH use SATA's `go2_torque.urdf`
  limits (front thigh [0,1.5] / rear [0,2.0]; calf [-2.7227,-0.838]), not the wider Isaac Lab USD.
- terrain `difficulty_range=(0.5,0.9)` to match SATA's discrete {0.5,0.75,0.9} (was uniform [0,1)).
- DEFERRED (documented, minor/regularization): `loss_rate=0.1` obs/action dropout; per-link mass
  jitter ±5/16 (base mass + COM are done); USD thigh physical stop not clamped (the SATA-limit
  reward + hard-limit termination handle the joint range instead).

### E. Authoritative runs
16 runs launched 2026-06-03 02:28 on GPU0/1/2/3 (4 seeds/GPU, ~8 h):
`go2_sr_s1-8` (Isaac-Velocity-Rough-Go2-Sata-v0, SATA slopes) + `go2_dr_s1-8`
(Isaac-Velocity-Rough-Go2-Sata-Default-v0, Isaac Lab default rough). 3000 it / 4096 envs. Flat
seeds 1–4 (`go2_sata_fix2_s1-4`, model_2999) kept as the flat comparison point.

PENDING: envelope eval @G=1 (Play, growth_deploy_scale=1.0) on both rough terrains — these are the
authoritative Tier-2 numbers; ALL prior flat numbers (incl. 22.75 / 23.86) are invalid. Then the deck.

## Native Isaac Lab (Isaac Sim 5.1) render — investigated 2026-06-06, BLOCKED by a driver bug

Goal: record a native RTX clip of a trained policy (`scripts/play_go2.py --video`) instead of the
Isaac-Gym kinematic replay. Worked through several layers (each a real fix, kept here so the next
person doesn't re-walk them), then hit a hard driver-side wall.

Fixed along the way (these ARE correct and worth keeping):
1. **Do NOT set `CUDA_VISIBLE_DEVICES`** for Isaac Sim. It desyncs CUDA vs Omniverse device
   enumeration → "CUDA being in bad state" → all GPUs skipped → early `librtx.scenedb` segfault.
   Pin the GPU the Isaac-native way instead: `--device cuda:0` (AppLauncher sets `active_gpu`).
2. **`--kit_args` needs `=` syntax** when the value starts with `--`:
   `--kit_args="--/renderer/multiGpu/enabled=false"` (disables the IOMMU P2P multi-GPU path).
3. **Render headless + offscreen camera** (`--headless --enable_cameras`), NOT a GUI window on VNC —
   the windowed path crashes building the Kit toolbar (`_rebuild_toolbar`), unrelated to the robot.
4. **Vulkan ICD was missing from the system path** but the NVIDIA Vulkan libs ARE installed
   (`libGLX_nvidia.so.0`, `libnvoptix`, `libnvidia-rtcore`, …, matching kernel driver 595.58.03).
   Created `~/.local/share/vulkan/icd.d/nvidia_icd.json` → `libGLX_nvidia.so.0`. Verified working:
   `VK_ICD_FILENAMES=… vulkaninfo --summary` lists all 4 A6000s, Vulkan 1.4.329, RT extensions
   present (`VK_KHR_ray_tracing_pipeline`/`acceleration_structure`/`ray_query`), exit 0, on both the
   conda loader and Isaac's bundled `libvulkan.so.1.3.239`. **So the GPU + Vulkan + ray-tracing API
   all work.** (`vulkan-tools` installed via conda for this diagnosis.)

THE WALL (not fixable without root): with all of the above, the render gets fully into RTX setup
(RtxRenderContext, MDL, neuraylib) and then **segfaults in `rtx.scenedb.plugin` at
`carbOnPluginStartup`** when the Hydra RTX engine is created. This is a **known incompatibility
between NVIDIA driver 595.x and Isaac Sim's RTX scene-DB plugin** (`TLAS limit: valid true, within:
false`), reported across Isaac Sim 4.5 / 5.0 / 5.1. Isaac Sim itself logs that driver 595.58 is
above its tested range (recommended 535.161.07; "latest may work but is not fully tested"). Ruled
out as causes by direct test: scene size / content (crashes identically at `--num_envs 1` AND with
an *empty* scene — `scripts/tutorials/00_sim/create_empty.py --enable_cameras` — so it is not our
robot/terrain), shader cache (cleared `~/.cache/ov` etc., no change), multi-GPU (renderer
`multiGpu/enabled=false` + `maxGpuCount=1` + `activeGpu=0`; gpu.foundation still runs P2P but P2P
completes ~20 s before the crash, so not the trigger), and the GUI vs headless path. Training/eval is
unaffected because it loads `isaaclab.python.headless.kit` (PhysX + CUDA only, no RTX renderer) and
never touches `librtx.scenedb` — which is why all 8-seed numbers are valid and only on-screen
rendering is blocked. Upgrading Isaac Sim does NOT
help (the bug spans 4.5–5.1); the only known fix is a **driver downgrade to ≤591 / 580 /
535.161.07**, which needs root — out of reach in this no-sudo container.
Refs: github.com/isaac-sim/IsaacSim issue #537 (595.79 fails, 580 works); NVIDIA forum threads on
`rtx.scenedb.plugin` crashes with 595.x.

Consequence: deck/README visuals use the Isaac-Gym kinematic replay of the Isaac-Lab-trained
trajectory (honestly labelled as such). The native clip can be produced later, unchanged-command,
once an admin downgrades the driver into Isaac Sim 5.1's supported range.

## Rough-terrain kinematic replay (the render we actually ship) — 2026-06-06

Since the native RTX render is blocked (above), the deck/README clean-gait visual is a **kinematic
replay**: the joint states are the real Isaac-Lab-trained trajectory, but they are *rendered* in
Isaac Gym (its older GL renderer works on this box; Isaac Lab's RTX does not). This is NOT a native
Isaac Lab render — we set the recorded base pose + joint angles each frame, no physics, no policy.
To make it honest about what the policy walks on, we extended two of OUR OWN scripts (not the SATA
repo) to replay on the ACTUAL rough terrain with a forward command. Output is clip 06 in
`results/README.md`: `results/deck/06_roughReplay_cleanWalk.{mp4,gif}` + `..._still.png`.

### `scripts/eval_metrics.py` — fixed command + terrain dump
- **`--cmd_vx/--cmd_vy/--cmd_wz`** lock the velocity command to a fixed value
  (`scripts/eval_metrics.py:90-92` add the args; `:209-222` fetch the `base_velocity` command term,
  set `resampling_time_range = (1e9, 1e9)` to disable resampling, clear `is_standing_env`, and pin
  `vel_command_b`; `:235-236` re-pin it after each step so an episode reset doesn't reseed it).
  WHY: the Play task otherwise samples a random command (every 5 s; lin_vel_x range (-0.5, 1.5)),
  which can land near 0 so the robot steps in place. With `--cmd_vx 1.0` the s4 policy walks — we
  observed net horizontal base displacement ~2.40 m over 500 steps, vs ~0.24 m unlocked.
- When **`--traj_out`** is set, it also dumps the terrain mesh (world-frame vertices + faces) to a
  sibling `*_terrain.npz`, read straight from the imported USD terrain prim
  (`scripts/eval_metrics.py:311-359`). We read from USD because `TerrainImporter.meshes` is
  deprecated/empty in this Isaac Lab version, AND re-generating the terrain is not reproducible (the
  SATA rough-slope noise draws from the global NumPy RNG with cfg `seed=None`). Flat envs have no
  terrain prim, so this is skipped.

### `scripts/render_replay_isaacgym.py` — terrain mesh + cosmetic restyle
- **`--terrain <npz>`** loads that mesh via `gym.add_triangle_mesh` in the same world frame as the
  trajectory, so the feet align with the bumps instead of floating over a flat plane
  (`scripts/render_replay_isaacgym.py:55-57` arg; `:80-93` load-and-add, falling back to a ground
  plane when no terrain is given).
- Cosmetic restyle so this Isaac-Lab-reproduction clip is visually distinguishable from the original
  SATA Isaac-Gym videos (same renderer otherwise): robot tinted teal via `set_rigid_body_color`,
  cool key light via `set_light_parameters`, closer follow-camera
  (`scripts/render_replay_isaacgym.py:120-128` colour + light; `:169-173` the closer camera offset).
  This touches only appearance — joint states and terrain are untouched.

### Pipeline (re-runnable for any seed)
Eval in the `isaaclab` env headless (dumps trajectory + terrain), then replay in the `sata` env with
`DISPLAY=:0`:
```bash
# 1. eval (isaaclab env, headless): dump trajectory CSV + terrain npz, forward command locked
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
  ~/workspace/isaaclab-torque-locomotion/scripts/eval_metrics.py \
  --task Isaac-Velocity-Rough-Go2-Sata-Play-v0 --load_run <run_dir> \
  --num_envs 1 --steps 500 --headless --cmd_vx 1.0 \
  --traj_out results/traj_rough_s4.csv
# 2. replay (sata env, VNC display): render the rough terrain + trajectory
DISPLAY=:0 python scripts/render_replay_isaacgym.py \
  --traj results/traj_rough_s4.csv --terrain results/traj_rough_s4_terrain.npz \
  --out results/deck/06_roughReplay_cleanWalk.mp4 --gpu 0
```
Reminder: this is a cross-engine replay (real Isaac-Lab joint states, Isaac-Gym rendering), not a
native Isaac Lab render — labelled that way wherever the clip appears.
