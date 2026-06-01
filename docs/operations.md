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
- **Tier-0 gate IN PROGRESS:** stock `Isaac-Velocity-Flat-Unitree-Go2-v0`
  smoke-train (64 envs, 5 iters) launched on GPU 0; correct python confirmed,
  Isaac Sim booting (first launch is slow — shader compile + Go2 USD pull).
  Not yet confirmed training. Log: `~/workspace/go2_smoke.log`.

## Next steps (resume here)
1. Confirm Tier-0: `grep "Learning iteration" ~/workspace/go2_smoke.log`. If it
   trains, Phase 0 is fully cleared.
2. **Task 1.2** — Go2 effort-control env cfg. FIRST verify the installed 5.1 API:
   the real env-cfg class name, the `actions` term type, and the Go2 actuator
   group key (the plan guesses `"base_legs"`):
   ```bash
   python -c "import isaaclab_tasks.manager_based.locomotion.velocity.config.unitree_go2 as m; print(m.__file__)"
   ```
   Then implement `src/torque_loco/go2_torque_env_cfg.py` + `__register__.py` per
   the plan (`docs/plans/2026-06-01-implementation-plan.md`, Task 1.2).
3. Continue plan tasks 1.3 → 2.2 → 2.3 → 3.2 (Tier 1 then Tier 2).
