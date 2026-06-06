# results/ — chronological index

Numbered by **order of production** (00 = earliest). Read top-to-bottom to follow the debugging arc.

## TOP LEVEL = the final run
`go2_sr_s1..8.log` — 8 seeds on SATA-rough, all cleanroom fixes, 3000 it / 4096 envs, completed
2026-06-04 (branch `repro-v2-cleanroom`). The default-rough variant (`go2_dr_*`) was dropped: it is
not needed for the headline (the SATA-rough run is the faithful comparison). This run is the basis
for the headline below and supersedes everything in `experiments/`.

## experiments/ — the debugging timeline (each folder = one iteration)
| # | folder | what changed | result (reward / episode / hardlimit) |
|---|--------|--------------|------|
| 00 | `00_pre_cleanroom/` | flat-terrain runs, gym envelope, render/RTX attempts (pre-branch) | flat baseline; terrain was wrong control variable |
| 01 | `01_v1_buggy_16run_evals/` | first rough 16-run (reward-2× + no carry-over bugs) + its evals | reward 66 (2× basis), stiff gait, calf locked |
| 02 | `02_v2_rewardDt_carryOver/` | +reward single-substep +control-freq carry-over | better peak, still regressed; hardlimit 0.6 |
| 03 | `03_v3_calfVel/` | +calf velocity limit 20.07 | no improvement; hardlimit 0.6–0.9 |
| 04 | `04_v4_friction/` | +friction effective-μ match | no improvement; hardlimit 0.7–0.8 |
| 05 | `05_v5_physicalLimits/` | +SATA physical joint limits written to sim (the fix that closed the gait gap) | reward 83±4 (v5, 4 seeds), episode ~1750, hardlimit ~0.01, calf-locking 0% ✅ |
| — | (top level, final) | +angular_damping=0, 8-seed SATA-rough run | reward 76.8±16.2 (8 seeds; see Headline) |

Each experiment folder has: `go2_*_s*.log` (training), `dispatch_*` (launchers),
`metrics_*.csv` / `traj_*.csv` (G=1 eval rollouts).

## deck/ — verification clips (Gym kinematic replay of the Isaac-Lab-trained trajectories)
| # | file | shows |
|---|------|-------|
| 01 | `01_flat_rewardBug_crawl` | the reward-bug belly-crawl (0.10 m) |
| 02 | `02_flat_rewardFix_walk` | flat reward-fix upright walk (0.30 m) |
| 03 | `03_v1buggy_roughTraj_onFlatRender` | v1 rough trajectory replayed on flat (looks broken — render artifact) |
| 04 | `04_v1buggy_flatEval_tallButLocked` | v1 stands tall but front calves locked straight |
| 05 | `05_v5_cleanWalk` | v5 clean bent-leg gait, base 0.31 m, calves cycling (flat eval) ✅ |
| 06 | `06_roughReplay_cleanWalk` | **clean gait on the ACTUAL SATA-rough terrain** (s4 model_2999; terrain mesh read from the Isaac Lab eval's USD so feet align with the bumps) ✅ |

## Headline (final 8-seed, SATA-rough)
Mean reward (mean ± sample std, ddof=1): **Isaac Lab 76.8 ± 16.2** vs **Isaac Gym ref 103.6 ± 16.0**.
Both engines show 7 seeds clustering + 1 late PPO collapse (ours s5→38.4; Gym s7→68.4); clean-7 is
82.3 ± 5.0 vs 108.6 ± 8.0. Decomposed per-step, the gap ≈ 10% shorter episodes × 16% lower per-step
reward; the task/tracking terms match Gym to ~0.002/step, and ~93% of the per-step deficit is one
penalty term (`joint_acc`) with a documented cross-engine definitional difference. The porting fix
that closed the gait gap was writing SATA's joint position limits to the physics sim. Full provenance:
`../docs/REPRODUCTION_NOTES.md` (numbers + per-term derivation) and `../docs/operations.md`.
