# SATA → Isaac Lab reproduction notes (carry-forward checklist)

Hard-won learnings from the v1 debugging journey (branch `main`, through commit `992139e`),
carried into the `repro-v2-cleanroom` attempt. Goal: a faithful, *direct* port — avoid the
"clever" deviations that caused v1's bugs, and judge the residual gap cleanly.

Ground truth = the user's working Isaac Gym reproduction (`~/workspace/SATA/legged_gym`,
`go2_torque`). (An early 3-seed run gave a prettier mean reward 114 ± 6; the **8-seed reference we
actually compare against is 103.6 ± 16.0** — see the "8-seed reproduction result" section below,
which supersedes this line. We report the 8-seed number.)

## ✅ VALIDATED — keep as-is (don't re-derive)
- **Bio math** (`bio_constraints.py`): tanh-EMA activation (γ=0.6, new-sample weight), Hill
  `τ=τ_lim·α·(1−sign(α)·q̇/q̇_lim)`, fatigue `(ζ+|τ|dt)·β` (β=0.9). 19 sim-free tests pass.
- **Growth** (`growth.py`): Gompertz `exp(−exp(−k(t−x0)))`, k=3e-5, x0=24000; τ 7.05→23.5 (ALL
  joints incl. rear — SATA's `r_leg_scaled`≡1.0 is a no-op, NOT a rear-constant); freq 100→200 Hz.
- **PPO**: reuse Isaac Lab `UnitreeGo2RoughPPORunnerCfg` — every hyperparam == SATA's GO2TorqueCfgPPO
  ([512,256,128], elu, lr 1e-3 adaptive, entropy 0.01, clip 0.2, γ0.99, λ0.95, KL0.01, 24 steps,
  5×4 epochs×minibatch). Train 3000 it / 4096 envs.
- **Robot**: Isaac Lab Go2 USD mass 15.019 kg == SATA exact.
- **Terrain**: trimesh ROUGH — `terrain_proportions=[0.2 smooth slope, 0.8 rough slope]`,
  curriculum OFF, difficulty {0.5,0.75,0.9}≈range(0.5,0.9), slopes 0..0.1 rad + ±0.06 m noise.
  (`sata_terrain.py`. NOT flat — flat was v1's first bug.)
- **Reward terms + scales** (`sata_mdp.py`): forward 10, head_height 5, moving_y 5, moving_yaw 5,
  soft_dof_pos_limits −5, roll −5, lin_vel_z −5, motor_fatigue −0.05, dof_acc −1e-6. φ=exp(−4|x|).
  `_reward_forward` = SINGLE exp of growth-blended target (ceiling 1.0), NOT the paper Table-II
  two-term sum. `joint_acc` = PhysX `data.joint_acc` (finite-diff trained WORSE — tested, reverted).
- **obs** = 60-dim, NO height-scan to policy, incl. applied_torque + motor_fatigue; clip ±100.
- **Domain rand**: friction U[0.5,1.25], base mass U[−1,5] + COM x±0.2/y,z±0.1, reset joint
  ×U(0.95,1.05) + base ±1 m, push×G. Terminations: flip-over + hard-limit ±0.05 (SATA urdf
  limits) + time_out.

## ⚠️ PITFALLS — confirmed bugs from v1 (avoid in any rewrite)
1. **Terrain must be ROUGH, not flat** (flat under-estimates the torque envelope).
2. **Reward fed to PPO = a SINGLE substep** (`dt = PHYSICS_DT`). SATA's `compute_reward` ZEROS
   `rew_buf` each call → the per-env-step reward is the LAST substep's value, NOT a sum over
   substeps. v1 used `dt = n_sub·PHYSICS_DT` → 2× reward during growth → destabilised value/LR.
3. **Variable-frequency loop MUST carry the residual** (`self._ctrl_dt %= 1/freq`, persistent).
   Resetting it each step pins n_sub=2 → control freq stuck at 100 Hz until G=1, then a hard jump
   to 200 Hz at deploy (train/deploy mismatch). With carry-over, n_sub varies 1↔2 → smooth 100→200.
4. **episode_length and growth_step advance PER SUBSTEP** (`+= n_sub`) — SATA counts substeps
   (correct 10 s horizon + G(t) timing). (Only the REWARD is single-substep; counters accumulate.)
5. **Calf physics velocity limit**: Isaac Lab USD caps the calf at **15.70 rad/s**, SATA urdf =
   **20.07**. CONFIRMED BINDING (calf saturates at 15.70 in the gait) → throttled gait, stiff
   extension-limit-riding front legs. Write 20.07 to sim at env init (`write_joint_velocity_limit
   _to_sim`). [The remaining config lever being tested on this branch.]

## What closed the gap
Fix 5 (calf-vel) alone did **not** close it. The config difference that mattered was a 6th item we
found later: SATA's joint position limits were only used in the reward/termination, never written
to the physics sim, so the wider stock-USD thigh limit let exploration over-extend the thigh and
trip the hard-limit termination on most episodes. Writing SATA's `go2_torque.urdf` limits to the
sim (`robot.write_joint_position_limit_to_sim`, `src/torque_loco/go2_sata_env.py`) raised episode
survival and produced a clean bent-leg gait. The 8-seed result and the residual-gap decomposition
below characterise what remains — we do **not** label the residual a "cross-engine gap" without
first decomposing it.

## 8-seed reproduction result (SATA-rough, branch `repro-v2-cleanroom`)
After fixes 1–6 + angular/linear damping = 0, we retrained 8 seeds on the SATA-rough terrain
(`go2_sr_s1..8`, 4096 envs, 3000 it). Raw training logs are gitignored (`results/*.log`); the
numbers below are recomputed from those logs and from the Gym reference tfevents under
`../SATA/legged_gym/logs/SATA/*ref_s*` (read-only). Reward is rsl_rl's mean episodic return; we
report mean ± sample std (ddof=1).

|                       | mean reward (8 seeds) | clean 7 (drop 1 collapse) |
|-----------------------|-----------------------|----------------------------|
| Isaac Lab (ours)      | **76.8 ± 16.2**       | 82.3 ± 5.0                 |
| Isaac Gym (SATA ref)  | 103.6 ± 16.0          | 108.6 ± 8.0                |

Both engines show the same structure: 7 seeds cluster, 1 collapses (ours `s5` → 38.4; the Gym
reference `s7` → 68.4). We report the 8-seed number (with the collapse) as the headline, not the
cleaner 7-seed number. Because both means are over the same population (4096 envs × full
command/terrain distribution, matched PPO, 3000 it), "the mean is dragged down by the hard tail"
applies to both and does not explain the difference. We decompose it per-step.

### Per-step decomposition (removes the episode-length confound)
per-step reward = mean episodic return / mean episode length (both rsl_rl fields):

|                  | reward | ep_len | per-step |
|------------------|--------|--------|----------|
| Gym clean-7      | 108.6  | 1959   | 0.0554   |
| Lab clean-7      |  82.3  | 1767   | 0.0465   |
| ratio (Lab/Gym)  | 0.76   | 0.90   | 0.84     |

0.90 × 0.84 = 0.76: ~10% of the gap is shorter episodes, ~16% is lower per-step reward.

### Per-term per-step (where the 16% lives)
Both engines log `Episode_Reward/<term> = mean(episode_sum) / max_episode_length_s` (Isaac Lab
`source/isaaclab/.../reward_manager.py:120`; legged_gym `legged_gym/envs/base/legged_robot.py:184-185`).
With matched weights, horizon (`episode_length_s=10`) and n_sub=1 at G=1 (env-step = substep), the
per-step term = tag × 10 / ep_len. Clean seeds only (Gym drops s7, Lab drops s5):

| term         | Gym/step | Lab/step | Lab−Gym |
|--------------|----------|----------|---------|
| forward      |  0.0400  |  0.0388  | −0.0012 |
| head_height  |  0.0130  |  0.0131  | +0.0001 |
| moving_y     |  0.0161  |  0.0138  | −0.0023 |
| moving_yaw   |  0.0127  |  0.0131  | +0.0004 |
| roll         | −0.0017  | −0.0014  | +0.0003 |
| lin_vel_z    | −0.0007  | −0.0007  |  0.0000 |
| fatigue      | −0.0130  | −0.0110  | +0.0020 |
| joint_acc    | −0.0111  | −0.0194  | −0.0083 |

What we observe (on the axis we measured — not a claim about the mechanism):
- The positive/task terms (forward, head_height, moving_y, moving_yaw) match the Gym reference to
  within ~0.002/step. The policy reproduces SATA's per-step task performance.
- The per-step deficit is concentrated almost entirely in one penalty term, `joint_acc`: Lab pays
  0.0083/step more, ≈93% of the ~0.0089/step total per-step gap. Other penalties (fatigue, roll)
  are slightly smaller in Lab.

### Why `joint_acc` differs (a documented definitional difference, not a tuning choice)
SATA's `_reward_dof_acc` is a finite difference `((last_dof_vel − dof_vel)/dt)^2`
(`../SATA/legged_gym/legged_gym/envs/base/legged_robot.py:899-901`); our term uses
PhysX5's instantaneous `data.joint_acc^2` (`src/torque_loco/sata_mdp.py:144-150`). We tried matching
the finite-difference form (it trained markedly worse — see that comment — and was reverted).
PhysX5's instantaneous acceleration captures contact-impact spikes the finite difference smooths
over, so the same motion is penalised more under our term. This suggests the residual per-step gap
is dominated by how this one term is measured, not by worse locomotion.

### Open questions (honest)
- We have **not** separated "the penalty measures acceleration differently" from "the Lab gait is
  genuinely jerkier." The matched positive terms make "the policy did not learn the task"
  implausible, but confirming the `joint_acc` gap is purely a measurement difference would need
  joint-acceleration trajectories logged under a single matched definition on both engines.
- The ~10% episode-length shortfall (Lab terminates a little earlier: 1767 vs 1959) is left as a
  real, unexplained residual rather than attributed to a cause we have not isolated.

We do not claim the engines are equivalent. We claim only: under matched config, the policy
reproduces SATA's per-step task performance, and the reward-number gap is dominated by one penalty
term with a documented cross-engine definitional difference.
