# SATA → Isaac Lab reproduction notes (carry-forward checklist)

Hard-won learnings from the v1 debugging journey (branch `main`, through commit `992139e`),
carried into the `repro-v2-cleanroom` attempt. Goal: a faithful, *direct* port — avoid the
"clever" deviations that caused v1's bugs, and judge the residual gap cleanly.

Ground truth = the user's working Isaac Gym reproduction (`~/workspace/SATA/legged_gym`,
`go2_torque`), which converges to **mean reward 114 ± 6** (seeds 122/108/112) at iter 3000,
episode length ~2026 (substeps), and walks well.

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
   _to_sim`). [The decisive remaining config lever being tested on this branch.]

## OPEN QUESTION
After fixes 1–4, v1 still trained to a lower reward than SATA with a stiff, calf-extension-limit
gait (front calves ride −0.838 even on flat; base height 0.38 = stands tall but locked-straight).
Fix 5 (calf-vel) is the last concrete config difference. **If it closes the gap → done. If not →
the residual is the genuine Isaac Gym ↔ Isaac Lab cross-engine dynamics gap, which is itself the
research finding of this migration (a literal-from-zero rewrite would hit the same engine gap).**
