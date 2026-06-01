# Design — Faithful SATA migration to Isaac Lab (full bio stack incl. growth curriculum)

**Date:** 2026-06-02
**Status:** approved (verbal "好", 2026-06-02)
**Supersedes the scope of:** `docs/specs/2026-06-01-design.md` §2/§4.3 — which deferred the
Hill model and growth curriculum for the one-week slice. After reading the SATA paper
([arXiv:2502.12674](https://arxiv.org/abs/2502.12674), RSS 2025) and the reproduction repo,
the user chose to **pull the full bio stack (incl. growth) back into scope** for a faithful
cross-engine reproduction. This document is the authoritative design for the bio port.

**Sources read first-hand for this spec:** the SATA paper (arXiv HTML → markdown, method
§III–IV read directly — Eqs 1–7, Tables I–III); SATA `legged_gym` source
(`go2_torque.py`, `go2_torque_config.py`); the reproduction repo
(`docs/sata-overview.md`, `training-internals.md`, `results/phase{1,3}`); and the installed
Isaac Lab 5.1 source (actuators, managers, env step). Where the reproduction's prose docs
conflict with the paper/code (obs order; EMA direction), **paper + code win** — both verified.

---

## 1. What SATA actually is (grounded in the paper + code)

SATA = *Safe and Adaptive Torque-based locomotion* (Li et al., MARMot Lab @ NUS). The problem:
torque control is more compliant/adaptive than position control but **hard to train** (highly
nonlinear action space, inefficient cold-start exploration). SATA shapes the torque action space
with three bio-inspired structures so it is both trainable and physically plausible:

1. **Biomechanical model (activation low-pass + Hill force-velocity).** *Critical for learning* —
   the paper's ablation: "without the biomechanical model the robot is completely unable to learn
   a coherent gait." Its measurable side-effect is a **hardware-realisable torque envelope**
   (ablating activation lets peak torque jump to 42.5 N·m, near the real Go2's 45 N·m limit).
2. **Motor-fatigue feedback state.** A per-joint leaky integrator, fed into the observation and
   lightly penalised (−0.05); enables adaptive load-shedding / leg-effort redistribution.
3. **Growth curriculum (the named "Adaptive / Animal-Learning" core).** A single Gompertz scalar
   G(t)∈[0,1] *progressively unlocks the body* over training — it is **not** task-difficulty
   curriculum but **embodiment growth**: "expands what the agent is physically allowed to do,
   leading to deeper exploration and reduced risk of suboptimal shortcuts." It improves early
   exploration **and** the deployed policy at the edges (the paper: `no_growth` "struggles to
   achieve high commanded velocities (1.5 m/s), especially above the training range" + worse OOD).

**Why faithfulness matters for *our* goal.** Our goal (repo README / `docs/specs/2026-06-01-design.md`):
(1) cross-engine reproducibility of the **feasibility-envelope finding** (Go2 ground-truth numbers);
(2) cross-engine zero-shot **OOD transfer** as a sim-to-real proxy. Growth is most relevant exactly
to (2) (it shapes OOD behaviour), so it cannot be dismissed as a warm-up trick — that was the
misread corrected during brainstorming.

---

## 2. Scope decisions (locked 2026-06-02)

- **Bio fidelity:** **full SATA incl. growth curriculum** — activation low-pass + Hill + fatigue +
  Gompertz growth (torque ceiling, control frequency, command range, domain-randomization strength,
  reward-weight modulation).
- **Tier-2 comparison structure:** train **only the full-bio reference**; validate it **reproduces
  SATA's ground-truth envelope numbers** in Isaac Lab (walks; peak τ ≈ 22.5 N·m inside the 23.5/45
  envelope; energy/jerk in range; reward magnitude sane). **No `no_bio` training** — the paper shows
  a full-no-bio policy cannot learn a coherent gait, and the user does not want the per-knob ablation
  study (that was the prior project). The envelope claim is made by comparison to the hardware lines
  (23.5 sim clip, 45 real) and the SATA ground-truth metrics, not by an in-repo ablation.
- **Engine reward not directly comparable** across engines; validation against ground truth is on the
  **envelope metrics** (peak τ / energy / jerk), which *are* comparable, plus qualitative "walks +
  tracks command". (Anti-over-claim, carried from the 2026-06-01 guardrails.)

---

## 3. Architecture

Manager-based, reusing the stock Go2 velocity task plumbing from Tier-1, with **one** structural
addition: a `ManagerBasedRLEnv` subclass whose `step()` is overridden to support SATA's
variable control frequency. Everything else is standard managers (obs / reward / command /
curriculum / event / actuator).

### 3.1 `Go2SataEnv(ManagerBasedRLEnv)` — variable-frequency step
Research finding: `decimation` is read **once** and hard-wired in `ManagerBasedRLEnv.step()`
(`for _ in range(self.cfg.decimation)`), so frequency growth is the **only** SATA growth component
not expressible via managers. We do **not** rewrite a full `DirectRLEnv` (that would discard the
manager reuse and is far heavier for one piece). Instead we subclass `ManagerBasedRLEnv` and
override `step()`, replacing the fixed decimation loop with SATA's accumulator:

```
while current_dt * current_freq < 1:   # current_freq = 100 + 100*G(t)
    apply action; sim.step(); current_dt += physics_dt
current_dt %= 1 / current_freq
```

Physics runs at a fixed 200 Hz (`sim.dt = 0.005`); the **control** rate grows 100→200 Hz
(effective decimation 2→1) as G(t) ramps. The accumulator handles fractional frequency exactly as
SATA does. The override is a small, contained copy of the parent `step()` body — accept coupling to
Isaac Lab's internal `step()` (a known upgrade-maintenance cost; documented).

### 3.2 Gompertz growth — single G(t) buffer driving five schedules
`G(t) = exp(-exp(-k·(step - x0)))`, **k=3e-5, x0=24000**, driven by `env.common_step_counter`
(cumulative env steps, not PPO iterations). G(t) is stored on the env (`env._G`, per-step) inside
the overridden `step()` so the actuator, reward, command, and event terms all read one source.

| # | Schedule | Mechanism (manager-based) |
|---|----------|---------------------------|
| a | torque ceiling τ_start→τ_end **7.05→23.5 N·m** (front legs; rear stays 23.5) | BioActuator reads `τlim = τ_end·(τscale via G)` from `env._G`; Eq 6 |
| b | control frequency **f_start→f_end 100→200 Hz** | `Go2SataEnv.step()` accumulator (3.1); Eq 7 |
| d | push-velocity DR scales with G | push event term reads `env._G` (SATA: `max_push_vel·general_scale`). Mass(≤5kg)/friction([0.5,1.25])/COM(x±0.2,y/z±0.1)/hold-prob(10%) are **fixed** ranges (paper §IV-B) |
| e | reward modulation (Table II) | reward funcs read `env._G`: `forward` blends target midpoint→command, `moving_y`/`moving_yaw` ×G, `base_height` ×(1+G) |

**Corrected vs an earlier draft:** command *ranges* are **FIXED** (vx[-0.5,1.5], vy[-0.5,0.5],
yaw[-1.5,1.5]); SATA's "growth on commands" lives in the `forward` reward (blend target
midpoint→actual via G — paper Table II / code `_reward_forward`), **not** a widening
command-range curriculum. No command-range curriculum term is needed.
Constants (paper Table III, confirmed vs code): k=3e-5, t0=24000, τ_start=7.05, τ_end=23.5,
f_start=100, f_end=200, κ_scale=5, γ=0.6, β=0.9. During training G asymptotes <1; **at
deployment f,τ are set to their max** (f_end=200, τ_end=23.5).

### 3.3 `BioActuator(IdealPDActuator)` — the biomechanical model + fatigue
Feasibility confirmed against installed source (stateful per-env buffers; `reset(env_ids)`
auto-called by `Articulation.reset()`; `self.effort_limit`/`self.velocity_limit` are
`(num_envs, num_joints)` tensors; `compute()` receives `joint_vel`; `applied_torque` exposed on
`asset.data`; custom buffers reachable via `asset.actuators["base_legs"]`).

`compute(control_action, joint_pos, joint_vel)` — `effort_cmd = control_action.joint_efforts`
(= policy action × 5 from a `JointEffortActionCfg(scale=5)`):
```
τlim   = effort_limit_max * τscale(G)            # 7.05→23.5 front; 23.5 rear
act_t  = tanh(effort_cmd / τlim)                 # activation_process
act    = act_t * 0.6 + act_prev * 0.4            # EMA, γ=0.6 new-weight (matches code+paper)
τ      = act * τlim * (1 - sign(act) * joint_vel / vel_limit)   # Hill; vel_limit per-joint
fatigue = (fatigue + |τ| * dt) * 0.9             # leaky integrator, β=0.9
return τ                                          # NO _clip_effort — envelope is the tanh, not a clamp
```
- `vel_limit` per-joint: hip/thigh 30.1, calf 20.07 rad/s (SATA URDF values).
- `dt` injected after construction (actuator has no native dt); fatigue accumulates per physics step.
- Flags `activation_process / hill_model / motor_fatigue` exist (mirroring SATA) but the migration
  only ever runs **all-on** (reference). No-bio path (all-off → τ = action×5) is implemented for
  completeness/plumbing parity but not trained as a comparison (see §2).
- `act_prev`, `fatigue` reset to 0 on `reset(env_ids)`.

### 3.4 Observation (60-dim) — paper Eq + code agree
Paper §III-B1: `o_t = [v_t, w_t, g_t, q, q̇, v_cmd, τ, ζ]`. Concretely (scales from code):
`[ lin_vel×2, ang_vel×0.25, proj_grav, (dof_pos−default)×1, dof_vel×0.05, cmd×[2,2,0.25],
   applied_torque(raw), motor_fatigue(raw) ]`. Last two are custom obs terms reading
`asset.data.applied_torque` and `asset.actuators["base_legs"].motor_fatigue`.
(`training-internals.md` lists `prev_action` and a different order — **wrong**; paper Eq + code
`go2_torque.py:287-295` agree on the order above.)

### 3.5 Reward — SATA's 9 terms (paper Table I/II, custom mdp functions)
Weights (paper Table I, each ×dt; dt=0.005): tracking_x +10, tracking_y +5, tracking_yaw +5,
base_height +5, roll −5, velocity_z −5, joint_limits −5, fatigue −0.05, joint_acc −1e-6.
- `φ(x)=e^{−4|x|}` (= `exp(−|x|/0.25)`, code's tracking_sigma=0.25 — equivalent).
- `fatigue = ζ·|τ_d·κ_scale|` (= fatigue × |scaled action|, κ_scale=5).
- Growth (Table II) read `env._G`: `tracking_x = φ(vx−mid)(1−G)+φ(vx−vx_cmd)(1+G)`,
  `tracking_y/yaw ×G`, `base_height = min(h_b,h_t)(1+G) − max(g_x, −min(0,0.2(1.5−2G)))`.
- Reward `×dt` is confirmed in the paper; for Isaac Lab preserve the *ratios* (verify the
  RewardManager's own step_dt convention at implementation so totals aren't double-scaled).

### 3.6 Commands & DR (paper §IV-B — fixed ranges)
Command ranges **fixed**, resampled every 5 s: vx[-0.5,1.5], vy[-0.5,0.5], yaw[-1.5,1.5].
(Growth on commands is in the `forward` reward, §3.5 — not a range curriculum.)
Domain randomization: base mass +≤5 kg; friction ∈[0.5,1.25]; COM shift x∈[−0.2,0.2], y/z∈[−0.1,0.1];
**10 % probability to hold last action/observation** (the `loss_rate`); **push velocity scales with G(t)**.
Reset: SATA repositions the robot **lying flat with random initial fatigue** — replicate via an event
that randomizes initial fatigue (and consider the prone init) to match the generalization init.

### 3.7 Control / training constants
`sim.dt=0.005` (200 Hz physics); control 100→200 Hz via §3.1 (at deployment 200 Hz, τ=23.5 full);
`episode_length_s=10` (≈2000 steps); PPO `num_steps_per_env=24`, 3000 iterations (SATA, ~20 min on a
4090 / ~65 min on an A6000 in the repro), actor/critic `[512,256,128]` ELU, 60-dim obs, 12-dim
Gaussian action, adaptive-KL target 0.01. Reuse the stock `UnitreeGo2FlatPPORunnerCfg`, adjusting
where it differs. `render_interval` set explicitly (decimation=1 default would render every step).
Terrain: rough (≤12 cm) + slopes (SATA); flat is acceptable for the first reproduction pass.

---

## 4. Sim-free, TDD-first components (no GPU)

These are pure-torch and unit-tested before any sim run (mirrors the existing `bio_constraints.py`
/ `metrics.py` discipline):
- **`bio_constraints.py` (rewrite):** activation low-pass (tanh+EMA) + Hill + fatigue leaky
  integrator — replacing the earlier hard-clip capacity model, which matched the *old* deferred
  scope, not SATA. Tests: EMA lag→converge; Hill reduces torque with same-direction velocity;
  fatigue accumulates under load and decays when idle; tanh bounds |act|<1 (soft envelope).
- **`growth.py`:** Gompertz G(t); tests for G(0)≈0-ish at small step, G→1 asymptote, monotonic.
- **`metrics.py` (already done):** peak torque / jerk / energy. Align reducers to the reproduction's
  `eval_under_conditions.py` (peak = max|τ|; energy = Σ|τ·ω|·dt; jerk = Σ|Δa|/dt).

---

## 5. Ground-truth validation targets (Tier-2 success)

From the reproduction repo's Phase-1/3 (`results/phase1-reference`, `phase3-bio-claims...`):
- **Walks** + tracks (vx,vy,yaw) without falling (qualitative, via a rendered `*-Play-v0` clip).
- **Reference peak torque ≈ 22.5 N·m** — inside the 23.5 sim clip / well under the 45 N·m real limit.
- Energy/jerk in the reference's nominal range (the no_fatigue blow-ups — 2.5× energy, 35× jerk — are
  NOT reproduced here since we don't train no-bio; they remain the prior project's result we cite).
- Reward magnitude sane and stable across seeds (SATA ref 114±6 / 104±16 — **not** directly comparable
  cross-engine; used only as an order-of-magnitude sanity check).

---

## 6. Risks & items to verify during implementation

1. **Cold-start convergence** — growth exists precisely to make cold-start torque-RL converge; if a
   first faithful run stalls, that is a signal the growth schedule (esp. early torque ceiling / freq)
   needs tuning, not a reason to drop it.
2. **`step()` override coupling** — small contained copy of Isaac Lab's internal `step()`; pin the
   Isaac Lab commit and re-diff on upgrade.
3. **USD joint effort-limit clamp** — verify PhysX does not clamp our returned torque below what the
   bio model intends (raise the USD drive effort limit if needed; SATA used raw force application).
4. **Reward `×dt` convention** — confirm Isaac Lab `RewardManager` weight semantics vs SATA's
   `scale×dt`; preserve term ratios.
5. **G(t) cadence** — frequency growth needs per-step G(t), so G is computed in the overridden
   `step()` and written to `env._G`; actuator (τ ceiling), reward (Table II), and the push-DR event
   all read it live. No CurriculumManager term is required (torque ceiling is applied inside the
   BioActuator, not via PhysX effort-limit writes; command ranges are fixed).
6. **Low-seed honesty** — report preliminary, sim-only; seeds resource-gated.

---

## 7. Honesty / framing guardrails (carried over)

- Sim-only; "within envelope" = within the rated torque number, not hardware-validated.
- Cross-engine reward magnitudes are not directly comparable; envelope metrics are.
- The bio layer is a feasibility constraint, not a reward device — same spine as the prior project.
- Document *why* each design choice diverges from SATA's Isaac-Gym implementation as a
  platform-migration finding (esp. the variable-frequency `step()` override).

---

## 8. Component boundaries (for the implementation plan)

| Unit | Purpose | Depends on |
|------|---------|-----------|
| `bio_constraints.py` | pure-torch activation+Hill+fatigue math | torch only (sim-free) |
| `growth.py` | pure-torch Gompertz G(t) | torch only (sim-free) |
| `metrics.py` | peak/jerk/energy reducers | torch only (sim-free) |
| `bio_actuator.py` | `BioActuator(IdealPDActuator)` wrapping bio math + fatigue buffer | isaaclab.actuators, bio_constraints |
| `go2_sata_env.py` | `Go2SataEnv(ManagerBasedRLEnv)` step() override (variable freq) + cfg | isaaclab.envs, bio_actuator |
| `sata_mdp.py` | custom obs terms (torque, fatigue); SATA's 9 reward terms (reading `env._G` for Table II growth); push-DR event term reading `env._G`; random-fatigue reset event | isaaclab managers, growth |
| `__register__.py` | gym task registration (string entry points) | gymnasium |
| `scripts/train_go2.py` | register-then-delegate launcher (already built) | — |

Files that change together live together under `src/torque_loco/`.
