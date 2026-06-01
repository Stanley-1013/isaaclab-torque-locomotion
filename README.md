# isaaclab-torque-locomotion

Bio-inspired **torque-control** reinforcement learning for quadruped locomotion,
migrated to **NVIDIA Isaac Lab** (the successor to the now-deprecated Isaac Gym).

This is a follow-on to
[bio-inspired-adaptive-locomotion](https://github.com/Stanley-1013/bio-inspired-adaptive-locomotion)
— the SATA reproduction (Phases 1–4) built on Isaac Gym. That study found SATA's
bio-inspired torque layers act as **sim-to-real feasibility constraints, not
reward devices** (removing them *raises* training reward, but only by leaving the
hardware-realisable torque envelope).

This repo ports that idea to Isaac Lab, **anchored on the Unitree Go2**, and asks
two questions:

1. **Cross-engine reproducibility** — does the feasibility-envelope finding
   survive an engine change (Isaac Gym → Isaac Lab)? The Go2 numbers from the
   original study serve as ground truth.
2. **Cross-engine zero-shot transfer** — the engine gap is a proxy for the
   sim-to-real gap. Does the bio feasibility-envelope help a policy survive that
   gap better than the ablated policies?

**Design:** [`docs/specs/2026-06-01-design.md`](docs/specs/2026-06-01-design.md)

**Status:** bootstrapping (2026-06-01).
