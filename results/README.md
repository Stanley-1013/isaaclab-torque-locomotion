# results/

**Active (current authoritative run — 2026-06-03 rough-terrain, post faithfulness fixes):**
- `go2_sr_s1..8.log`  — SATA-rough (Isaac-Velocity-Rough-Go2-Sata-v0), 8 seeds
- `go2_dr_s1..8.log`  — default-rough (Isaac-Velocity-Rough-Go2-Sata-Default-v0), 8 seeds
- `dispatch_gpu0..3.log` — per-GPU dispatchers
- `deck/` — before/after verification clips (Gym kinematic replay of Isaac-Lab-trained policies)

**archive/** — superseded artifacts, kept for provenance:
- `flat_faithful_run/` — first faithful run on FLAT terrain (wrong control variable; terrain was the bug)
- `flat_rewardfix/`    — flat reward-over-scale fix batch
- `gym_flat_envelope/` — first 8-seed envelope eval (flat; numbers invalid, superseded by rough @G=1)
- `render_attempts/`   — RTX/native-render diagnostic logs (container IOMMU/Vulkan blocked)
- `scratch_frames/`    — extracted render frames
- `misc_logs/`         — inspect_robot / smoke-test logs
