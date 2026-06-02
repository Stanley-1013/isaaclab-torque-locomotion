#!/usr/bin/env python
"""Print the Isaac Lab Go2 articulation's physical parameters (mass, joint limits, armature,
default pose) for comparing against SATA's go2_torque.urdf. Headless, no rendering."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
import torque_loco.__register__  # noqa: E402,F401

import argparse  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401


def _main():
    cfg = parse_env_cfg("Isaac-Velocity-Flat-Go2-Sata-Play-v0", num_envs=1)
    env = gym.make("Isaac-Velocity-Flat-Go2-Sata-Play-v0", cfg=cfg)
    env.reset()  # populate articulation data + physx views
    robot = env.unwrapped.scene["robot"]
    d = robot.data
    names = robot.joint_names
    print("[INSPECT] env reset done; reading params", flush=True)

    print("\n========== Isaac Lab Go2 (USD) physical params ==========", flush=True)
    # joint limits FIRST (robot.data tensors are safe post-reset; avoid physx_view bindings)
    print("ILAB joint (name: pos lo..hi, vel_lim, effort_lim, armature, default):", flush=True)
    pos_lim = d.joint_pos_limits[0]      # (J,2)
    for i, n in enumerate(names):
        lo, hi = pos_lim[i].tolist()
        def g(attr):
            t = getattr(d, attr, None)
            return float(t[0][i]) if t is not None else float("nan")
        print(f"  {n:16s} lo={lo:.3f} hi={hi:.3f}  vel={g('joint_vel_limits'):.2f}  "
              f"eff={g('joint_effort_limits'):.2f}  armature={g('joint_armature'):.4f}  "
              f"default={float(d.default_joint_pos[0][i]):.3f}", flush=True)
    # masses via robot.data.default_mass (per-body tensor; no physx_view call)
    dm = getattr(d, "default_mass", None)
    if dm is not None:
        masses = dm[0]
        print(f"\nILAB total mass = {float(masses.sum()):.3f} kg ; n_bodies = {len(robot.body_names)}", flush=True)
        for bn, m in zip(robot.body_names, masses.tolist()):
            if "base" in bn or "trunk" in bn:
                print(f"  base/trunk body '{bn}' mass = {m:.3f} kg", flush=True)
    else:
        print("ILAB default_mass not available on robot.data", flush=True)

    print("\nSATA URDF for reference: total 15.019 kg (base 6.921); "
          "hip lo/hi ±1.047, thigh [0,1.5]F/[0,2.0]R, calf [-2.723,-0.838]; "
          "vel hip/thigh 30.1 calf 20.07; eff hip/thigh 23.7 calf 35.55")
    env.close()


def main():
    import traceback
    try:
        _main()
    except Exception:
        print("[INSPECT] EXCEPTION:\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
