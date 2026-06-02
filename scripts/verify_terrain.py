#!/usr/bin/env python
"""Verify a Sata rough task actually produces non-flat terrain: print the spread of the
height-scanner ground heights across envs (flat -> ~0 std; rough slopes -> > a few cm)."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
import torque_loco.__register__  # noqa: E402,F401

import argparse  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-Go2-Sata-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401


OUT = "/tmp/terrain_verify.txt"

def _w(line):
    with open(OUT, "a") as f:
        f.write(line + "\n")
        f.flush()

def main():
    import traceback
    try:
        cfg = parse_env_cfg(args_cli.task, num_envs=256)
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset()
        u = env.unwrapped
        scanner = u.scene.sensors.get("height_scanner") if hasattr(u.scene, "sensors") else None
        _w(f"[VERIFY] {args_cli.task}")
        if scanner is None:
            _w("  NO height_scanner (flat terrain)")
        else:
            z = scanner.data.ray_hits_w[..., 2]            # (num_envs, num_rays)
            per_env = torch.mean(z, dim=1)                 # ground height under each robot
            within = torch.std(z, dim=1)                   # roughness under each robot
            _w(f"  ground height across envs:  min={per_env.min():.3f} max={per_env.max():.3f} "
               f"std={per_env.std():.3f} m  (slopes -> non-zero)")
            _w(f"  within-env ray std (mean):  {within.mean():.4f} m  (roughness -> non-zero)")
            tg = u.scene.cfg.terrain.terrain_generator
            if tg is not None:
                _w(f"  sub_terrains: {list(tg.sub_terrains.keys())}  curriculum={tg.curriculum}")
        env.close()
    except Exception:
        _w("[VERIFY] EXCEPTION:\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
    app.close()
