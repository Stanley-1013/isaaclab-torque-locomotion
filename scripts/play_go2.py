#!/usr/bin/env python
"""Launcher: register the SATA Go2 tasks, then delegate to Isaac Lab's stock rsl_rl
``play.py`` (same register-then-delegate trick as ``train_go2.py``).

Use to roll out a trained checkpoint and — with ``--video`` — record an mp4 clip
(written under the run's ``videos/`` dir). On a display-less box the RTX renderer needs
a GL context: run with ``DISPLAY=:0`` pointed at the TigerVNC X server.

Example:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
  export OMNI_KIT_ACCEPT_EULA=YES CMAKE_POLICY_VERSION_MINIMUM=3.5 DISPLAY=:0
  cd ~/workspace/IsaacLab
  CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
    ~/workspace/isaaclab-torque-locomotion/scripts/play_go2.py \
    --task Isaac-Velocity-Flat-Go2-Sata-Play-v0 --load_run <run_dir> \
    --num_envs 16 --video --video_length 400
"""

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
import torque_loco.__register__  # noqa: E402,F401  (registers Go2-Sata tasks)

_ISAACLAB = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/workspace/IsaacLab"))
_PLAY = os.path.join(_ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl", "play.py")
if not os.path.isfile(_PLAY):
    raise FileNotFoundError(
        f"Isaac Lab play.py not found at {_PLAY}. Set ISAACLAB_PATH to your IsaacLab clone."
    )

# play.py does `import cli_args` from its own directory.
sys.path.insert(0, os.path.dirname(_PLAY))

runpy.run_path(_PLAY, run_name="__main__")
