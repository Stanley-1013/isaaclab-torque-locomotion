#!/usr/bin/env python
"""Launcher: register the torque-control Go2 tasks, then delegate to Isaac Lab's
stock rsl_rl ``train.py`` in the SAME process.

Why a wrapper (the "working mechanism" Task 1.2 asks us to record):
  - The Gymnasium registry is process-global, so our ``gym.register`` must run in
    the same interpreter that later calls ``gym.make``.
  - Isaac Lab's ``train.py`` resolves the task cfg via a hydra decorator at module
    import, so our task must already be registered before that import.
  - Registration (``torque_loco.__register__``) is pure ``gym.register`` with string
    entry points, so it is safe to import BEFORE the Omniverse app launches (no pxr).

We therefore: add ``src/`` to sys.path, import the registration, then ``runpy`` the
stock train.py under ``__main__`` (its dir prepended to sys.path so its local
``import cli_args`` resolves). All CLI args after this script flow through sys.argv
into train.py's own argparse untouched. Isaac Lab is left unmodified.

Run via Isaac Lab's python, e.g.:
  conda activate isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
  cd ~/workspace/IsaacLab
  CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
    ~/workspace/isaaclab-torque-locomotion/scripts/train_go2.py \
    --task Isaac-Velocity-Flat-Go2-Torque-v0 --headless --num_envs 1024 --max_iterations 10
"""

import os
import runpy
import sys

# Make the torque_loco package importable (src layout), then register tasks.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
import torque_loco.__register__  # noqa: E402,F401  (registers Go2-Torque tasks)

# Locate Isaac Lab's rsl_rl train.py; allow override via ISAACLAB_PATH.
_ISAACLAB = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/workspace/IsaacLab"))
_TRAIN = os.path.join(_ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl", "train.py")
if not os.path.isfile(_TRAIN):
    raise FileNotFoundError(
        f"Isaac Lab train.py not found at {_TRAIN}. Set ISAACLAB_PATH to your IsaacLab clone."
    )

# train.py does `import cli_args` from its own directory.
sys.path.insert(0, os.path.dirname(_TRAIN))

# Delegate. train.py launches the sim app, resolves our registered task, and trains.
runpy.run_path(_TRAIN, run_name="__main__")
