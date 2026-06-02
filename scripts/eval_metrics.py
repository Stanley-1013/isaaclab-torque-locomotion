#!/usr/bin/env python
"""Rollout a trained SATA checkpoint and dump per-step torque/velocity/action CSV.

Register-then-delegate preamble: adds src/ to sys.path and registers the
Isaac-Velocity-Flat-Go2-Sata-* tasks BEFORE the Omniverse app launches, mirroring
train_go2.py. The AppLauncher is then built (which starts Isaac Sim), and only
after that are the heavy isaaclab/rsl_rl modules imported.

Typical usage (run via Isaac Lab's python):
    conda activate isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
    cd ~/workspace/IsaacLab
    ./isaaclab.sh -p ~/workspace/isaaclab-torque-locomotion/scripts/eval_metrics.py \
        --task Isaac-Velocity-Flat-Go2-Sata-Play-v0 \
        --load_run <run_dir> \
        --num_envs 32 \
        --steps 1000 \
        --headless

Output CSV columns: step, tau_0..tau_11, vel_0..vel_11, act_0..act_11
"""

# ---------------------------------------------------------------------------
# 1. Register our tasks BEFORE the Omniverse app launches (no pxr imports here).
# ---------------------------------------------------------------------------
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
import torque_loco.__register__  # noqa: E402,F401  (registers Sata tasks)

# ---------------------------------------------------------------------------
# 2. Argparse + AppLauncher (must happen BEFORE importing isaaclab env modules,
#    mirroring play.py lines 10-52).  cli_args lives next to play.py in Isaac Lab.
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

# Make Isaac Lab's local cli_args.py importable (same trick as train_go2.py with train.py).
_ISAACLAB = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/workspace/IsaacLab"))
_PLAY_DIR = os.path.join(_ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl")
if _PLAY_DIR not in sys.path:
    sys.path.insert(0, _PLAY_DIR)

import cli_args  # isort: skip  (local to Isaac Lab's rsl_rl scripts dir)

parser = argparse.ArgumentParser(description="Eval: rollout SATA checkpoint, dump CSV.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Go2-Sata-Play-v0",
    help="Gymnasium task ID to evaluate.",
)
# NOTE: --load_run and --checkpoint are provided by cli_args.add_rsl_rl_args() below
# (defining them here too caused an argparse conflict). Do not re-add them.
parser.add_argument(
    "--num_envs",
    type=int,
    default=32,
    help="Number of parallel environments (default 32).",
)
parser.add_argument(
    "--steps",
    type=int,
    default=1000,
    help="Number of rollout steps to record (default 1000).",
)
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_REPO, "results", "metrics_sata.csv"),
    help="Output CSV path.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent configuration entry point name (default rsl_rl_cfg_entry_point).",
)

# Append rsl_rl CLI args (--experiment_name, --load_run, --resume, --checkpoint, …)
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher CLI args (--headless, --device, --enable_cameras, …)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv so that Hydra (called inside the task config decorator) only
# sees its own args, exactly as play.py does (line 48).
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse / Isaac Sim.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 3. Heavy imports — AFTER the app is running (play.py lines 56-92).
# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
import isaaclab_tasks  # noqa: F401  (side-effect: registers stock tasks)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")

# ---------------------------------------------------------------------------
# 4. Main — wrapped in hydra_task_config exactly like play.py (line 97-98).
# ---------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Load checkpoint, roll out for --steps, write CSV."""

    # --- mirror play.py lines 101-114 ---
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --- resolve checkpoint path (mirror play.py lines 117-128) ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # --- create environment (mirror play.py lines 135-155) ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- load policy (mirror play.py lines 157-168) ---
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- rollout loop ---
    # Access the robot articulation via the unwrapped (Isaac Lab) env.
    # env.unwrapped already returns the underlying Go2SataEnv (the rsl_rl
    # wrapper delegates .unwrapped to the base env); do NOT add a second
    # .unwrapped.  scene["robot"] returns the Articulation asset.
    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]

    rows = []  # list of dicts, one per recorded step

    obs = env.get_observations()
    print(f"[INFO] Starting rollout for {args_cli.steps} steps ...")

    with torch.inference_mode():
        for step_idx in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Reset recurrent states for terminated episodes (mirror play.py lines 211-214).
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                # For older rsl_rl we need the underlying policy_nn reference.
                # We do a best-effort: skip reset (no recurrent states in vanilla PPO).
                pass

            # Capture env-0 data AFTER the step so applied_torque is populated.
            # applied_torque and joint_vel are (num_envs, num_joints) tensors.
            tau = robot.data.applied_torque[0].cpu()   # (J,)
            vel = robot.data.joint_vel[0].cpu()        # (J,)
            act = actions[0].cpu()                     # (J,)

            num_joints = tau.shape[-1]
            row = {"step": step_idx}
            for j in range(num_joints):
                row[f"tau_{j}"] = tau[j].item()
            for j in range(num_joints):
                row[f"vel_{j}"] = vel[j].item()
            for j in range(num_joints):
                row[f"act_{j}"] = act[j].item()
            rows.append(row)

    print(f"[INFO] Rollout complete. Writing {len(rows)} rows to: {args_cli.out}")

    # --- write CSV (stdlib csv, no pandas) ---
    out_path = os.path.expanduser(args_cli.out)
    _d = os.path.dirname(out_path)
    if _d:
        os.makedirs(_d, exist_ok=True)

    num_joints = tau.shape[-1]
    fieldnames = (
        ["step"]
        + [f"tau_{j}" for j in range(num_joints)]
        + [f"vel_{j}" for j in range(num_joints)]
        + [f"act_{j}" for j in range(num_joints)]
    )
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
