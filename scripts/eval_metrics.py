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
parser.add_argument(
    "--traj_out",
    type=str,
    default=None,
    help="If set, also write a kinematic-replay trajectory CSV (base pose + joint angles + "
         "joint names) for rendering the motion in a working renderer (e.g. Isaac Gym).",
)
# Lock the velocity command to a fixed value (otherwise the Play task samples a random command
# that can be ~0, so the robot steps in place). Use e.g. --cmd_vx 1.0 for a forward-walking clip.
parser.add_argument("--cmd_vx", type=float, default=None, help="fixed forward velocity command (m/s)")
parser.add_argument("--cmd_vy", type=float, default=None, help="fixed lateral velocity command (m/s)")
parser.add_argument("--cmd_wz", type=float, default=None, help="fixed yaw-rate command (rad/s)")

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

    # Optionally LOCK the velocity command to a fixed value for a clean walking clip. The Play task
    # otherwise samples a random command (every 5 s) that may be near zero -> the robot steps in
    # place. Disable resampling and pin vel_command_b so the policy is driven forward and the obs
    # reflects it.
    _cmd_term = None
    if any(v is not None for v in (args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_wz)):
        _cmd_term = isaac_env.command_manager.get_term("base_velocity")
        _fixed_cmd = torch.tensor(
            [args_cli.cmd_vx or 0.0, args_cli.cmd_vy or 0.0, args_cli.cmd_wz or 0.0],
            device=isaac_env.device,
        )
        _cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)  # never resample during the clip
        if hasattr(_cmd_term, "is_standing_env"):
            _cmd_term.is_standing_env[:] = False
        _cmd_term.vel_command_b[:] = _fixed_cmd
        if hasattr(_cmd_term, "time_left"):
            _cmd_term.time_left[:] = 1.0e9
        print(f"[INFO] velocity command LOCKED to (vx, vy, wz) = {_fixed_cmd.tolist()}")

    rows = []  # list of dicts, one per recorded step
    traj_rows = []  # base pose + joint angles per step (for kinematic replay)
    joint_names = list(robot.data.joint_names)  # Isaac Lab joint ordering

    obs = env.get_observations()
    print(f"[INFO] Starting rollout for {args_cli.steps} steps ...")

    with torch.inference_mode():
        for step_idx in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if _cmd_term is not None:  # keep the command locked even across an episode reset
                _cmd_term.vel_command_b[:] = _fixed_cmd

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

            if args_cli.traj_out is not None:
                # root_state_w = [pos(3), quat wxyz(4), lin_vel(3), ang_vel(3)]
                root = robot.data.root_state_w[0, :7].cpu()       # pos + quat(wxyz)
                qpos = robot.data.joint_pos[0].cpu()              # (J,) Isaac Lab order
                trow = {
                    "step": step_idx,
                    "bpx": root[0].item(), "bpy": root[1].item(), "bpz": root[2].item(),
                    "bqw": root[3].item(), "bqx": root[4].item(),
                    "bqy": root[5].item(), "bqz": root[6].item(),
                }
                for j, name in enumerate(joint_names):
                    trow[f"q:{name}"] = qpos[j].item()
                traj_rows.append(trow)

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

    if args_cli.traj_out is not None and traj_rows:
        traj_path = os.path.expanduser(args_cli.traj_out)
        _td = os.path.dirname(traj_path)
        if _td:
            os.makedirs(_td, exist_ok=True)
        traj_fields = (["step", "bpx", "bpy", "bpz", "bqw", "bqx", "bqy", "bqz"]
                       + [f"q:{n}" for n in joint_names])
        with open(traj_path, "w", newline="") as fh:
            tw = csv.DictWriter(fh, fieldnames=traj_fields)
            tw.writeheader()
            tw.writerows(traj_rows)
        print(f"[INFO] Saved trajectory ({len(traj_rows)} steps, joints={joint_names}) to: {traj_path}")

        # Also dump the terrain mesh (world frame) so the Isaac-Gym kinematic replay can render the
        # actual rough ground the robot walked on (feet align: same world coords). The TerrainImporter
        # no longer keeps the trimesh (`.meshes` is deprecated/empty) and re-generating is NOT
        # reproducible (the custom rough-slope noise draws from the global NumPy RNG, cfg seed=None),
        # so read the imported mesh straight from USD. Flat envs have no terrain prim -> skip.
        import numpy as _np
        terrain = getattr(isaac_env.scene, "terrain", None)
        prim_paths = list(getattr(terrain, "terrain_prim_paths", []) or []) if terrain is not None else []
        if prim_paths:
            import omni.usd
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            vlist, flist, voff = [], [], 0
            for path in prim_paths:
                root = stage.GetPrimAtPath(path)
                if not root or not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    mesh = UsdGeom.Mesh(prim)
                    pts = _np.asarray(mesh.GetPointsAttr().Get(), dtype=_np.float64)
                    counts = _np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=_np.int64)
                    idx = _np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=_np.int64)
                    if len(pts) == 0 or len(idx) == 0:
                        continue
                    # local -> world (USD row-vector convention: p_world = [x,y,z,1] @ M)
                    M = _np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()), dtype=_np.float64).reshape(4, 4)
                    ptsw = (_np.c_[pts, _np.ones(len(pts))] @ M)[:, :3]
                    tris, o = [], 0  # triangulate (terrain is triangles, but stay general)
                    for c in counts:
                        for k in range(1, c - 1):
                            tris.append((idx[o], idx[o + k], idx[o + k + 1]))
                        o += int(c)
                    if not tris:
                        continue
                    vlist.append(ptsw.astype(_np.float32))
                    flist.append(_np.asarray(tris, dtype=_np.int64) + voff)
                    voff += len(ptsw)
            if vlist:
                verts = _np.concatenate(vlist, axis=0)
                faces = _np.concatenate(flist, axis=0)
                terr_path = os.path.splitext(traj_path)[0] + "_terrain.npz"
                _np.savez(terr_path, vertices=verts, faces=faces)
                print(f"[INFO] Saved terrain mesh ({len(verts)} verts, {len(faces)} faces, from USD) "
                      f"to: {terr_path}")
            else:
                print("[WARN] terrain prim(s) found but no UsdGeom.Mesh extracted; replay stays flat.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
