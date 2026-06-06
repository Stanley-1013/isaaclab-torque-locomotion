#!/usr/bin/env python
"""Kinematically replay an Isaac-Lab-trained Go2 trajectory in Isaac Gym and record an mp4.

WHY: Isaac Lab's RTX/Hydra renderer crashes on this container (driver 595 / IOMMU; rtx.scenedb
plugin), so we can't record from Isaac Lab directly. Isaac Gym's older GL renderer DOES work here
(SATA recorded videos this way). We therefore replay the recorded base-pose + joint-angle
trajectory (from eval_metrics.py --traj_out) on the Go2 URDF in a minimal Isaac Gym sim — no
physics/policy, just set states each frame and render an off-screen camera.

Run in the SATA conda env (which has isaacgym + imageio), with the VNC display:
    source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh  # or: conda activate sata
    DISPLAY=:0 python scripts/render_replay_isaacgym.py \
        --traj results/traj_sata_s1.csv --out results/go2_sata_s1_walk.mp4 --gpu 0

Quaternion note: the trajectory stores Isaac Lab's (w,x,y,z); Isaac Gym root state wants (x,y,z,w).
Joints are matched BY NAME (Isaac Lab order in the CSV header -> Isaac Gym DOF order).
"""
from __future__ import annotations

import argparse
import csv
import os

from isaacgym import gymapi, gymtorch  # must import before torch
import numpy as np
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

URDF_DIR = os.path.expanduser("~/workspace/SATA/legged_gym/resources/robots/go2/urdf")
URDF_FILE = "go2.urdf"


def load_traj(path):
    """Return (steps, joint_names, base[N,7] wxyz, q[N,J] in CSV/Isaac-Lab order)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty trajectory CSV: {path}")
    joint_names = [k[2:] for k in rows[0] if k.startswith("q:")]
    base = np.array([[float(r[c]) for c in ("bpx", "bpy", "bpz", "bqw", "bqx", "bqy", "bqz")]
                     for r in rows], dtype=np.float32)
    q = np.array([[float(r[f"q:{n}"]) for n in joint_names] for r in rows], dtype=np.float32)
    return len(rows), joint_names, base, q


def main():
    ap = argparse.ArgumentParser(description="Replay an Isaac-Lab Go2 trajectory in Isaac Gym -> mp4.")
    ap.add_argument("--traj", required=True, help="trajectory CSV from eval_metrics.py --traj_out")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth trajectory step")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--terrain", default=None,
                    help="optional terrain mesh .npz (vertices,faces) from eval_metrics --traj_out; "
                         "renders the actual rough ground instead of a flat plane")
    args = ap.parse_args()

    n_steps, traj_joint_names, base, q = load_traj(args.traj)
    print(f"[INFO] trajectory: {n_steps} steps, {len(traj_joint_names)} joints: {traj_joint_names}")

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 200.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)   # kinematic replay -> no gravity
    sim_params.use_gpu_pipeline = False               # CPU pipeline: simplest for set_*_tensor
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.use_gpu = False

    sim = gym.create_sim(args.gpu, args.gpu, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise SystemExit("create_sim failed")

    if args.terrain:
        # Render the actual Isaac-Lab rough terrain (world frame == trajectory frame, so feet align).
        tdata = np.load(os.path.expanduser(args.terrain))
        tverts = np.ascontiguousarray(tdata["vertices"], dtype=np.float32)
        tfaces = np.ascontiguousarray(tdata["faces"], dtype=np.uint32)
        tm = gymapi.TriangleMeshParams()
        tm.nb_vertices = tverts.shape[0]
        tm.nb_triangles = tfaces.shape[0]
        gym.add_triangle_mesh(sim, tverts.flatten(), tfaces.flatten(), tm)
        print(f"[INFO] terrain mesh: {tverts.shape[0]} verts, {tfaces.shape[0]} tris")
    else:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(sim, plane)

    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = False
    asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_opts.collapse_fixed_joints = True
    # MUST match legged_gym's Go2 load or the robot renders "disassembled": the Unitree .dae
    # meshes are y-up and must be flipped to z-up. Without flip_visual_attachments the visual
    # meshes attach mis-oriented even though joint STATES are correct.
    asset_opts.flip_visual_attachments = True
    asset_opts.replace_cylinder_with_capsule = True
    asset = gym.load_asset(sim, URDF_DIR, URDF_FILE, asset_opts)
    dof_names = gym.get_asset_dof_names(asset)
    print(f"[INFO] Isaac Gym DOF order: {dof_names}")

    # map: Isaac Gym dof index -> column index in the trajectory's joint order (by name)
    name_to_traj = {n: i for i, n in enumerate(traj_joint_names)}
    missing = [n for n in dof_names if n not in name_to_traj]
    if missing:
        raise SystemExit(f"joint-name mismatch; trajectory missing {missing}")
    remap = [name_to_traj[n] for n in dof_names]

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    start = gymapi.Transform()
    start.p = gymapi.Vec3(0.0, 0.0, 0.4)
    actor = gym.create_actor(env, asset, start, "go2", 0, 1)

    # Visual styling so this Isaac-Lab-reproduction clip is distinguishable at a glance from the
    # original SATA Isaac-Gym videos (same renderer otherwise). Tint the robot teal and use a cool
    # key light. Purely cosmetic — joint states / terrain are untouched.
    robot_color = gymapi.Vec3(0.12, 0.62, 0.70)
    for bi in range(gym.get_actor_rigid_body_count(env, actor)):
        gym.set_rigid_body_color(env, actor, bi, gymapi.MESH_VISUAL, robot_color)
    # cool directional light (intensity, ambient, direction)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.85, 0.9, 1.05),
                             gymapi.Vec3(0.32, 0.34, 0.42), gymapi.Vec3(0.6, 0.4, -1.0))

    cam_props = gymapi.CameraProperties()
    cam_props.width = args.width
    cam_props.height = args.height
    cam_props.enable_tensors = False
    cam = gym.create_camera_sensor(env, cam_props)

    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))  # (1,13)
    dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))          # (J,2): pos,vel
    n_dof = dof.shape[0]

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8)

    for i in range(n_steps):
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        # set base pose: pos(3) + quat(xyzw); zero velocities
        root[0, 0:3] = torch.from_numpy(base[i, 0:3])
        root[0, 3] = float(base[i, 4])  # qx
        root[0, 4] = float(base[i, 5])  # qy
        root[0, 5] = float(base[i, 6])  # qz
        root[0, 6] = float(base[i, 3])  # qw
        root[0, 7:13] = 0.0
        # set joint angles (remapped to Isaac Gym DOF order); zero velocities
        qg = torch.from_numpy(q[i, remap])
        dof[:n_dof, 0] = qg
        dof[:n_dof, 1] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if i % args.stride != 0:
            continue
        b = base[i, 0:3]
        gym.set_camera_location(
            cam, env,
            gymapi.Vec3(b[0] - 1.25, b[1] - 1.25, b[2] + 0.65),
            gymapi.Vec3(b[0], b[1], b[2] + 0.1),
        )
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
        img = img.reshape(args.height, args.width, 4)[:, :, :3]
        writer.append_data(np.ascontiguousarray(img))

    writer.close()
    gym.destroy_sim(sim)
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()
