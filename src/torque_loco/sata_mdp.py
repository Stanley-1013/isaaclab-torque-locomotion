# src/torque_loco/sata_mdp.py
"""Custom SATA manager terms: observations (torque, fatigue), the 9 SATA reward terms
(growth-modulated via env._G), and the G-scaled push event.
phi(x) = exp(-4|x|) (SATA Gaussian-shaped tracking kernel; == exp(-|x|/0.25))."""
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _phi(x):
    return torch.exp(-4.0 * x.abs())

def _G(env):
    return float(getattr(env, "_G", 1.0))

def _actuator(env, name="base_legs"):
    return env.scene["robot"].actuators[name]

# ---- observations ----
def applied_torque(env, asset_cfg=SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]

def motor_fatigue(env, asset_cfg=SceneEntityCfg("robot")):
    return _actuator(env).motor_fatigue

# ---- reward terms (return (num_envs,)) ----
def track_x(env, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command(command_name)
    g = _G(env)
    rng = env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x
    mid = 0.5 * (rng[0] + rng[1])
    # SATA's CODE (_reward_forward) is a SINGLE exp of a growth-blended target (ceiling 1.0),
    # NOT the paper Table II two-term sum (which would have ceiling ~2.0 and over-weight forward
    # vs head_height -> low crawl). Match the code: target = mid*max(1-2G,0) + cmd*min(2G,1).
    target = mid * max(1.0 - 2.0 * g, 0.0) + cmd[:, 0] * min(2.0 * g, 1.0)
    return _phi(vx - target)

def track_y(env, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _phi(asset.data.root_lin_vel_b[:, 1] - cmd[:, 1]) * _G(env)

def track_yaw(env, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _phi(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2]) * _G(env)

def base_height(env, target_height=0.3, asset_cfg=SceneEntityCfg("robot")):
    # Matches SATA _reward_head_height exactly: base_height*(1+G) + head_up,
    # head_up = -(grav_x.clip(min=m)), m = min(0, -0.2*(1.5-2G)).
    # SATA: base_height = mean(root_z - measured_heights) (terrain-RELATIVE). On rough terrain we
    # subtract the height-scanner ground height; on flat (no scanner) this is just root_z.
    asset = env.scene[asset_cfg.name]
    g = _G(env)
    height = asset.data.root_pos_w[:, 2]
    scanner = getattr(env.scene, "sensors", {}).get("height_scanner")
    if scanner is not None:
        # RayCaster returns +/-inf for rays that miss the mesh (an env can roam to the terrain-map
        # edge over a long rollout). SATA's _get_heights is always finite, so an inf here is a
        # platform difference, not SATA behaviour — and a single inf poisons the reward -> NaN ->
        # PPO std NaN -> crash. Average only the rays that actually hit; fall back to root-z if all
        # miss (so base_height stays finite, matching SATA's always-finite measured_heights).
        ground = scanner.data.ray_hits_w[..., 2]
        finite = torch.isfinite(ground)
        denom = finite.sum(dim=1).clamp(min=1)
        ground_mean = torch.where(finite, ground, torch.zeros_like(ground)).sum(dim=1) / denom
        height = height - ground_mean
    bh = torch.clamp(height, max=target_height)
    gx = asset.data.projected_gravity_b[:, 0]
    m = min(0.0, -0.2 * (1.5 - 2.0 * g))
    head_up = -torch.clamp(gx, min=m)
    return bh * (1.0 + g) + head_up

def roll_penalty(env, asset_cfg=SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.projected_gravity_b[:, 1].abs()

def lin_vel_z(env, asset_cfg=SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.root_lin_vel_b[:, 2] ** 2

def soft_dof_pos_limits(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    lo = asset.data.soft_joint_pos_limits[..., 0]
    hi = asset.data.soft_joint_pos_limits[..., 1]
    out = -(q - lo).clamp(max=0.0) + (q - hi).clamp(min=0.0)
    return out.sum(dim=1)

def fatigue_penalty(env, kappa_scale=5.0, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    fatigue = _actuator(env).motor_fatigue
    action_scaled = (asset.data.joint_effort_target * kappa_scale).abs()
    return (fatigue * action_scaled).sum(dim=1)

def joint_acc_l2(env, asset_cfg=SceneEntityCfg("robot")):
    return (env.scene[asset_cfg.name].data.joint_acc ** 2).sum(dim=1)

# ---- events ----
def push_scaled_by_growth(env, env_ids, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    """Push by setting base velocity, magnitude scaled by env._G (SATA max_push_vel*general_scale)."""
    asset: Articulation = env.scene[asset_cfg.name]
    g = _G(env)
    vel = torch.zeros((len(env_ids), 6), device=env.device)
    for i, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        if key in velocity_range:
            lo, hi = velocity_range[key]
            vel[:, i] = (torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo) * g
    root = asset.data.root_state_w[env_ids].clone()
    root[:, 7:13] += vel
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
