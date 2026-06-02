# src/torque_loco/go2_sata_env.py
"""Go2SataEnv: manager-based Go2 velocity env with SATA's full bio stack.

Overrides step() to (a) update the Gompertz growth scalar env._G each step and (b) run a
variable-frequency physics loop (control 100->200 Hz over training) instead of fixed decimation.
All other managers are stock. The cfg installs the BioActuator, SATA reward/obs/event terms,
SATA command ranges, SATA defaults (base z=0.10, thigh 1.45, calf -2.5), and 200 Hz physics.

step() body mirrors isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv.step
(IsaacLab source/isaaclab/isaaclab/envs/manager_based_rl_env.py, lines 153-240).
The ONLY change is the fixed `for _ in range(self.cfg.decimation):` loop -> `for _ in range(n_sub):`.
Render gating keeps the stock `self._sim_step_counter % self.cfg.sim.render_interval` test, which
remains correct under a variable sub-step count and is skipped entirely under headless training
(is_rendering is False).
"""
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from . import sata_mdp
from .bio_actuator import BioActuatorCfg
from .growth import gompertz, control_freq
from .sata_terrain import SATA_TERRAINS_CFG

PHYSICS_HZ = 200.0
PHYSICS_DT = 1.0 / PHYSICS_HZ      # 0.005


class Go2SataEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Deployment override: SATA restores FULL capacity (G=1) at deployment/eval
        # (paper IV-B). Training uses the Gompertz schedule (growth_deploy_scale=None).
        self._deploy_G = getattr(cfg, "growth_deploy_scale", None)
        # SATA's growth counter `step_count` increments once per PHYSICS SUBSTEP (it lives in
        # _update_growth_scale, called inside the substep loop), NOT per env.step. Since n_sub=2
        # for the entire growth phase (n_sub drops to 1 only at G=1/200Hz), SATA's curriculum
        # advances 2x faster (in env-steps) than a per-env-step counter would. Track substeps.
        self._growth_step = 0
        self._G = self._deploy_G if self._deploy_G is not None else gompertz(0)
        act = self.scene["robot"].actuators["base_legs"]
        if hasattr(act, "set_runtime"):
            act.set_runtime(PHYSICS_DT, self)

    def step(self, action):
        self._G = self._deploy_G if self._deploy_G is not None else gompertz(self._growth_step)
        freq = control_freq(self._G)              # 100 -> 200 Hz
        accum, n_sub = 0.0, 0
        while accum * freq < 1.0:
            n_sub += 1
            accum += PHYSICS_DT
        n_sub = max(1, n_sub)
        if self._deploy_G is None:
            self._growth_step += n_sub            # SATA: step_count += 1 per substep
        return self._stepped(action, n_sub)

    def _stepped(self, action, n_sub):
        """Mirror of ManagerBasedRLEnv.step with the fixed decimation loop replaced by n_sub."""
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping (variable count: SATA control freq grows 100 -> 200 Hz)
        for _ in range(n_sub):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # SATA runs post_physics_step (counters, reward, termination) once per physics substep, so
        # at cold start (n_sub=2) episode length and reward accumulate over 2 substeps. We compute
        # once per env.step but scale by the substep count: reward/command/event dt = n_sub*dt
        # (== summing n_sub substeps, since state barely changes over n_sub*0.005s) and episode
        # length advances by n_sub (so the 10s episode horizon matches SATA in sim time).
        eff_dt = n_sub * PHYSICS_DT
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += n_sub  # advance by substeps (SATA episode clock is per-substep)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=eff_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=eff_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=eff_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


def _apply_sata_bio(self):
    """Install SATA's full bio stack (actuator, obs, rewards, terminations, events, defaults)
    onto a LocomotionVelocity-derived cfg. Terrain is set by each subclass (flat / SATA-slopes /
    Isaac-Lab-default-rough) BEFORE calling this — the bio stack is terrain-agnostic."""
    self.sim.dt = PHYSICS_DT
    # decimation=1 -> step_dt = physics_dt = 0.005s regardless of the variable n_sub set in
    # Go2SataEnv.step(). Intentional: at cold start (n_sub=2, 100Hz control) each env step
    # covers 0.010s of physics but rewards/events use step_dt=0.005s, halving reward scale
    # early — consistent with SATA's growth-based cold-start stabilization.
    self.decimation = 1
    self.sim.render_interval = 4
    self.episode_length_s = 10.0
    self.scene.robot.init_state.pos = (0.0, 0.0, 0.10)
    self.scene.robot.init_state.joint_pos = {
        "FL_hip_joint": 0.1, "RL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 1.45, "RL_thigh_joint": 1.45, "FR_thigh_joint": 1.45, "RR_thigh_joint": 1.45,
        "FL_calf_joint": -2.5, "RL_calf_joint": -2.5, "FR_calf_joint": -2.5, "RR_calf_joint": -2.5,
    }
    self.actions.joint_pos = None
    self.actions.joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0,
    )
    self.scene.robot.actuators["base_legs"] = BioActuatorCfg(
        joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
        effort_limit=1000.0, velocity_limit=30.0,
    )
    cr = self.commands.base_velocity.ranges
    cr.lin_vel_x = (-0.5, 1.5); cr.lin_vel_y = (-0.5, 0.5); cr.ang_vel_z = (-1.5, 1.5)
    self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
    # SATA samples ang_vel_yaw directly (heading_command=False) and has no standing-still envs.
    self.commands.base_velocity.heading_command = False
    self.commands.base_velocity.rel_standing_envs = 0.0
    # SATA scales command ranges by the growth scalar G during resampling (lin_vel_x narrows to
    # its midpoint early, vy/yaw -> 0 early, opening up as G grows). Swap in the growth command.
    self.commands.base_velocity.class_type = sata_mdp.GrowthVelocityCommand
    p = self.observations.policy
    p.base_lin_vel.scale = 2.0; p.base_ang_vel.scale = 0.25
    p.joint_pos.scale = 1.0; p.joint_vel.scale = 0.05
    # SATA does NOT feed terrain heights to the policy (num_observations=60, no height scan).
    # The height_scanner SENSOR (if present on rough terrain) is still used by the base_height
    # reward for a terrain-relative head height — it is just not an observation.
    p.height_scan = None
    p.actions = None
    p.applied_torque = ObservationTermCfg(func=sata_mdp.applied_torque)
    p.motor_fatigue = ObservationTermCfg(func=sata_mdp.motor_fatigue)
    # SATA clips observations to +/-clip_observations (=100) every step (legged_robot.step ->
    # torch.clip(obs_buf, -100, 100)). Without it, a transient torque/velocity spike on rough
    # terrain feeds an unbounded value to the policy -> network blow-up -> NaN. Match SATA: clip
    # every policy obs term. (Isaac Lab clips per-term before scaling; bound is what matters here.)
    for _term in vars(p).values():
        if isinstance(_term, ObservationTermCfg):
            _term.clip = (-100.0, 100.0)
    R = self.rewards
    for name in list(vars(R)):
        setattr(R, name, None)
    R.track_x = RewardTermCfg(func=sata_mdp.track_x, weight=10.0)
    R.track_y = RewardTermCfg(func=sata_mdp.track_y, weight=5.0)
    R.track_yaw = RewardTermCfg(func=sata_mdp.track_yaw, weight=5.0)
    R.base_height = RewardTermCfg(func=sata_mdp.base_height, weight=5.0)
    R.roll = RewardTermCfg(func=sata_mdp.roll_penalty, weight=-5.0)
    R.lin_vel_z = RewardTermCfg(func=sata_mdp.lin_vel_z, weight=-5.0)
    R.joint_limits = RewardTermCfg(func=sata_mdp.soft_dof_pos_limits, weight=-5.0)
    R.fatigue = RewardTermCfg(func=sata_mdp.fatigue_penalty, weight=-0.05)
    R.joint_acc = RewardTermCfg(func=sata_mdp.joint_acc_l2, weight=-1e-6)
    # SATA check_termination = flip-over (proj_grav_z>0) + HARD joint-limit ±0.05 + time_out.
    # base_contact is dropped (SATA's torque env overrides check_termination and never tests
    # contact; robot also starts prone). bad_orientation(1.4 rad) ≈ SATA's flip test. The
    # hard-limit termination uses SATA's go2_torque.urdf limits, so the folded calf start does NOT
    # fire (it's inside the hard range) — unlike the 0.9-soft-limit check we previously dropped.
    self.terminations.base_contact = None
    self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.4})
    self.terminations.joint_hard_limit = DoneTerm(func=sata_mdp.joint_pos_hard_limit)
    E = self.events
    E.push_robot = EventTermCfg(
        func=sata_mdp.push_scaled_by_growth, mode="interval", interval_range_s=(4.0, 4.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5),
                                   "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}},
    )
    # SATA domain_rand: friction U[0.5,1.25]; added base mass U[-1,5] + COM shift x±0.2,y/z±0.1;
    # reset dof = default*U(0.95,1.05); reset base shifted ±1 m in xy (no yaw/vel randomization).
    if hasattr(E, "physics_material") and E.physics_material is not None:
        E.physics_material.params["static_friction_range"] = (0.5, 1.25)
        E.physics_material.params["dynamic_friction_range"] = (0.5, 1.25)
    if hasattr(E, "add_base_mass") and E.add_base_mass is not None:
        E.add_base_mass.params["mass_distribution_params"] = (-1.0, 5.0)
    # re-create base_com (the rough cfg sets it to None): SATA shifts base COM x±0.2, y/z±0.1.
    E.base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}},
    )
    if hasattr(E, "reset_robot_joints") and E.reset_robot_joints is not None:
        E.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        E.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if hasattr(E, "reset_base") and E.reset_base is not None:
        E.reset_base.params["pose_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        E.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}


@configclass
class Go2SataEnvCfg(UnitreeGo2FlatEnvCfg):
    """SATA on FLAT ground. Kept as the flat baseline (NOT SATA-faithful terrain — SATA trains on
    rough slopes; see Go2SataRoughEnvCfg). Behaviour is unchanged from the original migration."""
    # Deployment growth scalar. None -> use the Gompertz schedule (training). A float (e.g. 1.0)
    # forces FULL capacity at every step, matching SATA's deployment ("restore f_end, tau_end").
    growth_deploy_scale: float | None = None

    def __post_init__(self):
        super().__post_init__()
        _apply_sata_bio(self)


@configclass
class Go2SataRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """SATA-FAITHFUL terrain: trimesh rough SLOPES (0.2 smooth + 0.8 rough), curriculum OFF —
    matching SATA's terrain_proportions=[0.2,0.8,0,0,0] & curriculum=False. Inherits the rough
    cfg's height_scanner sensor (used by the terrain-relative base_height reward, not by obs)."""
    growth_deploy_scale: float | None = None

    def __post_init__(self):
        super().__post_init__()
        # Replace Isaac Lab's default rough terrain (steep slopes + stairs + boxes, curriculum)
        # with SATA's gentle-slopes-only terrain, no curriculum.
        self.scene.terrain.terrain_generator = SATA_TERRAINS_CFG
        self.curriculum.terrain_levels = None   # base __post_init__ keys terrain curriculum off this
        _apply_sata_bio(self)


@configclass
class Go2SataDefaultRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """SATA bio stack on Isaac Lab's DEFAULT rough terrain (steep slopes + stairs + boxes +
    difficulty curriculum). Harder than SATA's terrain — a second comparison point, not a faithful
    SATA repro. Terrain curriculum stays ON (inherited)."""
    growth_deploy_scale: float | None = None

    def __post_init__(self):
        super().__post_init__()
        _apply_sata_bio(self)


@configclass
class Go2SataEnvCfg_PLAY(Go2SataEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)


@configclass
class Go2SataRoughEnvCfg_PLAY(Go2SataRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5


@configclass
class Go2SataDefaultRoughEnvCfg_PLAY(Go2SataDefaultRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


def _play_overrides(self):
    """Shared PLAY/eval overrides: small scene, no corruption/push, deploy at FULL capacity (G=1)
    — SATA's deployment setting (torque ceiling 23.5 N·m, 200 Hz). Without this, eval ran the
    policy in the crippled "infant" body (G~=0.13)."""
    self.scene.num_envs = 50
    self.observations.policy.enable_corruption = False
    self.events.push_robot = None
    self.growth_deploy_scale = 1.0
