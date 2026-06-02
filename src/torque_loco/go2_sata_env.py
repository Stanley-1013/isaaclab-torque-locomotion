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

from . import sata_mdp
from .bio_actuator import BioActuatorCfg
from .growth import gompertz, control_freq

PHYSICS_HZ = 200.0
PHYSICS_DT = 1.0 / PHYSICS_HZ      # 0.005


class Go2SataEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Deployment override: SATA restores FULL capacity (G=1) at deployment/eval
        # (paper IV-B). Training uses the Gompertz schedule (growth_deploy_scale=None).
        self._deploy_G = getattr(cfg, "growth_deploy_scale", None)
        self._G = self._deploy_G if self._deploy_G is not None else gompertz(0)
        act = self.scene["robot"].actuators["base_legs"]
        if hasattr(act, "set_runtime"):
            act.set_runtime(PHYSICS_DT, self)

    def step(self, action):
        self._G = self._deploy_G if self._deploy_G is not None else gompertz(self.common_step_counter)
        freq = control_freq(self._G)              # 100 -> 200 Hz
        accum, n_sub = 0.0, 0
        while accum * freq < 1.0:
            n_sub += 1
            accum += PHYSICS_DT
        n_sub = max(1, n_sub)
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
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

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
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


@configclass
class Go2SataEnvCfg(UnitreeGo2FlatEnvCfg):
    # Deployment growth scalar. None -> use the Gompertz schedule (training). A float (e.g. 1.0)
    # forces FULL capacity at every step, matching SATA's deployment ("restore f_end, tau_end").
    growth_deploy_scale: float | None = None

    def __post_init__(self):
        super().__post_init__()
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
        p = self.observations.policy
        p.base_lin_vel.scale = 2.0; p.base_ang_vel.scale = 0.25
        p.joint_pos.scale = 1.0; p.joint_vel.scale = 0.05
        p.height_scan = None
        p.actions = None
        p.applied_torque = ObservationTermCfg(func=sata_mdp.applied_torque)
        p.motor_fatigue = ObservationTermCfg(func=sata_mdp.motor_fatigue)
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
        # SATA terminations = flip-over (primary) + time_out (inherited). NOT base contact
        # (robot starts prone at z=0.10). joint_pos_out_of_limit is intentionally NOT used:
        # it tests the 0.9-scaled SOFT limits, and SATA's folded start (calf -2.5) sits at that
        # soft-limit edge -> it would fire instantly. The soft_dof_pos_limits REWARD penalty
        # (weight -5) supplies the joint-limit gradient instead (platform-difference vs SATA's
        # hard-limit reset).
        self.terminations.base_contact = None
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.4})
        E = self.events
        E.push_robot = EventTermCfg(
            func=sata_mdp.push_scaled_by_growth, mode="interval", interval_range_s=(4.0, 4.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5),
                                       "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}},
        )
        if hasattr(E, "add_base_mass") and E.add_base_mass is not None:
            E.add_base_mass.params["mass_distribution_params"] = (-1.0, 5.0)


@configclass
class Go2SataEnvCfg_PLAY(Go2SataEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # Deploy at FULL capacity (G=1): torque ceiling 23.5 N·m, 200 Hz — SATA's deployment
        # setting. (Without this, eval ran the policy in the crippled "infant" body, G~=0.13.)
        self.growth_deploy_scale = 1.0
