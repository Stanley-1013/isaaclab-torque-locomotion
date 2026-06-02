# src/torque_loco/__register__.py
"""Register the torque-control Go2 tasks with Gymnasium.

Importing this module performs ONLY ``gym.register`` calls with *string* entry
points. It deliberately does NOT import ``go2_torque_env_cfg`` (which pulls in
``isaaclab`` -> ``pxr``): registration must run before the Omniverse app is
launched, and the cfg is imported lazily by ``gym.make`` afterwards.

The rsl_rl PPO runner is reused from the stock Go2 flat task.
"""

import gymnasium as gym

_RSL_RL_PPO = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg"
)

# SATA-matched PPO: the FULL SATA bio task needs SATA's actor/critic net [512,256,128].
# Isaac Lab's *Flat* runner shrinks it to [128,128,128] (fine for easy position control on
# flat ground, too small for torque control) — so we reuse Isaac Lab's *Rough* runner, whose
# [512,256,128] + lr/entropy/clip/gamma/lam/KL/steps already EQUAL SATA's GO2TorqueCfgPPO.
# (No invented hyperparameters — Isaac Lab's own cfg that matches SATA.)
_RSL_RL_PPO_SATA = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Torque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_torque_env_cfg:Go2TorqueEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Torque-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_torque_env_cfg:Go2TorqueEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO,
    },
)

# --- Full faithful SATA task (custom env class with variable-frequency step) ---
# entry_point is the custom env class itself (not ManagerBasedRLEnv); gym.make instantiates it.
gym.register(
    id="Isaac-Velocity-Flat-Go2-Sata-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO_SATA,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sata-Play-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO_SATA,
    },
)

# --- SATA-FAITHFUL terrain: rough SLOPES (0.2 smooth + 0.8 rough), curriculum OFF.
# This is the control-variable-correct repro (SATA trains on trimesh rough slopes, not flat). ---
gym.register(
    id="Isaac-Velocity-Rough-Go2-Sata-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataRoughEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO_SATA,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sata-Play-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO_SATA,
    },
)

# --- Isaac Lab DEFAULT rough terrain (steeper slopes + stairs + boxes + curriculum): a second,
# harder comparison point (NOT SATA-faithful) for the terrain ablation. ---
gym.register(
    id="Isaac-Velocity-Rough-Go2-Sata-Default-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataDefaultRoughEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO_SATA,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sata-Default-Play-v0",
    entry_point="torque_loco.go2_sata_env:Go2SataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "torque_loco.go2_sata_env:Go2SataDefaultRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_PPO_SATA,
    },
)
