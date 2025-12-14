# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Template-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Template-Lift-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Visual Randomization Conditions for ACT Robustness Experiments
##

# Condition 1: Baseline (fixed lighting, textures, camera)
gym.register(
    id="Template-Lift-Cube-Franka-Demo-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.visual_randomization:FrankaCubeLiftEnvCfg_Demo_Baseline",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# Condition 2: Lighting randomization (intensity, direction, color temperature)
gym.register(
    id="Template-Lift-Cube-Franka-Demo-Lighting-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.visual_randomization:FrankaCubeLiftEnvCfg_Demo_Lighting",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# Condition 3: Texture randomization (block materials, table surface)
gym.register(
    id="Template-Lift-Cube-Franka-Demo-Texture-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.visual_randomization:FrankaCubeLiftEnvCfg_Demo_Texture",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# Condition 4: Combined (all visual randomizations)
gym.register(
    id="Template-Lift-Cube-Franka-Demo-Combined-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.visual_randomization:FrankaCubeLiftEnvCfg_Demo_Combined",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
