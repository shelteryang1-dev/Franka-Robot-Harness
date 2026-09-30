"""PPO task registration."""

import sys

import gymnasium as gym


TRAIN_TASK_ID = "Franka-Manipulation-Lift-PPO-v0"
PLAY_TASK_ID = "Franka-Manipulation-Lift-PPO-Play-v0"


def _register_once(task_id: str, environment_config: str) -> None:

    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": environment_config,
            "rsl_rl_cfg_entry_point": (
                "franka_manipulation_sim.rl.agent_cfg:FrankaLiftPPORunnerCfg"
            ),
        },
        disable_env_checker=True,
    )


def register_rl_tasks() -> list[str]:

    _register_once(
        TRAIN_TASK_ID,
        "franka_manipulation_sim.rl.env_cfg:FrankaLiftRLEnvCfg",
    )
    _register_once(
        PLAY_TASK_ID,
        "franka_manipulation_sim.rl.env_cfg:FrankaLiftRLEnvCfg_PLAY",
    )
    return sys.argv[1:]
