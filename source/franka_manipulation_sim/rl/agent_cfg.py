"""RSL-RL PPO configuration."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
    LiftCubePPORunnerCfg,
)


@configclass
class FrankaLiftPPORunnerCfg(LiftCubePPORunnerCfg):

    experiment_name = "franka_manipulation_lift_ppo"
