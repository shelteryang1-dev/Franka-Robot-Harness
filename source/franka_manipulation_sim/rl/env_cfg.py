"""Env cfg."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
    FrankaCubeLiftEnvCfg_PLAY,
)

from ..asset_compat import apply_franka_asset_compat


@configclass
class FrankaLiftRLEnvCfg(FrankaCubeLiftEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        apply_franka_asset_compat(self)


@configclass
class FrankaLiftRLEnvCfg_PLAY(FrankaCubeLiftEnvCfg_PLAY):

    def __post_init__(self):
        super().__post_init__()
        apply_franka_asset_compat(self)
