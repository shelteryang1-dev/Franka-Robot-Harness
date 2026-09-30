"""Env cfg."""

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

from ..asset_compat import apply_franka_asset_compat
from ..state_machine_termination import state_machine_finished


@configclass
class FrankaManipulationEnvCfg(FrankaCubeLiftEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        apply_franka_asset_compat(self)

        self.observations.policy.enable_corruption = False

        self.episode_length_s = 5.0

        self.terminations.state_machine_finished = DoneTerm(func=state_machine_finished, time_out=False)


def configure_randomization(
    env_cfg: FrankaManipulationEnvCfg,
    object_xy_range: float,
    target_xy_range: float,
    target_z: float,
    mass_scale_range: float,
) -> None:

    object_range = max(0.0, float(object_xy_range))
    target_range = max(0.0, float(target_xy_range))
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (-object_range, object_range),
        "y": (-object_range, object_range),
        "z": (0.0, 0.0),
    }
    env_cfg.commands.object_pose.ranges.pos_x = (0.5 - target_range, 0.5 + target_range)
    env_cfg.commands.object_pose.ranges.pos_y = (-target_range, target_range)
    env_cfg.commands.object_pose.ranges.pos_z = (float(target_z), float(target_z))

    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

    mass_range = min(max(0.0, float(mass_scale_range)), 0.95)
    env_cfg.events.randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0 - mass_range, 1.0 + mass_range),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
