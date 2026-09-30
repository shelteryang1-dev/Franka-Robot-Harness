"""Franka asset compatibility."""

FRANKA_6_0_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
    "Robots/FrankaRobotics/FrankaPanda/franka.usd"
)


def apply_franka_asset_compat(env_cfg) -> tuple[str, str]:
    """Override the asset path without changing Isaac Lab."""

    original_path = env_cfg.scene.robot.spawn.usd_path
    env_cfg.scene.robot.spawn.usd_path = FRANKA_6_0_USD_PATH
    return original_path, env_cfg.scene.robot.spawn.usd_path
