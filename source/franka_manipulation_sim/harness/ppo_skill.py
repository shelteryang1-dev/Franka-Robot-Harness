"""Frozen PPO lift skill."""

import torch
from isaaclab.utils.math import subtract_frame_transforms


class LiftPolicy:
    """Load the frozen lift policy."""

    def __init__(self, path, raw):
        self.raw = raw
        state = torch.load(path, map_location=raw.device, weights_only=False)["actor_state_dict"]
        self.network = torch.nn.Sequential(torch.nn.Linear(36, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 8)).to(raw.device).eval()
        weights = {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}
        self.network.load_state_dict(weights, strict=True)
        self.previous = torch.zeros((1, 8), device=raw.device)

    def reset(self, obj):
        self.previous.zero_()
        self.goal = self.raw.command_manager.get_command("object_pose").clone()
        self.goal[:, :3] = obj
        self.goal[:, 2] = .3

    def compute(self, object_world):
        data = self.raw.scene["robot"].data
        obj, _ = subtract_frame_transforms(data.root_pos_w.torch, data.root_quat_w.torch, object_world)
        obs = torch.cat([data.joint_pos.torch-data.default_joint_pos.torch,
                         data.joint_vel.torch-data.default_joint_vel.torch,
                         obj, self.goal, self.previous], dim=-1)
        action = self.network(obs)
        if not torch.isfinite(action).all():
            raise ValueError("Policy output non-finite")
        self.previous = action.detach().clone()
        # Offset by the default pose, not the current joints.
        joints = action[:, :7]*.5+data.default_joint_pos.torch[:, :7]
        return joints, float(action[0, 7])
