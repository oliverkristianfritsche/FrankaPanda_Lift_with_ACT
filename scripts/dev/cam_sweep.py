"""One-shot wrist-camera orientation sweep.

Adds 6 diagnostic cameras on panda_hand (looking +X/-X/+Y/-Y/+Z/-Z in the 'world'
convention), rolls out the PPO policy to the lift pose, and saves a frame from each
+ the scene cam. We pick the look-direction that frames the gripper + cube, then set
make_wrist_camera() to it.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=170)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch
import numpy as np
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # noqa: F401
import Lift.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1

# (w, x, y, z) rotations; 'world' convention default optical axis is +X.
DIRS = {
    "px": (1.0, 0.0, 0.0, 0.0),        # look +X
    "nx": (0.0, 0.0, 0.0, 1.0),        # look -X  (180 about Z)
    "py": (0.7071, 0.0, 0.0, 0.7071),  # look +Y  (+90 about Z)
    "ny": (0.7071, 0.0, 0.0, -0.7071), # look -Y  (-90 about Z)
    "pz": (0.7071, 0.0, -0.7071, 0.0), # look +Z  (-90 about Y)
    "nz": (0.7071, 0.0, 0.7071, 0.0),  # look -Z  (+90 about Y)
}
for nm, rot in DIRS.items():
    setattr(cfg.scene, "sweep_" + nm, CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/sweep_" + nm,
        update_period=0.0, height=240, width=320, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.01, 4.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=rot, convention="world"),
    ))

env = ManagerBasedRLEnv(cfg=cfg)

acfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
acfg["trainer"]["close_environment_at_exit"] = False
acfg["agent"]["experiment"]["write_interval"] = 0
acfg["agent"]["experiment"]["checkpoint_interval"] = 0
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
runner = Runner(wrapped, acfg)
runner.agent.load(a.checkpoint)
runner.agent.set_running_mode("eval")

obs, _ = wrapped.reset()
for t in range(a.steps):
    with torch.no_grad():
        out = runner.agent.act(obs, timestep=0, timesteps=0)
        if len(out) > 2 and isinstance(out[2], dict) and out[2].get("mean_actions") is not None:
            act = out[2]["mean_actions"]
        else:
            act = out[0]
    obs, _, _, _, _ = wrapped.step(act)


def save(camname, path):
    cam = env.scene.sensors[camname]
    rgb = cam.data.output["rgb"]
    img = rgb[0, ..., :3]
    if img.dtype == torch.float32:
        img = (img * 255).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img.cpu().numpy()).save(path)
    print("saved", path)


for nm in DIRS:
    save("sweep_" + nm, "/workspace/sweep_" + nm + ".png")
save("camera", "/workspace/sweep_scene.png")
print("SWEEP_DONE")
app.close()
