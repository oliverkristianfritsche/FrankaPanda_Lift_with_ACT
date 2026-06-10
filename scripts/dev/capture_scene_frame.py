# Oliver Fritsche
# June 10, 2026
# CS 7180 Advanced Perception

"""Capture one high-resolution frame from the scene camera pose (for figures).

  /isaac-sim/python.sh scripts/dev/capture_scene_frame.py --out /tmp/scene_hires.png \
      --height 960 --width 1280
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--height", type=int, default=960)
p.add_argument("--width", type=int, default=1280)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import imageio  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

cfg = load_cfg_from_registry("Template-Lift-Cube-Franka-Demo-Baseline-v0", "env_cfg_entry_point")
cfg.scene.num_envs = 1
# Same pose and optics as the policy's scene camera, just more pixels.
cfg.scene.camera.height = a.height
cfg.scene.camera.width = a.width
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
for _ in range(6):
    env.sim.step(render=True)
env.scene.update(env.physics_dt)
rgb = env.scene.sensors["camera"].data.output["rgb"][0, ..., :3]
if rgb.dtype == torch.float32:
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
imageio.imwrite(a.out, rgb.cpu().numpy())
print(f"WROTE {a.out}", flush=True)
import os
os._exit(0)
