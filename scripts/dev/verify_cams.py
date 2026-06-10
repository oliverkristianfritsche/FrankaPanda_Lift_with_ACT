"""Render BOTH production cameras (wrist + scene) at the lifted pose, to verify the
D455-intrinsics wrist cam and the rig-mounted scene cam after the hardware update.

Run (vscode container):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/verify_cams.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt
Outputs -> <repo>/cam_out/v_wrist.png, v_scene.png
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=160)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "cam_out"
OUT.mkdir(exist_ok=True)

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
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


def policy(o):
    with torch.no_grad():
        out = runner.agent.act(o, timestep=0, timesteps=0)
        if len(out) > 2 and isinstance(out[2], dict) and out[2].get("mean_actions") is not None:
            return out[2]["mean_actions"]
        return out[0]


for _ in range(a.steps):
    obs, _, _, _, _ = wrapped.step(policy(obs))


def save(name, path):
    rgb = env.scene.sensors[name].data.output["rgb"][0, ..., :3]
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    Image.fromarray(rgb.cpu().numpy()).save(path)
    print("saved", path.name, flush=True)


save("wrist", OUT / "v_wrist.png")
save("camera", OUT / "v_scene.png")
print("VERIFY_CAMS_DONE", flush=True)
import os  # noqa: E402
os._exit(0)  # skip app.close() (it hangs and leaves a GPU zombie); OS reclaims the GPU
