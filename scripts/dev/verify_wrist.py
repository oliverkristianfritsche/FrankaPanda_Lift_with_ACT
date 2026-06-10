"""Verify candidate wrist-camera mounts the PRODUCTION way: rigidly parented to
panda_hand with a static OffsetCfg (pos+rot), exactly like make_wrist_camera().

The batch sweep (batch_cam.py) found that offsets ~14cm along the hand's local
+-X axis, aimed back at the grasp, frame the cube + both fingers nicely. Its
read-back of the look-at orientation was stale (identity), so here the rotation is
computed analytically (optical +X -> view dir, up = world +Z) and CHECKED by
rendering the real parented camera at the lifted pose.

Run (container, repo at /workspace/repo):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/verify_wrist.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt

Outputs -> <repo>/cam_out/verify_<name>.png
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
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "cam_out"
OUT.mkdir(exist_ok=True)

# Candidates: (name, pos in panda_hand frame, rot wxyz in "world" convention).
# g09 / g08 are the analytic look-at rotations for the two best sweep offsets.
# 'existing' is the current make_wrist_camera() pose, for comparison.
CANDS = [
    ("g09", (-0.14, 0.0, 0.0), (0.1288, -0.9505, -0.0349, -0.2809)),
    ("g08", (0.14, 0.0, 0.0), (0.2916, -0.08963, -0.9073, -0.2893)),
    ("existing", (0.0, 0.0, 0.05), (0.7071, 0.0, -0.7071, 0.0)),
]

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
for name, pos, rot in CANDS:
    setattr(cfg.scene, f"vw_{name}", CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/vw_" + name,
        update_period=0.0, height=240, width=320, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.01, 6.0)),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
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


def policy(o):
    with torch.no_grad():
        out = runner.agent.act(o, timestep=0, timesteps=0)
        if len(out) > 2 and isinstance(out[2], dict) and out[2].get("mean_actions") is not None:
            return out[2]["mean_actions"]
        return out[0]


for _ in range(a.steps):
    obs, _, _, _, _ = wrapped.step(policy(obs))


def save(camname, path):
    rgb = env.scene.sensors[camname].data.output["rgb"][0, ..., :3]
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    Image.fromarray(rgb.cpu().numpy()).save(path)
    print("saved", path.name, flush=True)


for name, _, _ in CANDS:
    save(f"vw_{name}", OUT / f"verify_{name}.png")
print("VERIFY_DONE", flush=True)
app.close()
