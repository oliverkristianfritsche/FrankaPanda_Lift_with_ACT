"""Wrist-camera ROLL sweep — same validated mount, different roll about the optical axis.

The wrist-cam position/look-direction (g09: pos (-0.14,0,0), aimed at the grasp) is
good, but the roll (rotation about the viewing axis) needs tuning. This mounts one
parented camera per roll offset = base_rot composed with a roll about the optical
axis (+X in the 'world' convention), renders each at the lifted pose, and prints the
ready-to-paste OffsetCfg. Pick the upright one; its rot goes straight into
make_wrist_camera (no convention conversion — these ARE production-style mounts).

Run (vscode container, repo at /workspace/repo):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/roll_sweep.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt
Outputs -> <repo>/cam_out/roll_<deg>.png  (+ printed OffsetCfgs)
"""
import argparse
import math
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
from isaaclab.utils.math import quat_mul  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "cam_out"
OUT.mkdir(exist_ok=True)

# Validated g09 mount; only roll varies here.
BASE_POS = (-0.14, 0.0, 0.0)
BASE_ROT = (0.1288, -0.9505, -0.0349, -0.2809)
ROLLS_DEG = [-90, -45, -20, 0, 20, 45, 90, 180]


def label(d):
    return ("n" if d < 0 else "p") + f"{abs(d):03d}"


q_base = torch.tensor([BASE_ROT], dtype=torch.float32)
variants = []  # (deg, rot_tuple)
for d in ROLLS_DEG:
    r = math.radians(d)
    q_roll = torch.tensor([[math.cos(r / 2), math.sin(r / 2), 0.0, 0.0]], dtype=torch.float32)  # roll about optical +X
    q_var = quat_mul(q_base, q_roll)[0]
    variants.append((d, tuple(round(float(x), 4) for x in q_var.tolist())))

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
for d, rot in variants:
    setattr(cfg.scene, f"roll_{label(d)}", CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/roll_" + label(d),
        update_period=0.0, height=240, width=320, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.01, 6.0)),
        offset=CameraCfg.OffsetCfg(pos=BASE_POS, rot=rot, convention="world"),
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

print("=== ROLL SWEEP (pos fixed at g09; paste the upright one into make_wrist_camera) ===", flush=True)
for d, rot in variants:
    cam = env.scene.sensors[f"roll_{label(d)}"]
    rgb = cam.data.output["rgb"][0, ..., :3]
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    Image.fromarray(rgb.cpu().numpy()).save(OUT / f"roll_{label(d)}.png")
    print(f"roll={d:+4d} deg  rot={rot}  -> roll_{label(d)}.png", flush=True)
print("ROLL_DONE", flush=True)
app.close()
