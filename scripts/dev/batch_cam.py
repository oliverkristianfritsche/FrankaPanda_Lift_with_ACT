"""Coord-aware wrist-camera offset sweep — ONE Isaac boot, many candidate renders.

Boots Isaac Sim once, rolls the PPO policy to the lifted (grasped) pose, prints the
gripper/cube coordinates, then for a list of candidate eye-offsets (in the panda_hand
local frame) it:
  1. places a free camera at  eye = p_hand + R_hand * offset,
  2. aims it at the cube via set_world_poses_from_view (auto look-at),
  3. renders + saves a PNG,
  4. reads back the camera's world orientation and converts it into the equivalent
     STATIC offset rotation in the panda_hand frame — i.e. exactly the (pos, rot)
     you paste into make_wrist_camera()'s OffsetCfg for a rigid mount.

So picking the best-framed PNG directly gives a ready-to-use camera config. No live
viewer, no interactive command file, no reboot per try: one launch, one retrieval.

Run (in the launchable container, repo at /workspace/repo):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/batch_cam.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt

Outputs land in <repo>/cam_out/ (cam_<i>_<name>.png, scene.png, REPORT.txt).
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=160, help="policy steps to reach the lift pose")
p.add_argument("--focal", type=float, default=18.0, help="wrist lens focal length (mm)")
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
from isaaclab.utils.math import (  # noqa: E402
    subtract_frame_transforms, quat_apply, quat_mul, quat_conjugate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "cam_out"
OUT.mkdir(exist_ok=True)

# Candidate eye-offsets in the panda_hand local frame (metres). Broad first-pass
# spread: straight-behind, behind+offset in all four lateral directions, pure
# lateral, and diagonals. Whichever frames the gripper+cube best wins; refine after.
CANDIDATES = [
    ("back_10",   (0.00,  0.00, -0.10)),
    ("back_18",   (0.00,  0.00, -0.18)),
    ("backUp_px", (0.10,  0.00, -0.12)),
    ("backDn_nx", (-0.10, 0.00, -0.12)),
    ("backPy",    (0.00,  0.10, -0.12)),
    ("backNy",    (0.00, -0.10, -0.12)),
    ("sidePy",    (0.00,  0.14,  0.00)),
    ("sideNy",    (0.00, -0.14,  0.00)),
    ("topPx",     (0.14,  0.00,  0.00)),
    ("botNx",     (-0.14, 0.00,  0.00)),
    ("diagA",     (0.08,  0.08, -0.10)),
    ("diagB",     (0.08, -0.08, -0.10)),
]

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
# Free tuning camera at env root — we fully control its world pose each candidate.
cfg.scene.tune_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/tune_cam", update_period=0.0, height=240, width=320,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=a.focal, focus_distance=400.0,
                                     horizontal_aperture=20.955, clipping_range=(0.01, 6.0)),
    offset=CameraCfg.OffsetCfg(pos=(2.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
)

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

robot = env.scene["robot"]
hand_idx = robot.find_bodies("panda_hand")[0][0]
obj = env.scene["object"]
tune = env.scene.sensors["tune_cam"]


def poses():
    return (robot.data.body_pos_w[0, hand_idx], robot.data.body_quat_w[0, hand_idx],
            obj.data.root_pos_w[0])


def r(t):
    return [round(float(x), 4) for x in (t.tolist() if hasattr(t, "tolist") else t)]


ph, qh, pc = poses()
cih, _ = subtract_frame_transforms(ph.unsqueeze(0), qh.unsqueeze(0), pc.unsqueeze(0))
# Hand local axes expressed in world (tells us which offset dir is up/side/fingers).
ax = lambda v: r(quat_apply(qh.unsqueeze(0), torch.tensor([v], device=qh.device, dtype=qh.dtype))[0])

lines = []
def log(s):
    print(s, flush=True)
    lines.append(s)

log("=== LIFTED POSE COORDS (frame: panda_hand) ===")
log(f"p_hand_world      : {r(ph)}")
log(f"q_hand_wxyz       : {r(qh)}")
log(f"p_cube_world      : {r(pc)}")
log(f"cube_in_hand_frame: {r(cih[0])}   <- where the cube sits relative to the hand")
log(f"hand +X in world  : {ax((1.,0.,0.))}")
log(f"hand +Y in world  : {ax((0.,1.,0.))}")
log(f"hand +Z in world  : {ax((0.,0.,1.))}")
log("")
log("=== CANDIDATE WRIST-CAM OFFSETS (paste the winner into make_wrist_camera) ===")


def aim_and_capture(off):
    global obs
    a_ph, a_qh, a_pc = poses()
    v = torch.tensor([off], device=a_ph.device, dtype=a_ph.dtype)
    eye = a_ph + quat_apply(a_qh.unsqueeze(0), v)[0]
    tune.set_world_poses_from_view(eye.unsqueeze(0), a_pc.unsqueeze(0))
    for _ in range(3):
        obs, _, _, _, _ = wrapped.step(policy(obs))  # render the new camera pose
    # Re-read hand AFTER stepping so the offset rotation is consistent w/ the render.
    f_ph, f_qh, _ = poses()
    q_cam = tune.data.quat_w_world[0]                       # world-convention orientation
    rot_in_hand = quat_mul(quat_conjugate(f_qh).unsqueeze(0), q_cam.unsqueeze(0))[0]
    rgb = tune.data.output["rgb"][0, ..., :3]
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rot_in_hand, rgb


for i, (name, off) in enumerate(CANDIDATES):
    try:
        rot, rgb = aim_and_capture(off)
        path = OUT / f"cam_{i:02d}_{name}.png"
        Image.fromarray(rgb.cpu().numpy()).save(path)
        rr = r(rot)
        log(f"[{i:02d}] {name:9s} pos={tuple(off)}  rot_wxyz={tuple(rr)}")
        log(f"     OffsetCfg(pos={tuple(off)}, rot={tuple(rr)}, convention=\"world\")")
        log(f"     -> {path.name}")
    except Exception as e:
        log(f"[{i:02d}] {name:9s} ERROR {e!r}")

# Scene-cam reference frame for correlation.
scam = env.scene.sensors["camera"]
srgb = scam.data.output["rgb"][0, ..., :3]
if srgb.dtype == torch.float32:
    srgb = (srgb * 255).clamp(0, 255).to(torch.uint8)
Image.fromarray(srgb.cpu().numpy()).save(OUT / "scene.png")
log("")
log(f"scene reference -> scene.png")
(OUT / "REPORT.txt").write_text("\n".join(lines) + "\n")
log("BATCH_DONE")
app.close()
