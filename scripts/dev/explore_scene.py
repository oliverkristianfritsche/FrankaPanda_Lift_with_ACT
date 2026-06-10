"""Explore the lift scene to understand its geometry BEFORE further camera work.

Reports:
  - panda_hand frame: world pose + local X/Y/Z axes in world, the finger-opening
    axis and the approach axis (so a wrist cam can be mounted PARALLEL to the hand,
    not just aimed at a point), plus cube/finger positions in the hand frame.
  - prim tree: env children, the panda_hand subtree, all Camera prims, /Visuals.
  - all mesh bounding boxes (size + center), flagging tall/thin ones (the 'bar').
  - 3 overview renders (front / side / top) of the workspace.

Run (vscode container):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/explore_scene.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt
Text -> /tmp/explore.log ; images -> <repo>/cam_out/explore_{front,side,top}.png
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms, quat_apply  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "cam_out"
OUT.mkdir(exist_ok=True)

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
# free overview camera we drive ourselves
cfg.scene.ov_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ov_cam", update_period=0.0, height=360, width=480, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, focus_distance=400.0,
                                     horizontal_aperture=20.955, clipping_range=(0.05, 12.0)),
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


for _ in range(a.steps):  # to the grasp pose
    obs, _, _, _, _ = wrapped.step(policy(obs))

robot = env.scene["robot"]


def r(t):
    return [round(float(x), 4) for x in t.tolist()]


def body_pose(name):
    i = robot.find_bodies(name)[0][0]
    return robot.data.body_pos_w[0, i], robot.data.body_quat_w[0, i]


ph, qh = body_pose("panda_hand")
obj = env.scene["object"].data.root_pos_w[0]
print("=== PANDA_HAND FRAME (at grasp pose) ===", flush=True)
print("hand_pos_w :", r(ph), flush=True)
print("hand_quat  :", r(qh), flush=True)
for nm, v in [("+X", (1., 0, 0)), ("+Y", (0, 1., 0)), ("+Z", (0, 0, 1.))]:
    ax = quat_apply(qh.unsqueeze(0), torch.tensor([v], device=qh.device, dtype=qh.dtype))[0]
    print(f"hand {nm} in world: {r(ax)}", flush=True)
cih, _ = subtract_frame_transforms(ph.unsqueeze(0), qh.unsqueeze(0), obj.unsqueeze(0))
print("cube_in_hand_frame:", r(cih[0]), "  (the approach axis points at the cube)", flush=True)
for fn in ["panda_leftfinger", "panda_rightfinger"]:
    try:
        fp, _ = body_pose(fn)
        fih, _ = subtract_frame_transforms(ph.unsqueeze(0), qh.unsqueeze(0), fp.unsqueeze(0))
        print(f"{fn}_in_hand:", r(fih[0]), flush=True)
    except Exception as e:  # noqa: BLE001
        print(fn, "err", repr(e), flush=True)

stage = omni.usd.get_context().get_stage()
ENV = "/World/envs/env_0"


def wpos(prim):
    try:
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [round(t[0], 3), round(t[1], 3), round(t[2], 3)]
    except Exception:  # noqa: BLE001
        return None


print("\n=== ENV CHILDREN ===", flush=True)
for c in stage.GetPrimAtPath(ENV).GetChildren():
    print(" ", c.GetPath(), "|", c.GetTypeName(), flush=True)

print("\n=== PANDA_HAND SUBTREE ===", flush=True)
hand = stage.GetPrimAtPath(ENV + "/Robot/panda_hand")
base_depth = str(hand.GetPath()).count("/")
for prim in Usd.PrimRange(hand):
    d = str(prim.GetPath()).count("/") - base_depth
    print("  " * d, prim.GetName(), "|", prim.GetTypeName(), flush=True)

print("\n=== ALL CAMERA PRIMS ===", flush=True)
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        print(" ", prim.GetPath(), "world", wpos(prim), flush=True)

print("\n=== /Visuals + /World children (markers?) ===", flush=True)
for top in ["/Visuals", "/World"]:
    pr = stage.GetPrimAtPath(top)
    if pr and pr.IsValid():
        for c in pr.GetChildren():
            print(" ", c.GetPath(), "|", c.GetTypeName(), flush=True)

print("\n=== MESH BBOXES (size XYZ / center; flag tall-thin = bar) ===", flush=True)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for prim in stage.Traverse():
    pth = str(prim.GetPath())
    if not prim.IsA(UsdGeom.Mesh):
        continue
    under_robot = pth.startswith(ENV + "/Robot/")
    try:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        size = [round(mx[i] - mn[i], 3) for i in range(3)]
        ctr = [round((mx[i] + mn[i]) / 2, 3) for i in range(3)]
        tall_thin = size[2] > 0.2 and size[0] < 0.08 and size[1] < 0.08
        if under_robot and not tall_thin:
            continue  # skip robot's own meshes unless tall/thin
        flag = "   <== TALL/THIN (bar?)" if tall_thin else ""
        print(" ", pth[-66:], "size", size, "ctr", ctr, flag, flush=True)
    except Exception:  # noqa: BLE001
        pass

# Overview renders
ov = env.scene.sensors["ov_cam"]


def shot(name, eye, target):
    ov.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=ph.device),
                                 torch.tensor([target], dtype=torch.float32, device=ph.device))
    for _ in range(3):
        global obs
        obs, _, _, _, _ = wrapped.step(policy(obs))
    rgb = ov.data.output["rgb"][0, ..., :3]
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    Image.fromarray(rgb.cpu().numpy()).save(OUT / f"explore_{name}.png")
    print("saved", f"explore_{name}.png", flush=True)


hx, hy, hz = float(ph[0]), float(ph[1]), float(ph[2])
shot("front", [hx + 1.2, hy, hz - 0.1], [hx, hy, hz - 0.2])
shot("side", [hx, hy - 1.2, hz], [hx, hy, hz - 0.2])
shot("top", [hx + 0.05, hy, hz + 1.0], [hx, hy, hz - 0.25])
print("EXPLORE_DONE", flush=True)
import os  # noqa: E402
os._exit(0)  # skip app.close() (it hangs and leaves a GPU zombie); OS reclaims the GPU
