# Oliver Fritsche
# June 8, 2026
# CS 7180 Advanced Perception

"""Diagnose why the ACT policy scores 0% despite low training loss. Two tests:
  (A) Vision conditioning: reset N times (cube lands in different spots) and print
      the cube position + the policy's FIRST predicted action. If the actions
      barely change while the cube moves, the policy is ignoring the cameras
      (latent/mean-trajectory collapse) -> the bug is in training, not inference.
  (B) Behavior trace: roll one episode and log dist-to-goal / cube height /
      gripper every 30 steps, so we can see if it reaches, grasps, or just hovers.

  /isaac-sim/python.sh scripts/act_diag.py --ckpt_dir logs/act/checkpoints/act_<ts>
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--resets", type=int, default=5)
p.add_argument("--max_steps", type=int, default=250)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import numpy as np  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "act"))
sys.path.insert(0, os.path.join(REPO_ROOT, "act", "detr"))
from policy import ACTPolicy  # noqa: E402

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=cfg)
device = env.device

with open(os.path.join(a.ckpt_dir, "config.json")) as f:
    config = json.load(f)
nrm = torch.load(os.path.join(a.ckpt_dir, "normalizer.pt"), weights_only=False)
qm = torch.tensor(nrm["qpos_mean"], dtype=torch.float32, device=device)
qs = torch.tensor(nrm["qpos_std"], dtype=torch.float32, device=device)
am = torch.tensor(nrm["action_mean"], dtype=torch.float32, device=device)
ast = torch.tensor(nrm["action_std"], dtype=torch.float32, device=device)
cam_names = config.get("camera_names", ["camera", "wrist"])
chunk = config.get("num_queries", 100)
pcfg = {k: config.get(k) for k in config}
pcfg.update({"ckpt_dir": a.ckpt_dir, "policy_class": "ACT", "task_name": "lift", "seed": 42, "num_epochs": 1})
policy = ACTPolicy(pcfg)
ck = torch.load(os.path.join(a.ckpt_dir, "best_model.pt"), weights_only=False)
policy.model.load_state_dict(ck["model_state_dict"])
policy.model.eval()
print(f"Loaded best_model epoch={ck.get('epoch')} loss={ck.get('loss'):.4f}", flush=True)
sensors = {c: env.scene.sensors[c] for c in cam_names if c in env.scene.sensors}
print(f"cams={cam_names} chunk={chunk} action_mean={np.round(nrm['action_mean'],3)}", flush=True)
print(f"action_std={np.round(nrm['action_std'],3)} qpos_mean={np.round(nrm['qpos_mean'],3)}", flush=True)


def refresh_cameras(n=3):
    # Push fresh frames into the camera buffers after a reset WITHOUT commanding a
    # move: sim.step holds the arm at its current PD target and lets the cube settle,
    # render=True repaints, then update the sensors so .data.output is current.
    try:
        for _ in range(n):
            env.sim.step(render=True)
        env.scene.update(env.physics_dt)
    except Exception as e:  # noqa: BLE001
        print(f"  refresh_cameras failed: {e}", flush=True)


def cam_stats():
    s = {}
    for c in cam_names:
        rgb = sensors[c].data.output["rgb"][..., :3].float()
        s[c] = (round(rgb.mean().item(), 2), round(rgb.std().item(), 2))
    return s


def predict_chunk(obs, grip):
    with torch.no_grad():
        qn = (torch.cat([obs["policy"][:, 0:7], grip], dim=1) - qm) / qs
        imgs = [sensors[c].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0 for c in cam_names]
        return policy(qn, torch.stack(imgs, dim=1)) * ast + am


print("\n=== TEST A: vision conditioning (does first action track the cube?) ===", flush=True)
grip0 = torch.zeros(1, 1, device=device)
first_actions = []
cube_positions = []
for r in range(a.resets):
    obs, _ = env.reset()
    stale = cam_stats()
    refresh_cameras()           # guarantee fresh frames reflecting THIS reset's cube
    fresh = cam_stats()
    cube = obs["policy"][0, 18:21].cpu().numpy()
    goal = obs["policy"][0, 21:24].cpu().numpy()
    buf = predict_chunk(obs, grip0)
    fa = buf[0, 0, :].cpu().numpy()
    cube_positions.append(cube)
    first_actions.append(fa)
    print(f"reset {r}: cube={np.round(cube,3)} img_stale={stale} img_fresh={fresh}", flush=True)
    print(f"          first_action={np.round(fa,3)}", flush=True)

cube_positions = np.array(cube_positions)
first_actions = np.array(first_actions)
print(f"\ncube spread (std over resets): {np.round(cube_positions.std(0),4)}", flush=True)
print(f"first-action spread (std over resets): {np.round(first_actions.std(0),4)}", flush=True)
print(f"first-action range per dim: {np.round(first_actions.max(0)-first_actions.min(0),4)}", flush=True)
print("If cube moves but first_action barely changes -> policy IGNORES vision.", flush=True)


print("\n=== TEST B: behavior trace (one episode) ===", flush=True)
obs, _ = env.reset()
grip = torch.zeros(1, 1, device=device)
buf, idx = None, 0
min_dist, max_z = 9.9, -9.9
for t in range(a.max_steps):
    if buf is None or idx >= chunk:
        buf = predict_chunk(obs, grip)
        idx = 0
    action = buf[:, idx, :]
    idx += 1
    grip = action[:, -1:]
    obs, _, term, trunc, _ = env.step(action)
    po = obs["policy"]
    cube = po[0, 18:21]
    goal = po[0, 21:24]
    dist = torch.norm(cube - goal).item()
    z = cube[2].item()
    min_dist = min(min_dist, dist)
    max_z = max(max_z, z)
    if t % 30 == 0:
        print(f"  t={t:3d} dist={dist:.3f} cube_z={z:.3f} grip_cmd={grip.item():.3f} "
              f"img={cam_stats()} act[:3]={np.round(action[0,:3].cpu().numpy(),2)}", flush=True)
    if bool(term[0]) or bool(trunc[0]):
        break
print(f"\nEpisode end: min_dist_to_goal={min_dist:.3f} (success<0.05) max_cube_z={max_z:.3f}", flush=True)
print("ACT_DIAG_DONE", flush=True)
os._exit(0)
