# Oliver Fritsche
# June 8, 2026
# CS 7180 Advanced Perception

"""Evaluate the ACT policy at every saved training checkpoint to get a
success-rate-vs-training-epoch curve (writes success_curve.json). One Isaac boot
loops over checkpoint_*.pt (+ best_model.pt) so we can SEE the task success climb
as training progresses.

  /isaac-sim/python.sh scripts/eval_checkpoints.py \
      --ckpt_dir logs/act/checkpoints/act_<ts> --episodes 8
"""
import argparse
import glob
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--episodes", type=int, default=8)
p.add_argument("--max_steps", type=int, default=250)
p.add_argument("--query_freq", type=int, default=0,
               help="Re-predict every N steps (use first N of each chunk). 0 = execute full chunk.")
p.add_argument("--temporal_agg", action="store_true",
               help="Temporal aggregation: query every step, exp-weighted average of all overlapping chunk predictions for each timestep (ACT paper).")
p.add_argument("--ta_k", type=float, default=0.01, help="Temporal-agg decay weight (smaller = more uniform averaging)")
p.add_argument("--only_best", action="store_true", help="Evaluate only best_model.pt (fast)")
p.add_argument("--ckpt", default=None, help="Evaluate a single named checkpoint (e.g. checkpoint_0005.pt)")
p.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import numpy as np  # noqa: F401,E402
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
# Auto-detect goal-conditioning from the model's qpos size: 11 = joints+gripper+goal,
# 8 = no goal. Lets us eval both the goal-blind (Model A) and goal-conditioned (Model B).
goal_cond = int(config.get("state_dim", config.get("qpos_dim", 8))) >= 11
print(f"goal_cond={goal_cond} (qpos {'includes goal' if goal_cond else 'no goal'})", flush=True)
pcfg = {k: config.get(k) for k in config}
pcfg.update({"ckpt_dir": a.ckpt_dir, "policy_class": "ACT", "task_name": "lift", "seed": 42, "num_epochs": 1})
policy = ACTPolicy(pcfg)
sensors = {c: env.scene.sensors[c] for c in cam_names if c in env.scene.sensors}


def refresh_cameras(n=3):
    # Camera buffers hold the PREVIOUS episode's frame until a render pass. Step the
    # sim (arm held at its PD target) with render=True so the first prediction sees
    # THIS episode's cube, not the last one's. Without this the first chunk is garbage.
    try:
        for _ in range(n):
            env.sim.step(render=True)
        env.scene.update(env.physics_dt)
    except Exception:  # noqa: BLE001
        pass


def predict_chunk(obs, grip):
    qparts = [obs["policy"][:, 0:7], grip]
    if goal_cond:
        qparts.append(obs["policy"][:, 21:24])  # goal — must match training qpos
    qn = (torch.cat(qparts, dim=1) - qm) / qs
    imgs = [sensors[c].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
            for c in cam_names]
    return policy(qn, torch.stack(imgs, dim=1)) * ast + am  # [1, chunk, action_dim]


def run_episodes():
    # Two metrics: LIFT (goal-independent: cube raised >=10cm off the table) and PLACE
    # (goal-dependent: cube within 5cm of the goal).
    # Inference: either receding-horizon (re-predict every query_freq steps, execute the
    # first query_freq of each chunk) OR temporal aggregation (query every step, exp-weighted
    # average of all overlapping chunk predictions for each timestep — ACT-paper style).
    qf = a.query_freq if a.query_freq > 0 else chunk
    adim = int(am.shape[-1])
    place_succ, lift_succ = 0, 0
    dists = []
    for _ in range(a.episodes):
        obs, _ = env.reset()
        refresh_cameras()
        grip, placed = torch.zeros(1, 1, device=device), False
        mind = 9.9
        start_z = obs["policy"][0, 20].item()  # cube z (root frame) at start (on the table)
        max_z = start_z
        buf, idx = None, 0
        all_time = torch.zeros(a.max_steps, a.max_steps + chunk, adim, device=device) if a.temporal_agg else None
        for t in range(a.max_steps):
            with torch.no_grad():
                if a.temporal_agg:
                    all_time[t, t:t + chunk] = predict_chunk(obs, grip)[0]   # query every step
                    acts = all_time[:, t]                                    # every plan that covers t
                    acts = acts[acts.abs().sum(dim=1) > 1e-8]
                    w = torch.exp(-a.ta_k * torch.arange(len(acts), device=device, dtype=torch.float32))
                    action = (acts * (w / w.sum()).unsqueeze(1)).sum(dim=0, keepdim=True)
                else:
                    if buf is None or idx >= qf:
                        buf, idx = predict_chunk(obs, grip), 0
                    action = buf[:, idx, :]
                    idx += 1
            grip = action[:, -1:]
            obs, _, term, trunc, _ = env.step(action)
            po = obs["policy"]
            d = torch.norm(po[0, 18:21] - po[0, 21:24]).item()
            mind = min(mind, d)
            max_z = max(max_z, po[0, 20].item())
            if d < 0.05:
                placed = True
            if bool(term[0]) or bool(trunc[0]):
                break
        place_succ += int(placed)
        lift_succ += int((max_z - start_z) > 0.10)  # cube rose >=10cm off the table
        dists.append(round(mind, 3))
    print(f"   min_dists={dists}", flush=True)
    return place_succ, lift_succ


def epoch_of(path):
    m = re.search(r"checkpoint_(\d+)\.pt", os.path.basename(path))
    return int(m.group(1)) if m else 10 ** 9


best = os.path.join(a.ckpt_dir, "best_model.pt")
if a.ckpt:
    ckpts = [os.path.join(a.ckpt_dir, a.ckpt)]
elif a.only_best:
    ckpts = [best] if os.path.exists(best) else []
else:
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "checkpoint_*.pt")), key=epoch_of)
    if os.path.exists(best):
        ckpts.append(best)
print(f"query_freq={a.query_freq or chunk}  episodes={a.episodes}  ckpts={len(ckpts)}", flush=True)

results = []
for path in ckpts:
    ck = torch.load(path, weights_only=False)
    policy.model.load_state_dict(ck["model_state_dict"])
    policy.model.eval()
    ep = int(ck.get("epoch", epoch_of(path)))
    place, lift = run_episodes()
    psr = 100.0 * place / a.episodes
    lsr = 100.0 * lift / a.episodes
    tag = os.path.basename(path)
    print(f"CKPT {tag} epoch={ep} place={place}/{a.episodes} ({psr:.0f}%)  lift={lift}/{a.episodes} ({lsr:.0f}%)", flush=True)
    results.append({"epoch": ep, "episodes": a.episodes,
                    "place_success": place, "place_rate": psr,
                    "lift_success": lift, "lift_rate": lsr,
                    "success": place, "success_rate": psr,  # back-compat (place = the task success)
                    "ckpt": tag})

out = a.out or os.path.join(a.ckpt_dir, "success_curve.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"WROTE {out}", flush=True)
print("EVAL_CKPTS_DONE", flush=True)
os._exit(0)
