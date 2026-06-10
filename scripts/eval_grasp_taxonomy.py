# Oliver Fritsche
# June 9, 2026
# CS 7180 Advanced Perception

"""Closed-loop grasp FAILURE TAXONOMY for an ACT checkpoint: where exactly does
the ~22% policy lose the cube? For each episode (temporal-agg inference, same
protocol as eval_checkpoints.py) log:
  - when the EXECUTED gripper command first commits to close (<0 sustained)
  - the EE<->cube offset (lateral / vertical / norm) at that instant
  - the min EE<->cube distance over the episode (did it ever get close?)
  - whether the cube was disturbed (knocked) without being lifted
  - outcome class: GRASPED / TOUCHED_NOT_GRASPED / NEVER_REACHED
Teacher-forced analysis says gripper close TIMING is near-perfect on demos but
arm-action L1 during the ballistic approach is ~2.8x the hover error — this
script tests, closed-loop, whether failures are approach-offset misses at a
correctly-timed close (-> recovery-data fix) rather than timing/grip-force.

  /isaac-sim/python.sh scripts/eval_grasp_taxonomy.py \
      --ckpt_dir logs/act/checkpoints/act_<ts> --ckpt best_model.pt --episodes 50
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--ckpt", default="best_model.pt")
p.add_argument("--episodes", type=int, default=50)
p.add_argument("--max_steps", type=int, default=250)
p.add_argument("--ta_k", type=float, default=0.01)
p.add_argument("--query_freq", type=int, default=0,
               help="If >0, receding horizon (re-plan every N steps) instead of action averaging")
p.add_argument("--out", default=None)
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
goal_cond = int(config.get("state_dim", config.get("qpos_dim", 8))) >= 11
pcfg = {k: config.get(k) for k in config}
pcfg.update({"ckpt_dir": a.ckpt_dir, "policy_class": "ACT", "task_name": "lift", "seed": 42, "num_epochs": 1})
policy = ACTPolicy(pcfg)
ck = torch.load(os.path.join(a.ckpt_dir, a.ckpt), weights_only=False)
policy.model.load_state_dict(ck["model_state_dict"])
policy.model.eval()
print(f"taxonomy: ckpt={a.ckpt} epoch={ck.get('epoch')} goal_cond={goal_cond} chunk={chunk}", flush=True)
sensors = {c: env.scene.sensors[c] for c in cam_names if c in env.scene.sensors}


def refresh_cameras(n=3):
    try:
        for _ in range(n):
            env.sim.step(render=True)
        env.scene.update(env.physics_dt)
    except Exception:  # noqa: BLE001
        pass


def predict_chunk(obs, grip):
    qparts = [obs["policy"][:, 0:7], grip]
    if goal_cond:
        qparts.append(obs["policy"][:, 21:24])
    qn = (torch.cat(qparts, dim=1) - qm) / qs
    imgs = [sensors[c].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
            for c in cam_names]
    return policy(qn, torch.stack(imgs, dim=1)) * ast + am


def ee_cube_world():
    ee = env.scene["ee_frame"].data.target_pos_w[0, 0]
    cube = env.scene["object"].data.root_pos_w[0]
    return ee, cube


episodes = []
adim = int(am.shape[-1])
for ep in range(a.episodes):
    obs, _ = env.reset()
    refresh_cameras()
    grip = torch.zeros(1, 1, device=device)
    all_time = torch.zeros(a.max_steps, a.max_steps + chunk, adim, device=device)
    start_z = obs["policy"][0, 20].item()
    cube0 = env.scene["object"].data.root_pos_w[0].clone()
    max_z = start_z
    min_ee_cube, min_ee_cube_t = 9.9, -1
    close_t, close_geo = None, None
    grip_hist = []
    placed, mind = False, 9.9
    buf, bi = None, 0
    for t in range(a.max_steps):
        with torch.no_grad():
            if a.query_freq > 0:
                # receding horizon: re-plan every N steps, execute raw prefix
                if buf is None or bi >= a.query_freq:
                    buf, bi = predict_chunk(obs, grip), 0
                action = buf[:, bi, :]
                bi += 1
            else:
                all_time[t, t:t + chunk] = predict_chunk(obs, grip)[0]
                acts = all_time[:, t]
                acts = acts[acts.abs().sum(dim=1) > 1e-8]
                w = torch.exp(-a.ta_k * torch.arange(len(acts), device=device, dtype=torch.float32))
                action = (acts * (w / w.sum()).unsqueeze(1)).sum(dim=0, keepdim=True)
        grip = action[:, -1:]
        grip_hist.append(float(grip.item()))
        # geometry BEFORE the step: where is the EE relative to the cube when this
        # command is issued?
        ee, cube = ee_cube_world()
        d_ee = torch.norm(ee - cube).item()
        if d_ee < min_ee_cube:
            min_ee_cube, min_ee_cube_t = d_ee, t
        if close_t is None and len(grip_hist) >= 3 and all(g < 0 for g in grip_hist[-3:]):
            close_t = t - 2
            off = (ee - cube)
            close_geo = {
                "lateral_mm": round(float(torch.norm(off[:2]).item()) * 1000, 1),
                "vertical_mm": round(float(off[2].item()) * 1000, 1),
                "norm_mm": round(d_ee * 1000, 1),
            }
        obs, _, term, trunc, _ = env.step(action)
        po = obs["policy"]
        d = torch.norm(po[0, 18:21] - po[0, 21:24]).item()
        mind = min(mind, d)
        max_z = max(max_z, po[0, 20].item())
        if d < 0.05:
            placed = True
        if bool(term[0]) or bool(trunc[0]):
            break
    cube_end = env.scene["object"].data.root_pos_w[0]
    disturbed = float(torch.norm(cube_end[:2] - cube0[:2]).item())
    lifted = (max_z - start_z) > 0.10
    outcome = ("GRASPED" if lifted else
               "TOUCHED_NOT_GRASPED" if (min_ee_cube < 0.06 or disturbed > 0.01) else
               "NEVER_REACHED")
    rec = {
        "ep": ep, "outcome": outcome, "placed": bool(placed),
        "lifted": bool(lifted), "min_goal_dist": round(mind, 3),
        "close_t": close_t, "close_geo": close_geo,
        "min_ee_cube_mm": round(min_ee_cube * 1000, 1), "min_ee_cube_t": min_ee_cube_t,
        "cube_xy_disturbed_mm": round(disturbed * 1000, 1),
    }
    episodes.append(rec)
    print(f"EP {ep:02d} {outcome:>20s} close_t={close_t} geo={close_geo} "
          f"minEE={rec['min_ee_cube_mm']}mm@t{min_ee_cube_t} disturbed={rec['cube_xy_disturbed_mm']}mm "
          f"placed={placed}", flush=True)

n = len(episodes)
lifted_n = sum(e["lifted"] for e in episodes)
placed_n = sum(e["placed"] for e in episodes)
print(f"\n==== TAXONOMY SUMMARY ({n} eps) ====", flush=True)
print(f"lift={lifted_n}/{n} ({100*lifted_n/n:.0f}%)  place={placed_n}/{n} ({100*placed_n/n:.0f}%)", flush=True)
for oc in ["GRASPED", "TOUCHED_NOT_GRASPED", "NEVER_REACHED"]:
    sel = [e for e in episodes if e["outcome"] == oc]
    if not sel:
        continue
    cts = [e["close_t"] for e in sel if e["close_t"] is not None]
    lat = [e["close_geo"]["lateral_mm"] for e in sel if e["close_geo"]]
    ver = [e["close_geo"]["vertical_mm"] for e in sel if e["close_geo"]]
    mee = [e["min_ee_cube_mm"] for e in sel]
    print(f"{oc}: n={len(sel)}  close_t med={np.median(cts) if cts else None}  "
          f"lat@close med={np.median(lat) if lat else None}mm  "
          f"vert@close med={np.median(ver) if ver else None}mm  "
          f"minEE med={np.median(mee):.0f}mm", flush=True)

out = a.out or os.path.join(a.ckpt_dir, "grasp_taxonomy.json")
with open(out, "w") as f:
    json.dump(episodes, f, indent=2)
print(f"WROTE {out}", flush=True)
print("TAXONOMY_DONE", flush=True)
os._exit(0)
