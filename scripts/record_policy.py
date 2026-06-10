# Oliver Fritsche
# June 7, 2026
# CS 7180 Advanced Perception

"""Record a policy (PPO oracle or ACT student) performing the lift to an MP4, and
report success rate. Produces the README clips for both policies from a clean
cinematic 3rd-person camera (separate from the policy's own scene/wrist cameras).

  # ACT student (vision policy)
  /isaac-sim/python.sh scripts/record_policy.py --policy act \
      --checkpoint logs/act/checkpoints/act_<ts>/best_model.pt --out media/act.mp4
  # PPO oracle (state-based teacher)
  /isaac-sim/python.sh scripts/record_policy.py --policy ppo \
      --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt --out media/ppo.mp4
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--policy", choices=["act", "ppo"], required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--episodes", type=int, default=4)
p.add_argument("--max_steps", type=int, default=250)
p.add_argument("--out", required=True)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--gif", action="store_true", default=True,
               help="Also write a downscaled, looping GIF next to the MP4 (for inline README display)")
p.add_argument("--gif_max_w", type=int, default=760, help="Max GIF width in px (downscaled)")
p.add_argument("--gif_max_frames", type=int, default=200, help="Cap GIF frame count (subsampled)")
p.add_argument("--query_freq", type=int, default=25,
               help="ACT: re-predict every N steps (use first N of each chunk) for closed-loop control")
p.add_argument("--cine_h", type=int, default=360, help="cinematic camera height (ppo/triptych)")
p.add_argument("--cine_w", type=int, default=480, help="cinematic camera width")
p.add_argument("--layout", choices=["chunks", "triptych"], default="chunks",
               help="ACT video layout. 'chunks': scene cam with goal reticle, wrist cam, and live "
                    "predicted-chunk panels showing the executed prefix vs the re-planned remainder. "
                    "'triptych': legacy 3rd-person + camera feeds + joint trace.")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import numpy as np  # noqa: E402
import imageio  # noqa: E402
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"

cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
# Clean cinematic camera for the video (separate from the policy's scene/wrist cams).
cfg.scene.cine = CameraCfg(
    prim_path="{ENV_REGEX_NS}/cine", update_period=0.0, height=a.cine_h, width=a.cine_w, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=400.0,
                                     horizontal_aperture=20.955, clipping_range=(0.05, 8.0)),
    offset=CameraCfg.OffsetCfg(pos=(2.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
)
env = ManagerBasedRLEnv(cfg=cfg)
device = env.device
cine = env.scene.sensors["cine"]
SCENE = env.scene.sensors.get("camera")
WRIST = env.scene.sensors.get("wrist")
PANEL_H = a.cine_h


def refresh_cameras(n=3):
    # Camera buffers hold the PREVIOUS episode's frame until a render pass. Step the
    # sim (arm held at its PD target) with render=True after a reset so both the
    # policy's first prediction and the first recorded frame see THIS episode.
    try:
        for _ in range(n):
            env.sim.step(render=True)
        env.scene.update(env.physics_dt)
    except Exception:  # noqa: BLE001
        pass


def aim_cine():
    # Front-left, elevated 3rd-person view framing the arm + the workspace.
    eye = torch.tensor([[1.55, -0.95, 0.9]], device=device, dtype=torch.float32)
    tgt = torch.tensor([[0.45, -0.05, 0.2]], device=device, dtype=torch.float32)
    cine.set_world_poses_from_view(eye, tgt)


def _np(sensor):
    rgb = sensor.data.output["rgb"][0, ..., :3]
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rgb.cpu().numpy()


def _panel(img, text):
    w = int(img.shape[1] * PANEL_H / img.shape[0])
    img = cv2.resize(img, (w, PANEL_H))
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grab_frame():
    # ACT: triptych of 3rd-person + the two camera feeds the policy uses.
    # PPO oracle: clean 3rd-person (it is state-based, no camera inputs).
    cine_p = _panel(_np(cine), "Franka lift (3rd person)")
    if a.policy == "act" and SCENE is not None and WRIST is not None:
        return np.hstack([cine_p,
                          _panel(_np(SCENE), "scene cam (ACT input)"),
                          _panel(_np(WRIST), "wrist cam (ACT input)")])
    return cine_p


# ---- policy setup ----
if a.policy == "ppo":
    acfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
    acfg["trainer"]["close_environment_at_exit"] = False
    acfg["agent"]["experiment"]["write_interval"] = 0
    acfg["agent"]["experiment"]["checkpoint_interval"] = 0
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, acfg)
    runner.agent.load(a.checkpoint)
    runner.agent.set_running_mode("eval")

    def reset():
        o, _ = wrapped.reset()
        return o

    def act_fn(o, _state):
        with torch.no_grad():
            out = runner.agent.act(o, timestep=0, timesteps=0)
            if len(out) > 2 and isinstance(out[2], dict) and out[2].get("mean_actions") is not None:
                action = out[2]["mean_actions"]
            else:
                action = out[0]
        return action, None

    def step(action):
        return wrapped.step(action)
else:  # act
    sys.path.insert(0, os.path.join(REPO_ROOT, "act"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "act", "detr"))
    from policy import ACTPolicy  # noqa: E402
    ckpt_dir = os.path.dirname(a.checkpoint)
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)
    nrm = torch.load(os.path.join(ckpt_dir, "normalizer.pt"), weights_only=False)
    qpos_mean = torch.tensor(nrm["qpos_mean"], dtype=torch.float32, device=device)
    qpos_std = torch.tensor(nrm["qpos_std"], dtype=torch.float32, device=device)
    action_mean = torch.tensor(nrm["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(nrm["action_std"], dtype=torch.float32, device=device)
    cam_names = config.get("camera_names", ["camera", "wrist"])
    chunk = config.get("num_queries", 100)
    qf = a.query_freq if 0 < a.query_freq <= chunk else chunk
    goal_cond = int(config.get("state_dim", config.get("qpos_dim", 8))) >= 11  # 11 = goal in qpos
    pcfg = {k: config.get(k) for k in config}
    pcfg.update({"ckpt_dir": ckpt_dir, "policy_class": "ACT", "task_name": "lift", "seed": 42, "num_epochs": 1})
    policy = ACTPolicy(pcfg)
    ck = torch.load(a.checkpoint, weights_only=False)
    policy.model.load_state_dict(ck["model_state_dict"])
    policy.model.eval()
    print(f"Loaded ACT epoch {ck.get('epoch','?')} loss={ck.get('loss','?')}", flush=True)
    sensors = {c: env.scene.sensors[c] for c in cam_names if c in env.scene.sensors}
    _buf = {"buf": None, "i": 0, "grip": torch.zeros(1, 1, device=device)}

    def reset():
        o, _ = env.reset()
        _buf["buf"], _buf["i"] = None, 0
        _buf["grip"] = torch.zeros(1, 1, device=device)
        return o

    def act_fn(o, _state):
        if _buf["buf"] is None or _buf["i"] >= qf:
            with torch.no_grad():
                jp = o["policy"][:, 0:7]
                qparts = [jp, _buf["grip"]]
                if goal_cond:
                    qparts.append(o["policy"][:, 21:24])  # goal: must match training qpos
                qpos = torch.cat(qparts, dim=1)  # 8 (no goal) or 11 (goal-conditioned)
                qn = (qpos - qpos_mean) / qpos_std
                imgs = []
                for c in cam_names:
                    im = sensors[c].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
                    imgs.append(im)
                image = torch.stack(imgs, dim=1)
                _buf["buf"] = policy(qn, image) * action_std + action_mean
                _buf["i"] = 0
                _buf["chunk_np"] = _buf["buf"][0].detach().float().cpu().numpy()  # (chunk, 8)
        action = _buf["buf"][:, _buf["i"], :]
        _buf["i"] += 1
        _buf["grip"] = action[:, -1:]
        return action, None

    def step(action):
        return env.step(action)


def policy_obs(o):
    return o["policy"] if isinstance(o, dict) else o


class JointPlot:
    """Live joint-angle trace synced to the video: faint full trajectory + a bold
    portion that grows to the current step, with a moving cursor."""

    def __init__(self, joints, width, height=240):
        self.joints, self.T, self.W, self.H = joints, len(joints), width, height
        self.fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        x = np.arange(self.T)
        colors = plt.cm.turbo(np.linspace(0.05, 0.95, joints.shape[1]))
        self.lines = []
        for j in range(joints.shape[1]):
            self.ax.plot(x, joints[:, j], color=colors[j], alpha=0.13, lw=1)
            (ln,) = self.ax.plot([], [], color=colors[j], lw=2, label=f"j{j+1}")
            self.lines.append(ln)
        self.cursor = self.ax.axvline(0, color="0.35", lw=1)
        self.ax.set_xlim(0, max(self.T - 1, 1))
        lo, hi = float(joints.min()), float(joints.max())
        pad = 0.08 * (hi - lo + 1e-3)
        self.ax.set_ylim(lo - pad, hi + pad)
        self.ax.set_title("Franka joint angles (rad) — live", fontsize=11)
        self.ax.set_xlabel("step", fontsize=9)
        self.ax.grid(alpha=0.2)
        self.ax.tick_params(labelsize=8)
        self.ax.legend(ncol=7, fontsize=7, loc="upper center")
        self.fig.tight_layout(pad=0.5)
        self.fig.canvas.draw()

    def frame(self, t):
        x = np.arange(t + 1)
        for j, ln in enumerate(self.lines):
            ln.set_data(x, self.joints[:t + 1, j])
        self.cursor.set_xdata([t, t])
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return cv2.resize(np.ascontiguousarray(buf), (self.W, self.H))

    def close(self):
        plt.close(self.fig)


# ---- chunks layout: dark-theme panels with live predicted-chunk display ----
DARK_BG, DARK_PANEL = (17, 17, 14), "#0e1117"
BLUE_MPL, ORANGE_MPL, GRAY_MPL = "#4cc9f0", "#f4a261", "#5a6070"
BLUE_BGR = (240, 201, 76)  # cv2 uses BGR; this is #4cc9f0
GREEN_BGR = (126, 194, 46)
CH_H = 540  # layout height


def goal_pixel(goal_world):
    """Project a world point into the scene camera (pinhole, ROS convention)."""
    try:
        from isaaclab.utils.math import matrix_from_quat
        K = SCENE.data.intrinsic_matrices[0].cpu().numpy()
        cam_pos = SCENE.data.pos_w[0].cpu().numpy()
        R = matrix_from_quat(SCENE.data.quat_w_ros[0:1])[0].cpu().numpy()
        pc = R.T @ (np.asarray(goal_world) - cam_pos)
        if pc[2] <= 0.05:
            return None
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        return float(u), float(v)
    except Exception:  # noqa: BLE001
        return None


def scene_panel(goal_uv, scale):
    img = cv2.cvtColor(_np(SCENE), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if goal_uv is not None:
        u, v = int(goal_uv[0] * scale), int(goal_uv[1] * scale)
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.drawMarker(img, (u, v), GREEN_BGR, cv2.MARKER_STAR, 26, 2, cv2.LINE_AA)
            cv2.putText(img, "goal", (u + 14, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        GREEN_BGR, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), DARK_BG, -1)
    cv2.putText(img, "scene camera (policy input)", (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def wrist_panel(width, height):
    img = cv2.resize(_np(WRIST), (width, height), interpolation=cv2.INTER_CUBIC)
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (width, 26), DARK_BG[::-1], -1)
    cv2.putText(img, "wrist camera (policy input)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return img


class ChunkPanels:
    """Live predicted-chunk display: all joints (right panel) and the gripper channel
    (small panel). Executed prefix in blue, the re-planned remainder in gray, with a
    cursor at the current step inside the chunk."""

    def __init__(self, chunk_len, exec_len, joint_w, joint_h, grip_w, grip_h):
        for k, v in {"figure.facecolor": DARK_PANEL, "axes.facecolor": DARK_PANEL,
                     "axes.edgecolor": "#3a3f4a", "axes.labelcolor": "#e6e6e6",
                     "text.color": "#e6e6e6", "xtick.color": "#9aa0aa",
                     "ytick.color": "#9aa0aa", "grid.color": "#262b33"}.items():
            plt.rcParams[k] = v
        self.L, self.E = chunk_len, exec_len
        self.jw, self.jh, self.gw, self.gh = joint_w, joint_h, grip_w, grip_h
        x = np.arange(self.L)
        self.fig_j = plt.figure(figsize=(joint_w / 100, joint_h / 100), dpi=100)
        self.ax_j = self.fig_j.add_subplot(111)
        self.j_gray, self.j_blue = [], []
        for j in range(7):
            (g,) = self.ax_j.plot(x, np.zeros(self.L), color=GRAY_MPL, lw=1.5, alpha=0.85)
            (b,) = self.ax_j.plot(x[:self.E], np.zeros(self.E), color=BLUE_MPL, lw=2.6)
            self.j_gray.append(g); self.j_blue.append(b)
        self.j_cursor = self.ax_j.axvline(0, color="#e6e6e6", lw=1, alpha=0.6)
        self.ax_j.axvspan(0, self.E - 1, color=BLUE_MPL, alpha=0.07)
        self.ax_j.set_xlim(0, self.L - 1)
        self.ax_j.set_title(f"predicted chunk ({self.L} steps): blue = executed {self.E}, "
                            "gray = re-planned away", fontsize=10, color="#e6e6e6")
        self.ax_j.set_xlabel("steps ahead", fontsize=8)
        self.ax_j.grid(alpha=0.4); self.ax_j.tick_params(labelsize=7)
        self.fig_j.tight_layout(pad=0.4)
        self.fig_g = plt.figure(figsize=(grip_w / 100, grip_h / 100), dpi=100)
        self.ax_g = self.fig_g.add_subplot(111)
        (self.g_line,) = self.ax_g.plot(x, np.zeros(self.L), color=ORANGE_MPL, lw=2)
        (self.g_exec,) = self.ax_g.plot(x[:self.E], np.zeros(self.E), color=BLUE_MPL, lw=2.8)
        self.g_cursor = self.ax_g.axvline(0, color="#e6e6e6", lw=1, alpha=0.6)
        self.ax_g.axhline(0, color=GRAY_MPL, lw=0.8, ls=":")
        self.ax_g.set_title("gripper channel (close < 0)", fontsize=9, color="#e6e6e6")
        self.ax_g.grid(alpha=0.4); self.ax_g.tick_params(labelsize=7)
        self.fig_g.tight_layout(pad=0.4)

    def render(self, chunk, i):
        offs = np.arange(7) * 1.4
        base = chunk[0, :7]
        ymin, ymax = 1e9, -1e9
        for j in range(7):
            y = chunk[:, j] - base[j] + offs[j]
            self.j_gray[j].set_ydata(y)
            self.j_blue[j].set_ydata(y[:self.E])
            ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
        self.ax_j.set_ylim(ymin - 0.4, ymax + 0.4)
        self.j_cursor.set_xdata([i, i])
        self.fig_j.canvas.draw()
        jimg = np.asarray(self.fig_j.canvas.buffer_rgba())[..., :3]
        g = chunk[:, 7]
        self.g_line.set_ydata(g)
        self.g_exec.set_ydata(g[:self.E])
        self.ax_g.set_ylim(g.min() - 0.4, g.max() + 0.4)
        self.g_cursor.set_xdata([i, i])
        self.fig_g.canvas.draw()
        gimg = np.asarray(self.fig_g.canvas.buffer_rgba())[..., :3]
        return (cv2.resize(np.ascontiguousarray(jimg), (self.jw, self.jh)),
                cv2.resize(np.ascontiguousarray(gimg), (self.gw, self.gh)))

    def close(self):
        plt.close(self.fig_j); plt.close(self.fig_g)


def compose_chunks_frame(goal_uv, panels):
    scene = scene_panel(goal_uv, CH_H / 240.0)        # 720x540
    mid_w = 360
    wrist = wrist_panel(mid_w, CH_H // 2)             # 360x270
    jimg, gimg = panels                                # 640x540, 360x270
    mid = np.vstack([wrist, gimg])
    return np.hstack([scene, mid, jimg])


# ---- rollout + record ----
aim_cine()
video = []
successes = 0
use_chunks = a.policy == "act" and a.layout == "chunks" and SCENE is not None and WRIST is not None
for ep in range(a.episodes):
    obs = reset()
    aim_cine()
    refresh_cameras()  # fresh frames for this episode (first predict + first recorded frame)
    ep_success = False
    panels, joints = [], []
    cp = ChunkPanels(chunk, qf, 640, CH_H, 360, CH_H - CH_H // 2) if use_chunks else None
    goal_w = None
    for t in range(a.max_steps):
        action, _ = act_fn(obs, None)
        obs, _, term, trunc, _ = step(action)
        po = policy_obs(obs)
        if use_chunks:
            if goal_w is None:
                # goal in world frame = robot-frame goal + env origin (single env at origin offset)
                goal_w = (po[0, 21:24] + env.scene.env_origins[0]).detach().cpu().numpy()
            jg = cp.render(_buf["chunk_np"], _buf["i"] - 1)
            panels.append(compose_chunks_frame(goal_pixel(goal_w), jg))
        else:
            panels.append(grab_frame())
        joints.append(po[0, 0:7].detach().float().cpu().numpy())
        try:
            if torch.norm(po[0, 18:21] - po[0, 21:24]).item() < 0.05:
                ep_success = True
        except Exception:  # noqa: BLE001
            pass
        if bool(term[0]) or bool(trunc[0]):
            break
    successes += int(ep_success)
    if cp is not None:
        cp.close()
        video.extend(panels)
    else:
        jp = JointPlot(np.asarray(joints), panels[0].shape[1])
        for i in range(len(panels)):
            video.append(np.vstack([panels[i], jp.frame(i)]))
        jp.close()
    print(f"episode {ep+1}/{a.episodes} steps={len(panels)} success={ep_success}", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
imageio.mimsave(a.out, video, fps=a.fps, quality=8)
print(f"WROTE {a.out}  frames={len(video)}  size_mb={os.path.getsize(a.out)/1e6:.1f}", flush=True)

# Also emit a lightweight looping GIF (downscaled + frame-subsampled) for inline
# display in the README — GitHub renders GIFs but not MP4s via ![](...).
if a.gif:
    try:
        gif_path = os.path.splitext(a.out)[0] + ".gif"
        stride = max(1, len(video) // max(1, a.gif_max_frames))
        gframes = video[::stride]
        h, w = gframes[0].shape[:2]
        if w > a.gif_max_w:
            nh = int(round(h * a.gif_max_w / w))
            gframes = [cv2.resize(f, (a.gif_max_w, nh)) for f in gframes]
        imageio.mimsave(gif_path, gframes, fps=max(1, a.fps // stride), loop=0)
        print(f"WROTE {gif_path}  frames={len(gframes)}  size_mb={os.path.getsize(gif_path)/1e6:.1f}", flush=True)
    except Exception as e:  # noqa: BLE001 — never let GIF export sink the MP4 / success report
        print(f"GIF export failed (mp4 still saved): {e}", flush=True)

print(f"SUCCESS_RATE {a.policy} {successes}/{a.episodes}", flush=True)
print("RECORD_DONE", flush=True)
os._exit(0)
