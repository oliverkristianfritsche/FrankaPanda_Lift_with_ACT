"""3rd-person view that DRAWS the wrist camera's position + viewing frustum (cone),
rigidly tied to panda_hand, while the arm runs pick-lift episodes on a loop.

The cone is computed every frame from the LIVE panda_hand body pose composed with
make_wrist_camera()'s offset (pos/rot), so it tracks the wrist exactly as the arm
moves (USD stage transforms are stale under Fabric; the camera-sensor orientation
can lag, so we use the body pose + known offset directly). Lines: cyan = FOV cone,
GREEN = the camera's "up" (top edge + a stick) so the roll is obvious.

Run (vscode container):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/cam_gizmo_view.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt --livestream 2
Open <vscode-url>/viewer (mouse-drag to orbit if supported). pkill to stop.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--far", type=float, default=0.24, help="frustum draw depth (m)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402

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


robot = env.scene["robot"]
hand_idx = robot.find_bodies("panda_hand")[0][0]
dev = robot.data.body_pos_w.device

# make_wrist_camera() offset (keep in sync with visual_randomization.py).
OFF_POS = torch.tensor([[-0.14, 0.0, 0.0]], device=dev)
OFF_ROT = torch.tensor([[-0.0382, -0.9584, 0.0144, -0.2827]], device=dev)  # world conv
EX = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
EZ = torch.tensor([[0.0, 0.0, 1.0]], device=dev)
TAN_H = (20.955 / 2) / 18.0
TAN_V = (20.955 * 240 / 320 / 2) / 18.0

for _ in range(5):
    obs, _, _, _, _ = wrapped.step(policy(obs))

# Fixed perspective framing the whole workspace so the moving arm stays in view.
try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:  # noqa: BLE001
    from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(eye=[1.5, -1.05, 1.05], target=[0.45, 0.0, 0.25])

try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.util.debug_draw")
except Exception as e:  # noqa: BLE001
    print("EXT", repr(e), flush=True)
try:
    from isaacsim.util.debug_draw import _debug_draw
except Exception:  # noqa: BLE001
    from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()


def _t(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def cam_pose():
    hp = robot.data.body_pos_w[0, hand_idx].unsqueeze(0)
    hq = robot.data.body_quat_w[0, hand_idx].unsqueeze(0)
    pos = (hp + quat_apply(hq, OFF_POS))[0]
    q = quat_mul(hq, OFF_ROT)
    f = quat_apply(q, EX)[0]
    u = quat_apply(q, EZ)[0]
    return pos, f, u


def draw_frustum():
    pos, f, u = cam_pose()
    r = torch.linalg.cross(f, u)
    d = a.far
    hw, hh = d * TAN_H, d * TAN_V
    cf = pos + f * d
    c1, c2 = cf + r * hw + u * hh, cf + r * hw - u * hh
    c3, c4 = cf - r * hw - u * hh, cf - r * hw + u * hh
    starts = [pos, pos, pos, pos, c1, c2, c3, c4, cf]
    ends = [c1, c2, c3, c4, c2, c3, c4, c1, cf + u * 0.1]
    cyan, green = (0.0, 1.0, 1.0, 1.0), (0.1, 1.0, 0.1, 1.0)
    colors = [cyan, cyan, cyan, cyan, cyan, cyan, cyan, green, green]  # top edge + up-stick green
    sizes = [5.0] * len(starts)
    draw.clear_lines()
    draw.draw_lines([_t(s) for s in starts], [_t(e) for e in ends], colors, sizes)


# Diagnostic: body-derived cam pos vs the sensor's reported pos (should match).
p0, _, _ = cam_pose()
try:
    sp = env.scene.sensors["wrist"].data.pos_w[0]
    print("CAMPOS body=", [round(float(x), 3) for x in p0], "sensor=", [round(float(x), 3) for x in sp], flush=True)
except Exception as e:  # noqa: BLE001
    print("CAMPOS body=", [round(float(x), 3) for x in p0], "sensor_err", repr(e), flush=True)

print("GIZMO_VIEW_READY", flush=True)
while app.is_running():
    obs, _, _, _, _ = wrapped.step(policy(obs))  # loop pick-lift episodes (auto-reset)
    draw_frustum()
app.close()
