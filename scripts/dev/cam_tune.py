"""Persistent Isaac session for fast wrist-camera tuning — boots ONCE.

Boots Isaac Sim, rolls the PPO policy to the lifted pose, prints the cube's position
in the hand frame, then loops forever holding the pose. Each loop it reads an eye
offset (in the panda_hand local frame) from /tmp/tune_cmd, aims a free 'tune' camera
from that eye at the cube, renders, and writes /tmp/tune_view.png + /tmp/tune_ack.

Iterate by writing a new offset to /tmp/tune_cmd (e.g. `echo "0 -0.1 0.05" > /tmp/tune_cmd`)
then pulling /tmp/tune_view.png — no reboot. Write `quit` to stop.
"""
import argparse
import time
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
# Livestream is controlled by the --livestream CLI flag (pass `--livestream 2` to
# stream the viewport to <vscode-url>/viewer; omit it to run headless). Hardcoding it
# risked killing the loop if the streaming stack was down, so it stays a flag.
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
from isaaclab.utils.math import subtract_frame_transforms, quat_apply  # noqa: E402

TASK = "Template-Lift-Cube-Franka-Demo-Baseline-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 1
# Free tuning camera at env root — we fully control its world pose each step.
cfg.scene.tune_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/tune_cam", update_period=0.0, height=240, width=320,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
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
    return robot.data.body_pos_w[0, hand_idx], robot.data.body_quat_w[0, hand_idx], obj.data.root_pos_w[0]


ph, qh, pc = poses()
ch, _ = subtract_frame_transforms(ph.unsqueeze(0), qh.unsqueeze(0), pc.unsqueeze(0))
print("CUBE_IN_HAND", [round(x, 4) for x in ch[0].tolist()], flush=True)

CMD, VIEW, ACK = "/tmp/tune_cmd", "/tmp/tune_view.png", "/tmp/tune_ack"
with open(CMD, "w") as f:
    f.write("0 0 -0.15\n")  # first offset to try
last = None


def aim(off):
    a_ph, a_qh, a_pc = poses()
    v = torch.tensor(off, device=a_ph.device, dtype=a_ph.dtype)
    eye = a_ph + quat_apply(a_qh.unsqueeze(0), v.unsqueeze(0))[0]
    tune.set_world_poses_from_view(eye.unsqueeze(0), a_pc.unsqueeze(0))


print("TUNER_READY", flush=True)
while app.is_running():
    obs, _, _, _, _ = wrapped.step(policy(obs))  # hold the lifted pose
    try:
        cmd = open(CMD).read().strip()
    except Exception:
        cmd = None
    if cmd and cmd != last:
        if cmd == "quit":
            break
        try:
            aim([float(x) for x in cmd.split()])
            for _ in range(3):
                obs, _, _, _, _ = wrapped.step(policy(obs))  # render new pose
            rgb = tune.data.output["rgb"][0, ..., :3]
            if rgb.dtype == torch.float32:
                rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
            Image.fromarray(rgb.cpu().numpy()).save(VIEW)
            with open(ACK, "w") as f:
                f.write(cmd + " saved\n")
            print("AIMED", cmd, "saved", flush=True)
            last = cmd
        except Exception as e:
            print("ERR", repr(e), flush=True)
            last = cmd
    time.sleep(0.3)
app.close()
