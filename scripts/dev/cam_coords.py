"""Print gripper (panda_hand) + cube world coordinates and the cube's pose in the
hand's local frame, at the lifted pose. Use these numbers to set the wrist-camera
offset/orientation precisely (instead of guessing), then re-render a sweep.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

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
for t in range(a.steps):
    with torch.no_grad():
        out = runner.agent.act(obs, timestep=0, timesteps=0)
        if len(out) > 2 and isinstance(out[2], dict) and out[2].get("mean_actions") is not None:
            act = out[2]["mean_actions"]
        else:
            act = out[0]
    obs, _, _, _, _ = wrapped.step(act)


def r(t):
    return [round(x, 4) for x in t.tolist()]


robot = env.scene["robot"]
hand_idx = robot.find_bodies("panda_hand")[0][0]
p_hand = robot.data.body_pos_w[0, hand_idx]
q_hand = robot.data.body_quat_w[0, hand_idx]
obj = env.scene["object"]
p_cube = obj.data.root_pos_w[0]
ee = env.scene["ee_frame"]
p_tcp = ee.data.target_pos_w[0, 0]
cube_in_hand, _ = subtract_frame_transforms(p_hand.unsqueeze(0), q_hand.unsqueeze(0), p_cube.unsqueeze(0))
tcp_in_hand, _ = subtract_frame_transforms(p_hand.unsqueeze(0), q_hand.unsqueeze(0), p_tcp.unsqueeze(0))

print("=== COORDS (lifted pose, hand frame is panda_hand) ===")
print("p_hand_world:", r(p_hand))
print("q_hand_wxyz:", r(q_hand))
print("p_cube_world:", r(p_cube))
print("p_tcp_world:", r(p_tcp))
print("cube_in_hand_frame:", r(cube_in_hand[0]))
print("tcp_in_hand_frame:", r(tcp_in_hand[0]))
print("COORDS_DONE")
app.close()
