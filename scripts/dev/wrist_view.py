"""Live wrist-camera view for the streaming /viewer.

Sets the active (streamed) viewport to the PRODUCTION wrist-camera prim on
panda_hand, then runs pick-lift episodes on a loop. Watch <vscode-url>/viewer:
if the camera is rigidly + correctly mounted, the gripper fingers stay fixed in
the frame and the cube tracks through every approach -> grasp -> lift while the
wrist moves and rotates. This is the dynamic check a single saved frame can't give.

Run (in the vscode container, repo at /workspace/repo):
    unset CUDA_VISIBLE_DEVICES
    /isaac-sim/python.sh scripts/wrist_view.py \
        --checkpoint logs/skrl/franka_lift/<run>/checkpoints/best_agent.pt --livestream 2
Then open <vscode-url>/viewer (refresh once it says ready). Ctrl-C / pkill to stop.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--cam", default="wrist", help="which scene camera to stream (wrist | camera)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True  # livestream controlled via --livestream 2 on the CLI
app = AppLauncher(a).app

import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import Lift.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

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


# Point the streamed viewport at the actual wrist-camera prim. The cfg path is a
# regex template (e.g. "/World/envs/env_.*/Robot/panda_hand/wrist_cam"); resolve
# it to the concrete env-0 prim, falling back to a stage search.
import omni.usd  # noqa: E402
wrist_tmpl = env.scene.sensors[a.cam].cfg.prim_path
wrist_path = wrist_tmpl.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
_stage = omni.usd.get_context().get_stage()
if not _stage.GetPrimAtPath(wrist_path).IsValid():
    wrist_path = next((str(p.GetPath()) for p in _stage.Traverse() if p.GetName() == "wrist_cam"), wrist_path)
print("WRIST_PRIM", wrist_path, "valid=", _stage.GetPrimAtPath(wrist_path).IsValid(), flush=True)


def set_viewport_to_wrist():
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        if vp is None:
            return False
        vp.camera_path = wrist_path
        return True
    except Exception as e:  # noqa: BLE001
        print("VP_SET_RETRY", repr(e), flush=True)
        return False


# The viewport may not exist for the first few frames; step + retry.
done_set = False
for _ in range(60):
    obs, _, _, _, _ = wrapped.step(policy(obs))
    if not done_set and set_viewport_to_wrist():
        done_set = True
        print("VIEWPORT_SET", wrist_path, flush=True)
        break

print("WRIST_VIEW_READY", flush=True)
# Loop episodes forever; ManagerBasedRLEnv auto-resets terminated envs, so the
# arm keeps doing fresh pick-lifts (cube respawns) -> varied wrist motion/rotation.
while app.is_running():
    obs, _, _, _, _ = wrapped.step(policy(obs))
    if not done_set:  # in case the viewport came up late
        done_set = set_viewport_to_wrist()
app.close()
