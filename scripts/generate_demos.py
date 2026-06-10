# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""Demo generation script using trained RL policy."""

import argparse
import os
import torch
import h5py
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# Visual condition mapping (the previously-unregistered "camera" condition removed)
CONDITION_TASKS = {
    "baseline": "Template-Lift-Cube-Franka-Demo-Baseline-v0",
    "lighting": "Template-Lift-Cube-Franka-Demo-Lighting-v0",
    "texture": "Template-Lift-Cube-Franka-Demo-Texture-v0",
    "combined": "Template-Lift-Cube-Franka-Demo-Combined-v0",
}

# Cameras to capture, in order. Index 0 ("camera") is the primary scene view
# stored under the backward-compatible `images` key; "wrist" is the hand-mounted
# camera. Must match visual_randomization.CAMERA_NAMES.
CAMERA_NAMES = ["camera", "wrist"]

# Add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations from trained RL policy")
parser.add_argument("--task", type=str, default=None, help="Task name (overrides --condition)")
parser.add_argument("--condition", type=str, default="baseline", 
                    choices=["baseline", "lighting", "texture", "combined", "all"],
                    help="Visual randomization condition")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--num_demos", type=int, default=100, help="Number of successful demos to collect")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--output_dir", type=str, default="data/demos", help="Output directory for demos")
parser.add_argument("--max_episode_length", type=int, default=250, help="Max steps per episode")
parser.add_argument("--save_images", action="store_true", default=True, help="Save camera images")
parser.add_argument("--min_length", type=int, default=20, help="Minimum episode length")
# Goal-based success criteria
parser.add_argument("--goal_threshold", type=float, default=0.05, 
                    help="Distance threshold (m) to consider object at goal position")
parser.add_argument("--min_goal_steps", type=int, default=10, 
                    help="Minimum consecutive steps object must be at goal")
parser.add_argument("--target_minutes", type=float, default=None,
                    help="If set, collect until total demo time (sum of successful episode "
                         "lengths) reaches this many minutes, overriding --num_demos as the stop "
                         "criterion. Use for the '>=30 min of oracle demos' requirement.")
# DART-style recovery-data collection: EXECUTE oracle_mean + OU noise on the ARM
# joints, but STORE the clean oracle mean as the action label. The state-feedback
# oracle keeps correcting the perturbed state, so successful episodes densely cover
# off-nominal approach states WITH corrective labels — exactly the data a success-
# filtered deterministic oracle never produces, and the reason a 1cm approach error
# at the grasp is currently unrecoverable for ACT.
parser.add_argument("--noise_std", type=float, default=0.0,
                    help="Stationary std of OU noise added to the EXECUTED arm action "
                         "(raw action units; label stays the clean oracle mean). 0 = off.")
parser.add_argument("--noise_theta", type=float, default=0.15,
                    help="OU mean-reversion rate (correlation time ~1/theta steps)")
# Short, task-dense demos: the oracle finishes by ~step 62 worst case and the success
# latch (10 consecutive at-goal steps) lands by ~72, yet episodes run 250 steps — so
# ~82% of every recorded demo is the cube parked at the goal. Shorter episodes raise
# collection throughput; cutting the recording shortly after the success latch keeps
# the placement approach + a stabilize-at-goal beat without the long hover tail. The
# success filter still gates every demo, so nothing incomplete can be recorded.
parser.add_argument("--episode_seconds", type=float, default=None,
                    help="Override env episode length (s) for collection (default: task cfg)")
parser.add_argument("--end_after_goal", type=int, default=-1,
                    help="Cut each RECORDED demo N steps after the success latch (object at "
                         "goal for min_goal_steps consecutive steps). -1 = record full episode.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set task based on condition if not explicitly provided
if args_cli.task is None:
    if args_cli.condition != "all":
        args_cli.task = CONDITION_TASKS[args_cli.condition]

# Enable cameras for image capture
if args_cli.save_images:
    args_cli.enable_cameras = True
    # MUST render headless/offscreen. On current Isaac builds, GUI mode
    # (headless=False) silently returns a frozen frame-0 image for every step
    # (images decoupled from the moving robot). eval_checkpoints / verify_cams
    # set headless=True for the same reason; without it the demos are garbage.
    args_cli.headless = True

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import gymnasium as gym
import os
import torch
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg

import Lift.tasks  # noqa: F401


def load_skrl_agent_from_runner(checkpoint_path: str, env, experiment_cfg: dict):
    """Load a trained skrl PPO agent."""
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    
    # Configure the runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    
    # Create runner which builds the agent with correct architecture
    runner = Runner(wrapped_env, experiment_cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")
    
    return runner.agent, wrapped_env


def get_camera_images(env) -> dict:
    """Extract camera RGB images from environment."""
    if not hasattr(env, 'scene'):
        return None
        
    scene = env.scene
    
    def extract_rgb(camera):
        """Extract RGB data from a camera sensor."""
        rgb_data = camera.data.output.get('rgb', None)
        if rgb_data is None:
            return None
        
        if isinstance(rgb_data, torch.Tensor):
            if rgb_data.dtype == torch.float32:
                return (rgb_data[..., :3] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            else:
                return rgb_data[..., :3].cpu().numpy()
        else:
            return rgb_data[..., :3]
    
    images = {}
    
    # Try to get cameras from scene.sensors dict
    if hasattr(scene, 'sensors'):
        for name, sensor in scene.sensors.items():
            if name in CAMERA_NAMES:
                rgb = extract_rgb(sensor)
                if rgb is not None:
                    images[name] = rgb
    
    # Fallback: also check direct scene attributes for the configured cameras
    for attr in CAMERA_NAMES:
        if hasattr(scene, attr):
            camera = getattr(scene, attr)
            if camera is not None and hasattr(camera, 'data'):
                rgb = extract_rgb(camera)
                if rgb is not None:
                    images[attr] = rgb
    
    return images if images else None


def collect_demos(env, agent, num_demos: int, save_images: bool, min_reward: float, min_length: int,
                  unwrapped_env=None, goal_threshold: float = 0.05, min_goal_steps: int = 10,
                  condition: str = "baseline", target_minutes: float = None, output_path: str = None,
                  noise_std: float = 0.0, noise_theta: float = 0.15,
                  end_after_goal: int = -1):
    """Collect successful demonstrations from the environment.
    
    Args:
        env: The wrapped environment for stepping
        agent: The trained agent
        num_demos: Number of demos to collect
        save_images: Whether to save camera images
        min_reward: Minimum reward (unused, kept for compatibility)
        min_length: Minimum episode length
        unwrapped_env: Optional unwrapped environment for camera access
        goal_threshold: Distance threshold to consider object at goal (meters)
        min_goal_steps: Minimum steps object must be at goal position
        condition: Visual randomization condition (for lighting changes)
    
    Success criteria: Object must reach within goal_threshold of target position
    and stay there for at least min_goal_steps consecutive steps.
    
    Observation structure (36 dims total):
        - joint_pos: indices 0-8 (9 dims)
        - joint_vel: indices 9-17 (9 dims)  
        - object_position: indices 18-20 (3 dims) - object XYZ in robot frame
        - target_object_position: indices 21-27 (7 dims) - target pose (XYZ + quaternion)
        - actions: indices 28-35 (8 dims)
    """
    
    # Import lighting randomization if needed
    set_random_lighting = None
    if condition in ["lighting", "combined"]:
        try:
            from Lift.tasks.manager_based.lift.config.franka.visual_randomization import set_random_lighting
            print("Lighting randomization enabled - will change per demo")
            # Set initial random lighting
            lighting_params = set_random_lighting()
            if lighting_params:
                c = lighting_params['color']
                print(f"  Initial lighting: intensity={lighting_params['intensity']:.0f}, "
                      f"color=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
        except ImportError:
            print("Warning: Could not import set_random_lighting function")
    
    # Import texture randomization if needed
    set_random_texture = None
    if condition in ["texture", "combined"]:
        try:
            from Lift.tasks.manager_based.lift.config.franka.visual_randomization import set_random_texture
            print("Texture randomization enabled - each env gets different material per demo")
            # Set initial random texture (different for each env)
            texture_params = set_random_texture()
            if texture_params:
                c = texture_params['color']
                print(f"  Initial cube (env_0): color=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), "
                      f"roughness={texture_params['roughness']:.2f}, metallic={texture_params['metallic']:.2f})")
        except ImportError:
            print("Warning: Could not import set_random_texture function")
    
    # Incremental HDF5 writer: write each demo to disk as it is collected and free
    # it from RAM, so collection memory stays ~flat (only the in-progress episodes),
    # instead of holding the whole dataset (~0.46 MB/timestep) in RAM and OOMing.
    h5f = None
    written = 0
    stat_n = 0
    obs_sum = obs_sumsq = act_sum = act_sumsq = None
    total_episodes = 0
    successful_episodes = 0
    
    collected_timesteps = 0  # sum of successful-episode lengths (for --target_minutes)
    base_env = unwrapped_env if unwrapped_env is not None else getattr(env, "unwrapped", env)
    step_dt = getattr(base_env, "step_dt", 0.02)  # control dt = sim.dt * decimation (50 Hz -> 0.02)
    target_timesteps = int(round(target_minutes * 60.0 / step_dt)) if target_minutes else None

    def reached_target():
        if target_timesteps is not None:
            return collected_timesteps >= target_timesteps
        return successful_episodes >= num_demos

    if target_timesteps is not None:
        print(f"Collecting >= {target_minutes:.1f} min of demos "
              f"(~{target_timesteps} timesteps at {1.0/step_dt:.0f} Hz)...")
    else:
        print(f"Collecting {num_demos} successful demonstrations...")
    print(f"Success criteria: object within {goal_threshold}m of goal for {min_goal_steps}+ steps")
    if save_images:
        print("Camera image recording enabled")
    
    # Use unwrapped env for camera if provided, else try to get it
    camera_env = unwrapped_env if unwrapped_env is not None else getattr(env, 'unwrapped', env)
    
    # Check what cameras are available
    camera_names = []
    if save_images:
        test_images = get_camera_images(camera_env)
        if test_images:
            camera_names = list(test_images.keys())
            print(f"Found {len(camera_names)} camera(s): {camera_names}")

    # Open the incremental HDF5 now that camera_names is known.
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        h5f = h5py.File(output_path, 'w')
        h5f.attrs['timestamp'] = datetime.now().isoformat()
        h5f.attrs['condition'] = condition
        h5f.attrs['noise_std'] = noise_std
        h5f.attrs['noise_theta'] = noise_theta
        if camera_names:
            h5f.attrs['camera_names'] = ','.join(camera_names)
        h5f.attrs['has_images'] = bool(save_images and camera_names)
        print(f"Writing demos incrementally to {output_path}")

    # Storage for current episodes (per env)
    num_envs = env.num_envs
    episode_data = [{
        "observations": [],
        "actions": [],
        "rewards": [],
        "images": {name: [] for name in camera_names} if save_images else None,
        "goal_steps": 0,  # Track consecutive steps at goal
        "reached_goal": False,  # Track if goal was reached
    } for _ in range(num_envs)]
    
    obs, _ = env.reset()

    # OU noise state for DART-style recovery collection (arm joints only — the
    # gripper sign must stay the oracle's clean open/close signal). Stationary
    # std = noise_std, correlation time ~1/noise_theta steps.
    ou_state = None
    ou_sigma_w = 0.0
    if noise_std > 0:
        ou_sigma_w = noise_std * float(np.sqrt(1.0 - (1.0 - noise_theta) ** 2))
        print(f"OU noise on EXECUTED arm actions: std={noise_std}, theta={noise_theta} "
              f"(labels stay the clean oracle mean)")

    while not reached_target():
        # Get action from the policy. Use the DETERMINISTIC MEAN action (the
        # oracle's intended action), NOT a stochastic sample: sampling injects
        # per-step noise (~scale * policy_std) into the absolute joint targets,
        # which would be baked into the demos and imitated as jitter by ACT.
        with torch.no_grad():
            act_out = agent.act(obs, timestep=0, timesteps=0)
            if len(act_out) > 2 and isinstance(act_out[2], dict) and act_out[2].get("mean_actions") is not None:
                action = act_out[2]["mean_actions"]
            else:
                action = act_out[0]

        # Perturb the EXECUTED action only; `action` (the stored label) stays clean.
        exec_action = action
        if noise_std > 0:
            if ou_state is None:
                ou_state = torch.zeros_like(action[:, :7])
            ou_state = (1.0 - noise_theta) * ou_state + ou_sigma_w * torch.randn_like(ou_state)
            exec_action = action.clone()
            exec_action[:, :7] += ou_state
        
        # Get camera images before storing (from unwrapped env)
        images = get_camera_images(camera_env) if save_images else None
        
        # Handle observation format (may be dict or tensor depending on wrapper)
        if isinstance(obs, dict) and "policy" in obs:
            obs_tensor = obs["policy"]
        else:
            obs_tensor = obs
        
        # Store current observation, action, and image for all envs
        # Also check if object is at goal position
        for i in range(num_envs):
            obs_np = obs_tensor[i].cpu().numpy() if isinstance(obs_tensor, torch.Tensor) else obs_tensor[i]
            action_np = action[i].cpu().numpy() if isinstance(action, torch.Tensor) else action[i]
            
            # Extract object and target positions from observation
            # object_position: indices 18-20 (3 dims)
            # target_object_position: indices 21-23 (first 3 of 7 dims = XYZ)
            object_pos = obs_np[18:21]
            target_pos = obs_np[21:24]  # Just XYZ, ignore quaternion
            
            # Calculate distance to goal
            distance_to_goal = np.linalg.norm(object_pos - target_pos)
            
            # Track goal reaching
            if distance_to_goal < goal_threshold:
                episode_data[i]["goal_steps"] += 1
                if episode_data[i]["goal_steps"] >= min_goal_steps:
                    episode_data[i]["reached_goal"] = True
            else:
                episode_data[i]["goal_steps"] = 0  # Reset if moved away
            
            episode_data[i]["observations"].append(obs_np)
            episode_data[i]["actions"].append(action_np)
            if save_images and images is not None:
                for cam_name in camera_names:
                    if cam_name in images:
                        episode_data[i]["images"][cam_name].append(images[cam_name][i].copy())
        
        # Step environment with the (possibly noise-perturbed) executed action
        obs, rewards, terminated, truncated, info = env.step(exec_action)
        dones = terminated | truncated
        if noise_std > 0 and dones.any():
            # fresh noise state for envs that just reset
            ou_state[torch.where(dones)[0]] = 0.0
        
        # Store rewards
        for i in range(num_envs):
            episode_data[i]["rewards"].append(rewards[i].cpu().numpy())
        
        # Check for episode completions
        done_indices = torch.where(dones)[0].cpu().numpy()
        
        for idx in done_indices:
            total_episodes += 1
            ep_data = episode_data[idx]
            
            total_reward = float(np.sum(ep_data["rewards"]))
            episode_length = len(ep_data["observations"])
            reached_goal = ep_data["reached_goal"]
            
            # SUCCESS: Object reached goal position and stayed there
            if reached_goal and episode_length >= min_length:
                successful_episodes += 1

                obs_arr = np.array(ep_data["observations"])
                # Optionally cut the recording shortly after the success latch: find the
                # first step where the object has been within goal_threshold of the goal
                # for min_goal_steps consecutive steps, keep end_after_goal more steps.
                t_end = episode_length
                if end_after_goal >= 0:
                    dist = np.linalg.norm(obs_arr[:, 18:21] - obs_arr[:, 21:24], axis=1)
                    at_goal = dist < goal_threshold
                    run_len = 0
                    for t in range(len(at_goal)):
                        run_len = run_len + 1 if at_goal[t] else 0
                        if run_len >= min_goal_steps:
                            t_end = min(episode_length, t + 1 + end_after_goal)
                            break
                episode_length = t_end
                collected_timesteps += episode_length

                demo = {
                    "observations": obs_arr[:t_end],
                    "actions": np.array(ep_data["actions"])[:t_end],
                    "rewards": np.array(ep_data["rewards"])[:t_end],
                    "length": episode_length,
                    "total_reward": total_reward,
                    "reached_goal": True,
                }

                # Store images from all cameras
                if save_images and ep_data["images"]:
                    # Store each camera as separate key, and primary as 'images' for backwards compat
                    for cam_name, cam_images in ep_data["images"].items():
                        if cam_images:
                            demo[f"images_{cam_name}"] = np.array(cam_images[:t_end])
                    # Use first camera as default 'images' for backwards compatibility
                    if camera_names and ep_data["images"].get(camera_names[0]):
                        demo["images"] = np.array(ep_data["images"][camera_names[0]][:t_end])
                
                # Write this demo straight to disk and let it fall out of RAM
                # (instead of accumulating the whole dataset in a list).
                if h5f is not None:
                    grp = h5f.create_group(f'demo_{written}')
                    grp.create_dataset('observations', data=demo['observations'], compression='gzip')
                    grp.create_dataset('actions', data=demo['actions'], compression='gzip')
                    grp.create_dataset('rewards', data=demo['rewards'], compression='gzip')
                    grp.attrs['length'] = demo['length']
                    grp.attrs['total_reward'] = demo['total_reward']
                    grp.attrs['success'] = True
                    # lzf (fast, low-CPU) not gzip: these RGB frames barely gzip-
                    # compress, so gzip-4 just burns the scarce CPU and spikes load.
                    if save_images and 'images' in demo:
                        grp.create_dataset('images', data=demo['images'], compression='lzf')
                    for cam_name in camera_names:
                        k = f'images_{cam_name}'
                        if k in demo:
                            grp.create_dataset(k, data=demo[k], compression='lzf')
                    h5f.flush()
                    # Running stats so we don't have to keep all obs/actions in RAM.
                    o = demo['observations'].astype(np.float64)
                    a = demo['actions'].astype(np.float64)
                    if obs_sum is None:
                        obs_sum, obs_sumsq = o.sum(0), (o * o).sum(0)
                        act_sum, act_sumsq = a.sum(0), (a * a).sum(0)
                    else:
                        obs_sum += o.sum(0); obs_sumsq += (o * o).sum(0)
                        act_sum += a.sum(0); act_sumsq += (a * a).sum(0)
                    stat_n += len(o)
                written += 1

                # Build image info string
                img_info = ""
                if save_images and camera_names:
                    shapes = [f"{name}={demo.get(f'images_{name}', np.array([])).shape}" for name in camera_names]
                    img_info = f", {', '.join(shapes)}"
                progress = (f"{collected_timesteps*step_dt/60.0:.1f}/{target_minutes:.1f} min"
                            if target_timesteps is not None else f"{successful_episodes}/{num_demos} demos")
                print(f"Demo collected [{progress}] "
                      f"(length={episode_length}, reward={total_reward:.2f}, GOAL REACHED{img_info})")
                
                # Change lighting for next demo (if lighting randomization enabled)
                if set_random_lighting is not None:
                    lighting_params = set_random_lighting()
                    if lighting_params:
                        c = lighting_params['color']
                        print(f"  New lighting: intensity={lighting_params['intensity']:.0f}, "
                              f"color=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
                
                # Change texture for next demo (if texture randomization enabled)
                if set_random_texture is not None:
                    texture_params = set_random_texture()
                    if texture_params:
                        c = texture_params['color']
                        print(f"  New cube (env_0): color=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), "
                              f"roughness={texture_params['roughness']:.2f}, metallic={texture_params['metallic']:.2f})")
                
                if reached_target():
                    break
            else:
                # Failed episode - didn't reach goal
                if total_episodes % 10 == 0:
                    print(f"  Episode {total_episodes}: FAILED (length={episode_length}, "
                          f"reward={total_reward:.2f}, reached_goal={reached_goal})")
            
            # Reset episode storage for this env
            episode_data[idx] = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "images": {name: [] for name in camera_names} if save_images else None,
                "goal_steps": 0,
                "reached_goal": False,
            }
        
        # Print progress periodically
        if total_episodes % 50 == 0 and total_episodes > 0:
            success_rate = successful_episodes / total_episodes * 100
            print(f"Progress: {successful_episodes}/{num_demos} demos "
                  f"({total_episodes} total episodes, {success_rate:.1f}% success rate)")
    
    print(f"\nCollection complete!")
    print(f"Total episodes: {total_episodes}")
    print(f"Successful demos: {successful_episodes}")
    if total_episodes:
        print(f"Success rate: {successful_episodes/total_episodes*100:.1f}%")

    if h5f is not None:
        h5f.attrs['num_demos'] = written
        h5f.close()
        size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
        print(f"Saved {written} demos incrementally to {output_path} ({size_mb:.1f} MB)")
        if stat_n > 0:
            act_mean = act_sum / stat_n
            act_std = np.sqrt(np.maximum(act_sumsq / stat_n - act_mean ** 2, 0))
            print(f"  Total timesteps: {stat_n}")
            print(f"  Action mean: {act_mean}")
            print(f"  Action std: {act_std}")
    return written


def save_demos_hdf5(demos: list, output_path: str, condition: str, save_images: bool):
    """Save demonstrations to HDF5 format for ACT training."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Detect camera names from first demo
    camera_names = []
    if save_images and demos:
        for key in demos[0].keys():
            if key.startswith('images_'):
                camera_names.append(key.replace('images_', ''))
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata
        f.attrs['num_demos'] = len(demos)
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['condition'] = condition
        f.attrs['has_images'] = save_images and 'images' in demos[0]
        if camera_names:
            f.attrs['camera_names'] = ','.join(camera_names)
        
        for i, demo in enumerate(demos):
            grp = f.create_group(f'demo_{i}')
            grp.create_dataset('observations', data=demo['observations'], compression='gzip')
            grp.create_dataset('actions', data=demo['actions'], compression='gzip')
            grp.create_dataset('rewards', data=demo['rewards'], compression='gzip')
            grp.attrs['length'] = demo['length']
            grp.attrs['total_reward'] = demo['total_reward']
            grp.attrs['success'] = demo.get('reached_goal', True)
            
            # Save primary images (backwards compatible)
            if save_images and 'images' in demo:
                grp.create_dataset('images', data=demo['images'], 
                                   compression='gzip', compression_opts=4)
            
            # Save additional camera images
            for cam_name in camera_names:
                key = f'images_{cam_name}'
                if key in demo:
                    grp.create_dataset(key, data=demo[key],
                                       compression='gzip', compression_opts=4)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {len(demos)} demos to {output_path} ({file_size_mb:.1f} MB)")
    
    # Print dataset statistics
    all_obs = np.concatenate([d['observations'] for d in demos])
    all_actions = np.concatenate([d['actions'] for d in demos])
    
    print(f"\nDataset statistics:")
    print(f"  Condition: {condition}")
    print(f"  Total timesteps: {len(all_obs)}")
    print(f"  Observation shape: {demos[0]['observations'].shape}")
    print(f"  Action shape: {demos[0]['actions'].shape}")
    
    if save_images and 'images' in demos[0]:
        print(f"  Primary image shape: {demos[0]['images'].shape}")
        total_images = sum(d['images'].shape[0] for d in demos if 'images' in d)
        print(f"  Total images: {total_images}")
    
    if camera_names:
        print(f"  Cameras: {camera_names}")
        for cam_name in camera_names:
            key = f'images_{cam_name}'
            if key in demos[0]:
                print(f"    {cam_name}: {demos[0][key].shape}")
    
    print(f"  Obs mean: {all_obs.mean(axis=0)[:5]}...")
    print(f"  Obs std: {all_obs.std(axis=0)[:5]}...")
    print(f"  Action mean: {all_actions.mean(axis=0)}")
    print(f"  Action std: {all_actions.std(axis=0)}")


def create_env(task: str, num_envs: int, device: str = "cuda:0", episode_seconds: float = None):
    """Create Isaac Lab environment from task name."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.envs import ManagerBasedRLEnv
    
    # Load the environment config from registry
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    if episode_seconds is not None:
        env_cfg.episode_length_s = episode_seconds
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    return env


def load_experiment_cfg(task: str):
    """Load the experiment configuration for a task."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    
    # Load the skrl config from the task registry
    agent_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
    return agent_cfg


def generate_for_condition(condition: str, checkpoint: str, num_demos: int,
                            num_envs: int, output_dir: str, save_images: bool,
                            min_length: int, goal_threshold: float, min_goal_steps: int,
                            target_minutes: float = None,
                            noise_std: float = 0.0, noise_theta: float = 0.15,
                            episode_seconds: float = None, end_after_goal: int = -1):
    """Generate demos for a single visual condition."""
    
    task = CONDITION_TASKS[condition]
    print(f"\n{'='*60}")
    print(f"Generating demos for condition: {condition.upper()}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")
    
    # Create environment using Isaac Lab's method
    env = create_env(task, num_envs, episode_seconds=episode_seconds)
    
    print(f"Environment: {task}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Num envs: {env.num_envs}")
    
    # Load experiment config and trained agent using Runner (same as play.py)
    print(f"\nLoading checkpoint: {checkpoint}")
    experiment_cfg = load_experiment_cfg(task)
    agent, wrapped_env = load_skrl_agent_from_runner(checkpoint, env, experiment_cfg)
    
    # Collect demonstrations, written incrementally to disk (low, flat RAM).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{condition}_noise{noise_std:g}" if noise_std > 0 else condition
    output_path = os.path.join(output_dir, f"demos_{tag}_{timestamp}.hdf5")
    collect_demos(
        env=wrapped_env,
        agent=agent,
        num_demos=num_demos,
        save_images=save_images,
        min_reward=0.0,  # Not used anymore, kept for compatibility
        min_length=min_length,
        unwrapped_env=env,  # Pass unwrapped for camera access
        goal_threshold=goal_threshold,
        min_goal_steps=min_goal_steps,
        condition=condition,  # Pass condition for lighting randomization
        target_minutes=target_minutes,
        output_path=output_path,
        noise_std=noise_std,
        noise_theta=noise_theta,
        end_after_goal=end_after_goal,
    )

    # Cleanup
    env.close()
    
    return output_path


def main():
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    if args_cli.condition == "all":
        # Generate demos for all conditions
        print("Generating demos for ALL visual conditions...")
        conditions = ["baseline", "lighting", "texture", "combined"]
        output_paths = []
        
        for condition in conditions:
            path = generate_for_condition(
                condition=condition,
                checkpoint=args_cli.checkpoint,
                num_demos=args_cli.num_demos,
                num_envs=args_cli.num_envs,
                output_dir=args_cli.output_dir,
                save_images=args_cli.save_images,
                min_length=args_cli.min_length,
                goal_threshold=args_cli.goal_threshold,
                min_goal_steps=args_cli.min_goal_steps,
                target_minutes=args_cli.target_minutes,
                noise_std=args_cli.noise_std,
                noise_theta=args_cli.noise_theta,
                episode_seconds=args_cli.episode_seconds,
                end_after_goal=args_cli.end_after_goal,
            )
            output_paths.append(path)
        
        print(f"\n{'='*60}")
        print("ALL CONDITIONS COMPLETE")
        print(f"{'='*60}")
        for path in output_paths:
            print(f"  {path}")
    else:
        # Generate for single condition
        generate_for_condition(
            condition=args_cli.condition,
            checkpoint=args_cli.checkpoint,
            num_demos=args_cli.num_demos,
            num_envs=args_cli.num_envs,
            output_dir=args_cli.output_dir,
            save_images=args_cli.save_images,
            min_length=args_cli.min_length,
            goal_threshold=args_cli.goal_threshold,
            min_goal_steps=args_cli.min_goal_steps,
            target_minutes=args_cli.target_minutes,
            noise_std=args_cli.noise_std,
            noise_theta=args_cli.noise_theta,
            episode_seconds=args_cli.episode_seconds,
            end_after_goal=args_cli.end_after_goal,
        )

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()