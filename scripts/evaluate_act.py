# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""ACT evaluation script."""

import argparse
import os
import sys
import torch
import numpy as np

from isaaclab.app import AppLauncher

# Condition to task mapping
CONDITION_TASKS = {
    "baseline": "Template-Lift-Cube-Franka-Demo-Baseline-v0",
    "lighting": "Template-Lift-Cube-Franka-Demo-Lighting-v0",
    "texture": "Template-Lift-Cube-Franka-Demo-Texture-v0",
    "combined": "Template-Lift-Cube-Franka-Demo-Combined-v0",
}

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate ACT policy with vision in Isaac Lab")
parser.add_argument("--task", type=str, default=None, 
                    help="Task name (use demo task for camera)")
parser.add_argument("--condition", type=str, default=None,
                    choices=["baseline", "lighting", "texture", "combined", "all"],
                    help="Visual condition to evaluate (or 'all' for all conditions)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to ACT checkpoint")
parser.add_argument("--normalizer", type=str, default=None, help="Path to normalizer.pt (auto-detected if None)")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
parser.add_argument("--chunk_size", type=int, default=100, help="Action chunk size")
parser.add_argument("--temporal_agg", action="store_true", help="Use temporal aggregation")
parser.add_argument("--agg_weight", type=float, default=0.01, help="Temporal aggregation weight")
parser.add_argument("--goal_threshold", type=float, default=0.05, help="Distance threshold for success")
parser.add_argument("--output", type=str, default=None, help="Output directory for results (default: checkpoint dir)")
parser.add_argument("--save_trajectories", action="store_true", help="Save full per-step trajectory data")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Enable cameras
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import Lift.tasks  # noqa: F401

# Add ACT to path (repo root = parent of this script's dir; `act/` at <repo>/act).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "act"))
sys.path.insert(0, os.path.join(REPO_ROOT, "act", "detr"))
from policy import ACTPolicy as OriginalACTPolicy


# qpos indices matching training: joint_pos (7) + gripper from action
# Training uses: qpos = joint_pos (0-6) + last_action gripper (last dim of action)
QPOS_INDICES = list(range(0, 7))  # Joint positions from observation
STATE_DIM = 8  # 7 joint + 1 gripper


class ACTVisionPolicy:
    """Wrapper for ACT policy inference with vision in Isaac Lab."""
    
    def __init__(
        self,
        checkpoint_path: str,
        normalizer_path: str,
        device: str = "cuda:0",
        chunk_size: int = 100,
        temporal_agg: bool = False,
        agg_weight: float = 0.01,
    ):
        self.device = device
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.agg_weight = agg_weight
        
        # Load checkpoint (weights_only=False needed for numpy arrays in older checkpoints)
        print(f"Loading ACT checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load config.json from checkpoint directory
        import json
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = checkpoint.get('config', {})
        
        # Load normalizer (weights_only=False needed for numpy arrays)
        print(f"Loading normalizer: {normalizer_path}")
        normalizer = torch.load(normalizer_path, map_location=device, weights_only=False)
        self.qpos_mean = torch.tensor(normalizer['qpos_mean'], device=device, dtype=torch.float32)
        self.qpos_std = torch.tensor(normalizer['qpos_std'], device=device, dtype=torch.float32)
        self.action_mean = torch.tensor(normalizer['action_mean'], device=device, dtype=torch.float32)
        self.action_std = torch.tensor(normalizer['action_std'], device=device, dtype=torch.float32)
        
        # Build policy config for original ACT
        policy_config = {
            'lr': config.get('lr', 1e-5),
            'lr_backbone': config.get('lr_backbone', 1e-5),
            'backbone': config.get('backbone', 'resnet18'),
            'enc_layers': config.get('enc_layers', 4),
            'dec_layers': config.get('dec_layers', 7),
            'dim_feedforward': config.get('dim_feedforward', 3200),
            'hidden_dim': config.get('hidden_dim', 512),
            'dropout': config.get('dropout', 0.1),
            'nheads': config.get('nheads', 8),
            'num_queries': config.get('num_queries', chunk_size),
            'kl_weight': config.get('kl_weight', 10),
            'camera_names': config.get('camera_names', ['camera']),
            'state_dim': config.get('state_dim', STATE_DIM),  # Must match training
            'ckpt_dir': checkpoint_dir,
            'policy_class': 'ACT',
            'task_name': 'lift',
            'seed': 0,
            'num_epochs': 0,
        }
        
        # Create policy
        self.policy = OriginalACTPolicy(policy_config)
        self.policy.model.load_state_dict(checkpoint['model_state_dict'])
        self.policy.model.eval()
        
        # Action chunk buffer for temporal aggregation
        self.action_buffer = None
        self.buffer_idx = 0
        self.num_envs = 1
        
        print(f"ACT Vision Policy loaded (chunk_size={chunk_size}, temporal_agg={temporal_agg})")
    
    def reset(self, num_envs: int):
        """Reset policy state."""
        self.num_envs = num_envs
        # Action chunk buffer - recompute every chunk_size steps
        self.chunk_buffer = None  # Will hold (num_envs, chunk_size, action_dim)
        self.chunk_idx = 0  # Current index into chunk
        
        if self.temporal_agg:
            self.action_buffer = torch.zeros(
                num_envs, self.chunk_size, len(self.action_mean),
                device=self.device
            )
        self.buffer_idx = 0
    
    def normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        """Normalize qpos."""
        return (qpos - self.qpos_mean) / self.qpos_std
    
    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize actions."""
        return action * self.action_std + self.action_mean
    
    @torch.no_grad()
    def get_action(self, qpos: torch.Tensor, env) -> torch.Tensor:
        """Get action from policy with action chunking.
        
        Uses action chunking: only runs model every chunk_size steps.
        Only fetches camera images when model needs to be called.
        
        Args:
            qpos: (num_envs, qpos_dim) robot state
            env: The gym environment (to get images from when needed)
            
        Returns:
            (num_envs, action_dim) actions
        """
        # Check if we need to compute new action chunk
        need_new_chunk = (self.chunk_buffer is None or 
                         self.chunk_idx >= self.chunk_size)
        
        if need_new_chunk:
            # Only fetch images when we actually need them
            image = get_camera_images(env)
            if image is None:
                raise RuntimeError("No camera images available!")
            
            # Normalize qpos
            qpos_norm = self.normalize_qpos(qpos)
            
            # Process image to (batch, num_cam, C, H, W) float
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            # If shape is (B, num_cam, H, W, C), permute to (B, num_cam, C, H, W)
            if image.dim() == 5 and image.shape[-1] == 3:
                image = image.permute(0, 1, 4, 2, 3)
            # If shape is (B, H, W, C) - single camera legacy format
            elif image.dim() == 4 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2).unsqueeze(1)  # -> (B, 1, C, H, W)
            # If shape is (B, C, H, W) - single camera already CHW
            elif image.dim() == 4 and image.shape[1] == 3:
                image = image.unsqueeze(1)  # -> (B, 1, C, H, W)
            
            # Get action chunk from model
            # Original ACT expects: qpos (B, qpos_dim), image (B, num_cam, C, H, W), env_state (None for vision)
            # Model returns: (a_hat, is_pad_hat, [mu, logvar]) during inference
            model_output = self.policy.model(qpos_norm, image, env_state=None)
            
            # Extract action predictions (first element of tuple)
            if isinstance(model_output, tuple):
                action_chunk = model_output[0]  # a_hat: (B, chunk_size, action_dim)
            else:
                action_chunk = model_output
            
            # Denormalize and store in buffer
            self.chunk_buffer = self.denormalize_action(action_chunk)
            self.chunk_idx = 0
        
        # Get action from buffer (always use chunk buffer now)
        action = self.chunk_buffer[:, self.chunk_idx, :]
        self.chunk_idx += 1
        
        return action


def get_camera_images(env) -> torch.Tensor:
    """Extract camera RGB images from all cameras in the environment.
    
    Args:
        env: The unwrapped Isaac Lab environment
    
    Returns:
        images: (num_envs, num_cam, H, W, 3) uint8 tensor or None
    """
    if not hasattr(env.unwrapped, 'scene'):
        return None
        
    scene = env.unwrapped.scene
    
    # Collect images from all cameras
    camera_images = []
    camera_names = ['camera', 'wrist']  # Match training cameras (scene + wrist)
    
    for cam_name in camera_names:
        if cam_name in scene.sensors:
            camera = scene.sensors[cam_name]
            rgb_data = camera.data.output.get('rgb', None)
            if rgb_data is not None:
                # Convert to uint8 (H, W, 4) -> (H, W, 3)
                if rgb_data.dtype == torch.float32:
                    img = (rgb_data[..., :3] * 255).clamp(0, 255).to(torch.uint8)
                else:
                    img = rgb_data[..., :3]
                camera_images.append(img)
    
    if len(camera_images) == 0:
        print("Warning: No cameras found in scene!")
        return None
    
    # Stack cameras: list of (num_envs, H, W, 3) -> (num_envs, num_cam, H, W, 3)
    images = torch.stack(camera_images, dim=1)
    
    return images


def extract_qpos_from_obs(obs: torch.Tensor, last_action: torch.Tensor = None) -> torch.Tensor:
    """Extract qpos matching training: joint_pos (7) + gripper (1) = 8 dims.
    
    Full obs structure (36 dims):
    - joint_pos_rel: 0-6 (7)
    - joint_vel_rel: 7-13 (7)  
    - object_position: 14-16 (3)
    - etc.
    
    Training uses: qpos = joint_pos + gripper_from_action
    """
    joint_pos = obs[:, QPOS_INDICES]  # 7 dims
    
    # Get gripper from last action (last dim) or default to 0
    if last_action is not None:
        gripper = last_action[:, -1:]
    else:
        gripper = torch.zeros(obs.shape[0], 1, device=obs.device)
    
    return torch.cat([joint_pos, gripper], dim=-1)  # 8 dims


def compute_success(env, obs: torch.Tensor, goal_threshold: float = 0.05) -> torch.Tensor:
    """Compute success based on object-goal distance.
    
    Args:
        env: Isaac Lab environment
        obs: Full observation tensor
        goal_threshold: Distance threshold for success
        
    Returns:
        success: (num_envs,) bool tensor
    """
    # Observation structure (36 dims):
    # - joint_pos_rel: 0-8 (9 dims)
    # - joint_vel_rel: 9-17 (9 dims)
    # - object_position: 18-20 (3 dims)
    # - target_object_position: 21-27 (7 dims = pos + quat)
    # - last_action: 28-35 (8 dims)
    object_pos = obs[:, 18:21]  # object_position
    target_pos = obs[:, 21:24]  # target_object_position (XYZ only)
    
    distance = torch.norm(object_pos - target_pos, dim=-1)
    return distance < goal_threshold


def evaluate(env, policy, num_episodes: int, goal_threshold: float = 0.05, save_trajectories: bool = False):
    """Evaluate policy in environment."""
    
    # Get num_envs from unwrapped environment (gym wrappers don't expose it directly)
    num_envs = env.unwrapped.num_envs
    device = "cuda:0"
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    # Trajectory storage (if enabled)
    trajectories = [] if save_trajectories else None
    current_trajectories = [{
        'observations': [],
        'actions': [],
        'rewards': [],
        'object_positions': [],
        'target_positions': [],
        'goal_distances': [],
    } for _ in range(num_envs)] if save_trajectories else None
    
    episodes_completed = 0
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)
    current_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
    last_action = None  # Track last action for gripper state
    
    obs, _ = env.reset()
    policy.reset(num_envs)
    
    print(f"Evaluating for {num_episodes} episodes...")
    
    while episodes_completed < num_episodes:
        # Get observations
        if isinstance(obs, dict) and "policy" in obs:
            obs_tensor = obs["policy"]
        else:
            obs_tensor = obs
        
        # Extract qpos (joint_pos + gripper from last action)
        qpos = extract_qpos_from_obs(obs_tensor, last_action)
        
        # Get action from policy (images only fetched internally when needed)
        action = policy.get_action(qpos, env)
        last_action = action.clone()  # Save for next qpos extraction
        
        # Save trajectory data before step
        if save_trajectories:
            for i in range(num_envs):
                if episodes_completed + i < num_episodes:  # Only save if we need more episodes
                    obs_np = obs_tensor[i].cpu().numpy()
                    action_np = action[i].cpu().numpy()
                    object_pos = obs_np[18:21]  # object_position at indices 18-20
                    target_pos = obs_np[21:24]  # target XYZ at indices 21-23
                    goal_dist = np.linalg.norm(object_pos - target_pos)
                    
                    current_trajectories[i]['observations'].append(obs_np)
                    current_trajectories[i]['actions'].append(action_np)
                    current_trajectories[i]['object_positions'].append(object_pos)
                    current_trajectories[i]['target_positions'].append(target_pos)
                    current_trajectories[i]['goal_distances'].append(goal_dist)
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(action)
        dones = terminated | truncated
        
        current_rewards += rewards
        current_lengths += 1
        
        # Save rewards
        if save_trajectories:
            for i in range(num_envs):
                if episodes_completed + i < num_episodes:
                    current_trajectories[i]['rewards'].append(rewards[i].cpu().item())
        
        # Check for success (object at goal)
        if isinstance(obs, dict) and "policy" in obs:
            obs_for_success = obs["policy"]
        else:
            obs_for_success = obs
        
        success_now = compute_success(env, obs_for_success, goal_threshold)
        current_success = current_success | success_now
        
        # Check for episode completions
        done_indices = torch.where(dones)[0]
        
        for idx in done_indices:
            if episodes_completed < num_episodes:
                ep_reward = current_rewards[idx].item()
                ep_length = current_lengths[idx].item()
                ep_success = current_success[idx].item()
                
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                successes.append(ep_success)
                
                episodes_completed += 1
                
                if episodes_completed % 10 == 0:
                    mean_rew = np.mean(episode_rewards)
                    std_rew = np.std(episode_rewards)
                    print(f"Episodes: {episodes_completed}/{num_episodes}, "
                          f"Success rate: {np.mean(successes)*100:.1f}%, "
                          f"Mean Reward: {mean_rew:.4f} ± {std_rew:.4f}")
                
                # Save trajectory for this episode
                if save_trajectories:
                    traj = current_trajectories[idx.item()]
                    traj['total_reward'] = ep_reward
                    traj['length'] = int(ep_length)
                    traj['success'] = bool(ep_success)
                    # Convert lists to numpy arrays
                    traj['observations'] = np.array(traj['observations'])
                    traj['actions'] = np.array(traj['actions'])
                    traj['rewards'] = np.array(traj['rewards'])
                    traj['object_positions'] = np.array(traj['object_positions'])
                    traj['target_positions'] = np.array(traj['target_positions'])
                    traj['goal_distances'] = np.array(traj['goal_distances'])
                    trajectories.append(traj)
                    # Reset for next episode
                    current_trajectories[idx.item()] = {
                        'observations': [],
                        'actions': [],
                        'rewards': [],
                        'object_positions': [],
                        'target_positions': [],
                        'goal_distances': [],
                    }
            
            # Reset counters for this env
            current_rewards[idx] = 0
            current_lengths[idx] = 0
            current_success[idx] = False
            # Reset last_action gripper for this env
            if last_action is not None:
                last_action[idx] = 0
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes) * 100,
        'num_episodes': len(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': successes,
    }
    
    if save_trajectories:
        results['trajectories'] = trajectories
    
    return results


def main():
    # Auto-detect normalizer path if not provided
    normalizer_path = args_cli.normalizer
    if normalizer_path is None:
        checkpoint_dir = os.path.dirname(args_cli.checkpoint)
        normalizer_path = os.path.join(checkpoint_dir, "normalizer.pt")
    
    if not os.path.exists(normalizer_path):
        print(f"Error: Normalizer not found at {normalizer_path}")
        print("Please provide --normalizer path")
        return
    
    # Determine which conditions to evaluate
    if args_cli.condition == "all":
        conditions = ["baseline", "lighting", "texture", "combined"]
    elif args_cli.condition:
        conditions = [args_cli.condition]
    elif args_cli.task:
        conditions = [args_cli.task]  # Use task directly
    else:
        conditions = ["baseline"]  # Default
    
    # Create policy once (shared across conditions)
    policy = ACTVisionPolicy(
        checkpoint_path=args_cli.checkpoint,
        normalizer_path=normalizer_path,
        chunk_size=args_cli.chunk_size,
        temporal_agg=args_cli.temporal_agg,
        agg_weight=args_cli.agg_weight,
    )
    
    all_results = {}
    
    for condition in conditions:
        # Get task name
        if condition in CONDITION_TASKS:
            task = CONDITION_TASKS[condition]
        else:
            task = condition  # Assume it's a full task name
        
        print(f"\n{'='*50}")
        print(f"Evaluating: {condition}")
        print(f"Task: {task}")
        print(f"{'='*50}")
        
        # Create environment using Isaac Lab's helper
        from isaaclab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg(
            task,
            device="cuda:0",
            num_envs=args_cli.num_envs,
        )
        env = gym.make(task, cfg=env_cfg)
        
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Evaluate
        results = evaluate(env, policy, args_cli.num_episodes, args_cli.goal_threshold, 
                          save_trajectories=args_cli.save_trajectories)
        all_results[condition] = results
        
        import sys
        print(f"\nResults for {condition}:", flush=True)
        print(f"  Episodes: {results['num_episodes']}", flush=True)
        print(f"  Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}", flush=True)
        print(f"  Mean Length: {results['mean_length']:.1f}", flush=True)
        print(f"  Success Rate: {results['success_rate']:.1f}%", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Close environment before next condition
        env.close()
    
    # Print summary if multiple conditions
    if len(conditions) > 1:
        print(f"\n{'='*50}")
        print("SUMMARY - All Conditions")
        print(f"{'='*50}")
        print(f"{'Condition':<12} {'Success %':>10} {'Reward':>12}")
        print("-" * 36)
        for condition, results in all_results.items():
            print(f"{condition:<12} {results['success_rate']:>9.1f}% {results['mean_reward']:>12.4f}")
    
    # Save results
    output_dir = args_cli.output
    if output_dir is None:
        output_dir = os.path.dirname(args_cli.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Use a single eval_results.json per checkpoint, merge with existing
    checkpoint_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    summary_path = os.path.join(output_dir, f'eval_results_{checkpoint_name}.json')
    
    # Load existing results if present
    existing_results = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                existing_results = json.load(f)
            print(f"Merging with existing results from: {summary_path}")
        except:
            pass
    
    # Build new results
    new_results = {}
    for condition, results in all_results.items():
        new_results[condition] = {
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'mean_length': results['mean_length'],
            'success_rate': results['success_rate'],
            'num_episodes': results['num_episodes'],
            'episode_rewards': results['episode_rewards'],
            'episode_lengths': [int(x) for x in results['episode_lengths']],
            'episode_successes': [bool(x) for x in results['episode_successes']],
            'timestamp': timestamp,
        }
    
    # Merge: new results overwrite existing for same conditions
    merged_results = {**existing_results, **new_results}
    
    with open(summary_path, 'w') as f:
        json.dump(merged_results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
    
    # Save trajectories (NPZ) if enabled
    if args_cli.save_trajectories:
        for condition, results in all_results.items():
            if 'trajectories' in results:
                traj_path = os.path.join(output_dir, f'trajectories_{checkpoint_name}_{condition}.npz')
                # Convert to saveable format
                traj_data = {}
                for i, traj in enumerate(results['trajectories']):
                    traj_data[f'ep_{i}_observations'] = traj['observations']
                    traj_data[f'ep_{i}_actions'] = traj['actions']
                    traj_data[f'ep_{i}_rewards'] = traj['rewards']
                    traj_data[f'ep_{i}_object_positions'] = traj['object_positions']
                    traj_data[f'ep_{i}_target_positions'] = traj['target_positions']
                    traj_data[f'ep_{i}_goal_distances'] = traj['goal_distances']
                    traj_data[f'ep_{i}_success'] = np.array([traj['success']])
                    traj_data[f'ep_{i}_total_reward'] = np.array([traj['total_reward']])
                np.savez_compressed(traj_path, **traj_data)
                print(f"Trajectories saved to: {traj_path}")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
