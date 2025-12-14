#!/usr/bin/env python3
# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""View ACT policy evaluation with livestream."""

import argparse
import os
import sys

# Parse args BEFORE importing Isaac Lab
parser = argparse.ArgumentParser(description="View ACT model evaluation")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint (best_model.pt)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments (1 for clear viewing)")
parser.add_argument("--num_episodes", type=int, default=5,
                    help="Number of episodes to run")
parser.add_argument("--task", type=str, default="Template-Lift-Cube-Franka-Demo-Baseline-v0",
                    help="Task to evaluate on")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--livestream", type=int, default=2, choices=[0, 1, 2],
                    help="Livestream mode (2 = native)")
parser.add_argument("--enable_cameras", action="store_true", default=False,
                    help="Enable camera sensors")

# Parse known args, let AppLauncher handle the rest
args, unknown = parser.parse_known_args()

# Initialize Isaac Lab FIRST
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless, livestream=args.livestream, enable_cameras=True)
simulation_app = app_launcher.app

# Now we can import everything else
import json
import torch
import numpy as np

# Add ACT paths
sys.path.insert(0, "/workspace/frankapanda/Lift/act")
sys.path.insert(0, "/workspace/frankapanda/Lift/act/detr")

from policy import ACTPolicy

import isaaclab_tasks  # noqa: F401
import Lift.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym


def main():
    # Find checkpoint directory
    ckpt_dir = os.path.dirname(args.checkpoint)
    normalizer_path = os.path.join(ckpt_dir, 'normalizer.pt')
    config_path = os.path.join(ckpt_dir, 'config.json')
    
    if not os.path.exists(normalizer_path):
        print(f"Error: normalizer.pt not found in {ckpt_dir}")
        return
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {ckpt_dir}")
        return
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")
    print(f"Num episodes: {args.num_episodes}")
    print()
    
    # Load normalizer
    normalizer = torch.load(normalizer_path, weights_only=False)
    qpos_mean = torch.tensor(normalizer['qpos_mean'], dtype=torch.float32).cuda()
    qpos_std = torch.tensor(normalizer['qpos_std'], dtype=torch.float32).cuda()
    action_mean = torch.tensor(normalizer['action_mean'], dtype=torch.float32).cuda()
    action_std = torch.tensor(normalizer['action_std'], dtype=torch.float32).cuda()
    
    # Create policy
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
        'num_queries': config.get('num_queries', 100),
        'kl_weight': config.get('kl_weight', 10.0),
        'camera_names': config.get('camera_names', ['camera']),
        'state_dim': config.get('state_dim', 8),
        'ckpt_dir': ckpt_dir,
        'policy_class': 'ACT',
        'task_name': 'lift',
        'seed': 42,
        'num_epochs': 1,
    }
    
    print("Loading ACT policy...")
    policy = ACTPolicy(policy_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    policy.model.load_state_dict(checkpoint['model_state_dict'])
    policy.model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, loss={checkpoint.get('loss', '?'):.4f}")
    
    # Create environment
    print(f"\nCreating environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, use_fabric=True)
    env = gym.make(args.task, cfg=env_cfg)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\n" + "="*60)
    print("LIVESTREAM ACTIVE - Connect with Omniverse Streaming Client")
    print("="*60 + "\n")
    
    # qpos indices
    QPOS_INDICES = list(range(0, 7))  # Joint positions
    chunk_size = config.get('num_queries', 100)
    camera_names = config.get('camera_names', ['camera', 'camera2', 'camera3'])
    
    # Run episodes
    total_successes = 0
    
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        # Action chunking state
        action_buffer = None
        action_idx = 0
        
        last_gripper = torch.zeros(args.num_envs, 1, device='cuda')
        
        # Get camera sensors from scene
        camera_sensors = {}
        for cam_name in camera_names:
            if cam_name in env.unwrapped.scene.sensors:
                camera_sensors[cam_name] = env.unwrapped.scene.sensors[cam_name]
            elif hasattr(env.unwrapped.scene, cam_name):
                camera_sensors[cam_name] = getattr(env.unwrapped.scene, cam_name)
        
        if not camera_sensors:
            # Fallback to default camera
            camera_sensors['camera'] = env.unwrapped.scene["camera"]
        
        print(f"  Using cameras: {list(camera_sensors.keys())}")
        
        while not done and step < 300:
            # Get new action chunk if needed
            if action_buffer is None or action_idx >= chunk_size:
                # Prepare observation
                with torch.no_grad():
                    # Extract qpos (7 joint positions + gripper)
                    joint_pos = obs['policy'][:, QPOS_INDICES]  # (num_envs, 7)
                    qpos = torch.cat([joint_pos, last_gripper], dim=1)  # (num_envs, 8)
                    
                    # Normalize qpos
                    qpos_norm = (qpos - qpos_mean) / qpos_std
                    
                    # Get images from all cameras and stack
                    cam_images = []
                    for cam_name in camera_names:
                        if cam_name in camera_sensors:
                            cam_img = camera_sensors[cam_name].data.output["rgb"]  # (N, H, W, 4)
                            cam_img = cam_img[..., :3]  # (N, H, W, 3) RGB only
                            cam_img = cam_img.permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W)
                            cam_images.append(cam_img)
                    
                    # Stack cameras: (N, num_cam, C, H, W)
                    image = torch.stack(cam_images, dim=1)
                    
                    # Get action chunk from policy
                    action_buffer = policy(qpos_norm, image)  # (num_envs, chunk_size, action_dim)
                    
                    # Denormalize actions
                    action_buffer = action_buffer * action_std + action_mean
                    action_idx = 0
            
            # Get current action
            action = action_buffer[:, action_idx, :]  # (num_envs, 8)
            action_idx += 1
            
            # Update last gripper for next qpos
            last_gripper = action[:, -1:]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()
            episode_reward += reward.mean().item()
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}, reward so far: {episode_reward:.2f}")
        
        # Check success
        success = info.get('is_success', torch.zeros(args.num_envs)).any().item()
        total_successes += int(success)
        
        print(f"Episode {ep+1}: steps={step}, reward={episode_reward:.2f}, success={success}")
    
    print(f"\n{'='*60}")
    print(f"Results: {total_successes}/{args.num_episodes} successes ({100*total_successes/args.num_episodes:.1f}%)")
    print(f"{'='*60}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
