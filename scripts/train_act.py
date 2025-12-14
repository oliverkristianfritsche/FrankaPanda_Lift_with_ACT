# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""ACT training script."""

import argparse
import os
import sys
import json
import glob
import torch
import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader

# Add ACT (original repo) to path
sys.path.insert(0, "/workspace/frankapanda/Lift/act")
sys.path.insert(0, "/workspace/frankapanda/Lift/act/detr")
from policy import ACTPolicy


# ACT state/action dimensions for Franka
# qpos: 7 joint positions + 1 gripper = 8 dims
# actions: 8 dims (same as qpos)
STATE_DIM = 8  # joint_pos (7) + gripper (1)
QPOS_INDICES = list(range(0, 7))  # Joint positions from observation
GRIPPER_INDEX = 7  # Gripper from action (we'll use last action's gripper for qpos)


def parse_args():
    parser = argparse.ArgumentParser(description="ACT Training")
    
    # Data
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to demo HDF5 file or glob pattern")
    parser.add_argument("--output_dir", type=str, default="/workspace/frankapanda/Lift/logs/act/checkpoints",
                        help="Output directory for checkpoints")
    
    # Experiment settings
    parser.add_argument("--max_demos", type=int, default=None,
                        help="Maximum number of demos to use (None = all)")
    parser.add_argument("--eval_interval", type=int, default=0,
                        help="Evaluate every N epochs (0 = no eval during training)")
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="Episodes per evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed evaluation output")
    
    # Model architecture (paper defaults)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--backbone", type=str, default="resnet18")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="Save checkpoint every N epochs")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 avoids shared memory issues)")
    parser.add_argument("--num_eval_envs", type=int, default=16)
    parser.add_argument("--sample_per_timestep", action="store_true",
                        help="Sample every timestep instead of random timestep per episode (slower)")
    
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from latest checkpoint in output_dir")
    return parser.parse_args()


class ACTDataset(Dataset):
    """Dataset for ACT training with images."""
    
    def __init__(
        self,
        data_path: str,
        chunk_size: int = 100,
        max_demos: Optional[int] = None,
        sample_per_timestep: bool = False,  # False = original ACT style (per episode)
        camera_names: Optional[list] = None,  # List of camera names to use
    ):
        self.chunk_size = chunk_size
        self.sample_per_timestep = sample_per_timestep
        
        self.qpos_data = []
        self.image_data = []  # Will be list of dicts: {cam_name: images}
        self.action_data = []
        
        # Find files
        if '*' in data_path:
            files = sorted(glob.glob(data_path))
        else:
            files = [data_path]
        
        if len(files) == 0:
            raise ValueError(f"No files found matching: {data_path}")
        
        # Detect camera names from first file if not specified
        if camera_names is None:
            with h5py.File(files[0], 'r') as f:
                # Check for camera_names attribute
                if 'camera_names' in f.attrs:
                    camera_names = f.attrs['camera_names'].split(',')
                else:
                    # Auto-detect from first demo
                    demo = f['demo_0']
                    camera_names = []
                    if 'images' in demo:
                        camera_names.append('camera')
                    for key in demo.keys():
                        if key.startswith('images_'):
                            camera_names.append(key.replace('images_', ''))
                    if not camera_names:
                        camera_names = ['camera']
        
        self.camera_names = camera_names
        print(f"Using cameras: {self.camera_names}")
        
        total_demos = 0
        total_timesteps = 0
        
        for fpath in files:
            print(f"Loading {fpath}...")
            with h5py.File(fpath, 'r') as f:
                num_demos = f.attrs.get('num_demos', len([k for k in f.keys() if k.startswith('demo_')]))
                if max_demos:
                    num_demos = min(num_demos, max_demos - total_demos)
                
                pbar = tqdm(range(num_demos), desc="Loading demos", unit="demo")
                for i in pbar:
                    if max_demos and total_demos >= max_demos:
                        break
                    
                    demo = f[f'demo_{i}']
                    
                    # Get observations - only joint positions for qpos (like original ACT)
                    full_obs = demo['observations'][:]  # (T, 33)
                    actions_raw = demo['actions'][:].astype(np.float32)  # (T, 8)
                    
                    # qpos = joint_pos (7) + gripper (1) = 8 dims
                    # Use joint positions from obs, gripper from action
                    joint_pos = full_obs[:, QPOS_INDICES].astype(np.float32)  # 7 dims
                    gripper = actions_raw[:, -1:].astype(np.float32)  # 1 dim from action
                    qpos = np.concatenate([joint_pos, gripper], axis=1)  # 8 dims
                    
                    # Get images from all cameras
                    demo_images = {}
                    for cam_name in self.camera_names:
                        if cam_name == 'camera' and 'images' in demo:
                            demo_images[cam_name] = demo['images'][:]
                        elif f'images_{cam_name}' in demo:
                            demo_images[cam_name] = demo[f'images_{cam_name}'][:]
                        else:
                            raise ValueError(f"No images found for camera '{cam_name}' in {fpath}/demo_{i}")
                    
                    # Actions are already 8 dims
                    actions = actions_raw  # 8 dims
                    
                    self.qpos_data.append(qpos)
                    self.image_data.append(demo_images)
                    self.action_data.append(actions)
                    
                    total_demos += 1
                    total_timesteps += len(qpos)
                    pbar.set_postfix({'demos': total_demos, 'timesteps': total_timesteps})
        
        self.num_demos = total_demos
        
        # Compute normalization stats
        all_qpos = np.concatenate(self.qpos_data, axis=0)
        all_actions = np.concatenate(self.action_data, axis=0)
        
        self.qpos_dim = all_qpos.shape[1]
        self.action_dim = all_actions.shape[1]
        
        self.qpos_mean = all_qpos.mean(axis=0)
        self.qpos_std = np.clip(all_qpos.std(axis=0), 1e-2, np.inf)  # Original ACT clips to 1e-2
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = np.clip(all_actions.std(axis=0), 1e-2, np.inf)  # Original ACT clips to 1e-2
        
        # Build valid indices BEFORE converting to tensors (uses numpy data)
        if sample_per_timestep:
            self.valid_indices = self._build_valid_indices_numpy()
        
        # LAZY LOADING: Keep data on CPU as compact as possible
        # Images stay as uint8 (1 byte vs 4 bytes for float32) - ~2.4GB for 200 demos
        print("Preprocessing data (minimal memory mode)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pre-normalize qpos and actions on CPU (small, ~3MB total)
        qpos_mean = self.qpos_mean.astype(np.float32)
        qpos_std = self.qpos_std.astype(np.float32)
        action_mean = self.action_mean.astype(np.float32)
        action_std = self.action_std.astype(np.float32)
        
        self.qpos_normalized = [(q.astype(np.float32) - qpos_mean) / qpos_std for q in self.qpos_data]
        self.actions_normalized = [(a.astype(np.float32) - action_mean) / action_std for a in self.action_data]
        
        # Keep images as-is (uint8, THWC format) - transpose in __getitem__
        # This avoids creating a second copy during preprocessing
        self.images_raw = self.image_data  # Just rename, no copy
        
        # Free other arrays
        del self.qpos_data, self.action_data, self.image_data
        
        if sample_per_timestep:
            print(f"Training samples per epoch: {len(self.valid_indices)} (all timesteps)")
        else:
            print(f"Training samples per epoch: {self.num_demos} (like original ACT - random timestep per episode)")
        
        # Estimate memory usage (sum across all cameras)
        img_mem = 0
        for demo_imgs in self.images_raw:
            for cam_name in self.camera_names:
                img_mem += demo_imgs[cam_name].nbytes
        img_mem /= 1e9
        print(f"\nLoaded {self.num_demos} demos (CPU RAM for images: ~{img_mem:.1f} GB)")
        print(f"Cameras: {self.camera_names}")
        print(f"qpos dim: {self.qpos_dim}, action dim: {self.action_dim}")
    
    def _build_valid_indices_numpy(self):
        """Build list of valid (demo_idx, timestep) pairs for per-timestep sampling."""
        valid = []
        for demo_idx in range(self.num_demos):
            demo_len = len(self.qpos_normalized[demo_idx])
            # Need at least chunk_size actions ahead
            for t in range(demo_len - self.chunk_size):
                valid.append((demo_idx, t))
        return valid
    
    def __len__(self):
        # Original ACT: one sample per episode per epoch (with random timestep)
        if self.sample_per_timestep:
            return len(self.valid_indices)
        else:
            return self.num_demos
    
    def __getitem__(self, idx):
        if self.sample_per_timestep:
            # Per-timestep mode (like our original implementation)
            demo_idx, t = self.valid_indices[idx]
        else:
            # Original ACT mode: sample random timestep from episode idx
            demo_idx = idx
            demo_len = len(self.qpos_normalized[demo_idx])
            # Sample ANY timestep (original ACT does this, padding handles the rest)
            t = np.random.randint(0, demo_len)
        
        demo_len = len(self.qpos_normalized[demo_idx])
        
        # Get pre-normalized qpos at timestep t (move from CPU to GPU)
        qpos = torch.from_numpy(self.qpos_normalized[demo_idx][t]).to(self.device)
        
        # Get images from all cameras at timestep t
        # Stack as (num_cam, C, H, W) - the format ACT expects
        cam_images = []
        for cam_name in self.camera_names:
            img_hwc = self.images_raw[demo_idx][cam_name][t]  # (H, W, C) uint8
            img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32) / 255.0  # (C, H, W)
            cam_images.append(img_chw)
        image = np.stack(cam_images, axis=0)  # (num_cam, C, H, W)
        image = torch.from_numpy(image).to(self.device)
        
        # Get actions from t to end of episode (original ACT does this)
        action_chunk = torch.from_numpy(self.actions_normalized[demo_idx][t:]).to(self.device)
        action_len = len(action_chunk)
        
        # Pad to chunk_size with zeros (like original ACT)
        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool, device=self.device)
        
        if action_len < self.chunk_size:
            pad_size = self.chunk_size - action_len
            padding = torch.zeros(pad_size, action_chunk.shape[1], device=self.device)
            action_chunk = torch.cat([action_chunk, padding], dim=0)
            is_pad[action_len:] = True
        else:
            # Truncate to chunk_size
            action_chunk = action_chunk[:self.chunk_size]
        
        return {
            'qpos': qpos,
            'image': image,
            'actions': action_chunk,
            'is_pad': is_pad,
        }
    
    def get_normalizer_params(self):
        return {
            'qpos_mean': self.qpos_mean,
            'qpos_std': self.qpos_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
        }


def train_epoch(policy, dataloader, epoch):
    """Train for one epoch."""
    policy.model.train()
    
    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Data already on GPU from dataset
        qpos = batch['qpos']
        image = batch['image']
        actions = batch['actions']
        is_pad = batch['is_pad']
        
        # Forward pass through policy (handles loss computation)
        loss_dict = policy(qpos, image, actions, is_pad)
        
        # Backward
        policy.optimizer.zero_grad()
        loss_dict['loss'].backward()
        policy.optimizer.step()
        
        total_loss += loss_dict['loss'].item()
        total_l1 += loss_dict['l1'].item()
        total_kl += loss_dict['kl'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss_dict['loss'].item():.4f}",
            'l1': f"{loss_dict['l1'].item():.4f}",
            'kl': f"{loss_dict['kl'].item():.4f}",
        })
    
    return {
        'loss': total_loss / num_batches,
        'l1': total_l1 / num_batches,
        'kl': total_kl / num_batches,
    }


def evaluate_all_conditions(checkpoint_path, normalizer_path, config_path,
                            num_episodes, num_envs, chunk_size, conditions='all'):
    """
    Evaluate ACT model on all visual conditions in a single subprocess.
    This avoids multiple Isaac Sim startup overhead.
    
    Returns dict mapping condition -> metrics
    """
    import subprocess
    
    output_dir = os.path.dirname(checkpoint_path)
    cmd = [
        '/isaac-sim/python.sh',
        '/workspace/frankapanda/Lift/scripts/evaluate_act.py',
        '--checkpoint', checkpoint_path,
        '--normalizer', normalizer_path,
        '--condition', conditions,  # 'all' or specific condition
        '--num_envs', str(num_envs),
        '--num_episodes', str(num_episodes),
        '--chunk_size', str(chunk_size),
        '--headless',
        '--output', output_dir,
        '--save_trajectories',
    ]
    
    try:
        # Longer timeout for all conditions (4x)
        timeout = 1200 if conditions == 'all' else 600
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            print(f"Eval stderr: {result.stderr[-2000:]}")
            raise RuntimeError(f"Evaluation failed")
        
        # Parse results for each condition from output
        # Format: "Results for <task_name>:" followed by metrics
        stdout = result.stdout + "\n" + result.stderr
        
        all_metrics = {}
        current_condition = None
        
        # Map task names back to condition names
        task_to_condition = {
            'Template-Lift-Cube-Franka-Demo-Baseline-v0': 'baseline',
            'Template-Lift-Cube-Franka-Demo-Lighting-v0': 'lighting',
            'Template-Lift-Cube-Franka-Demo-Texture-v0': 'texture',
            'Template-Lift-Cube-Franka-Demo-Combined-v0': 'combined',
        }
        
        for line in stdout.split('\n'):
            # Detect condition from "Results for <task>:" line
            if 'Results for' in line:
                for task_name, cond in task_to_condition.items():
                    if task_name in line:
                        current_condition = cond
                        all_metrics[current_condition] = {
                            'success_rate': 0.0,
                            'mean_reward': 0.0,
                            'std_reward': 0.0,
                        }
                        break
            
            # Parse metrics for current condition
            if current_condition and current_condition in all_metrics:
                if 'Success Rate:' in line:
                    try:
                        all_metrics[current_condition]['success_rate'] = float(
                            line.split(':')[-1].strip().replace('%', ''))
                    except ValueError:
                        pass
                if 'Mean Reward:' in line and '±' in line:
                    try:
                        parts = line.split(':')[-1].strip()
                        mean_str, std_str = parts.split('±')
                        all_metrics[current_condition]['mean_reward'] = float(mean_str.strip())
                        all_metrics[current_condition]['std_reward'] = float(std_str.strip())
                    except ValueError:
                        pass
        
        return all_metrics
        
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}
    except Exception as e:
        return {'error': str(e)}


def save_results(results, output_path):
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Handle resume logic
    if args.resume:
        # Find latest checkpoint directory in output_dir
        all_ckpts = sorted(glob.glob(os.path.join(args.output_dir, 'act_*')), reverse=True)
        if not all_ckpts:
            print(f"No checkpoint directories found in {args.output_dir}, cannot resume.")
            return
        output_dir = all_ckpts[0]
        print(f"Resuming from latest checkpoint directory: {output_dir}")
        # Find latest checkpoint file
        ckpt_files = sorted(glob.glob(os.path.join(output_dir, 'checkpoint_*.pt')), reverse=True)
        if not ckpt_files:
            print(f"No checkpoint files found in {output_dir}, cannot resume.")
            return
        latest_ckpt = ckpt_files[0]
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        # Load results if available
        results_path = os.path.join(output_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
        else:
            results = None
    else:
        # Fresh run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f'act_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"ACT Training")
        print(f"============")
        print(f"Data: {args.data}")
        print(f"Output: {output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"")
        start_epoch = 1
        results = None

    # Load dataset
    print("Loading dataset...")
    dataset = ACTDataset(
        args.data,
        chunk_size=args.chunk_size,
        max_demos=args.max_demos,
        sample_per_timestep=args.sample_per_timestep,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Data already on GPU, no workers needed
        pin_memory=False,  # Data already on GPU
    )

    # Create policy config
    policy_config = {
        'lr': args.lr,
        'lr_backbone': args.lr_backbone,
        'backbone': args.backbone,
        'enc_layers': args.enc_layers,
        'dec_layers': args.dec_layers,
        'dim_feedforward': args.dim_feedforward,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'nheads': args.nheads,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'camera_names': dataset.camera_names,  # Use cameras detected from dataset
        'state_dim': STATE_DIM,  # 8 dims for Franka (7 joint + 1 gripper)
        # Required args for ACTPolicy
        'ckpt_dir': output_dir,
        'policy_class': 'ACT',
        'task_name': 'lift',
        'seed': args.seed,
        'num_epochs': args.epochs,
    }

    print("\nCreating ACT policy...")
    policy = ACTPolicy(policy_config)

    num_params = sum(p.numel() for p in policy.model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Save config if not resuming
    config = {
        'qpos_dim': dataset.qpos_dim,
        'action_dim': dataset.action_dim,
        **policy_config,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'num_params': num_params,
        'num_demos': dataset.num_demos,
        'num_samples': len(dataset),
    }
    if not args.resume:
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        # Save normalizer
        normalizer = dataset.get_normalizer_params()
        torch.save(normalizer, os.path.join(output_dir, 'normalizer.pt'))

    # Training results
    if results is None:
        results = {
            'config': config,
            'training_losses': [],
            'evaluations': [],
        }

    # Resume model/optimizer if needed
    best_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0
    if args.resume:
        policy.model.load_state_dict(checkpoint['model_state_dict'])
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Try to load best model if available
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            best_ckpt = torch.load(best_model_path)
            best_loss = best_ckpt.get('loss', float('inf'))
            best_epoch = best_ckpt.get('epoch', 0)
            best_model_state = best_ckpt.get('model_state_dict', None)
            best_optimizer_state = best_ckpt.get('optimizer_state_dict', None)
            print(f"Loaded best model from epoch {best_epoch}, loss={best_loss:.4f}")

    print(f"\nStarting training for {args.epochs} epochs (starting at epoch {start_epoch})...")

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_epoch(policy, dataloader, epoch)
        
        results['training_losses'].append({
            'epoch': epoch,
            'loss': metrics['loss'],
            'l1': metrics['l1'],
            'kl': metrics['kl'],
        })
        
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, l1={metrics['l1']:.4f}, kl={metrics['kl']:.4f}")
        
        # Track best model in memory
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in policy.model.state_dict().items()}
            # Deep copy optimizer state (contains nested dicts with tensors)
            import copy
            best_optimizer_state = copy.deepcopy(policy.optimizer.state_dict())
            print(f"  -> New best model (loss={best_loss:.4f})")
        
        # Save checkpoint (including best model) every save_freq epochs
        if epoch % args.save_freq == 0:
            # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.model.state_dict(),
                'optimizer_state_dict': policy.optimizer.state_dict(),
                'loss': metrics['loss'],
            }, os.path.join(output_dir, f'checkpoint_{epoch:04d}.pt'))
            
            # Save best model to disk
            if best_model_state is not None:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': best_optimizer_state,
                    'loss': best_loss,
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f"  -> Saved best model (epoch {best_epoch}, loss={best_loss:.4f})")
        
        # Evaluate
        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            print(f"\nEvaluating at epoch {epoch}...")
            
            # Save current checkpoint for eval
            eval_ckpt = os.path.join(output_dir, 'eval_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.model.state_dict(),
            }, eval_ckpt)
            
            eval_results = {'epoch': epoch}
            
            # Run all conditions in single subprocess (faster - avoids multiple Isaac Sim startups)
            print("  Running all conditions in single process...")
            all_metrics = evaluate_all_conditions(
                checkpoint_path=eval_ckpt,
                normalizer_path=os.path.join(output_dir, 'normalizer.pt'),
                config_path=os.path.join(output_dir, 'config.json'),
                num_episodes=args.eval_episodes,
                num_envs=args.num_eval_envs,
                chunk_size=args.chunk_size,
                conditions='all',
            )
            
            if 'error' in all_metrics:
                print(f"  Error: {all_metrics['error']}")
            else:
                for condition in ['baseline', 'lighting', 'texture', 'combined']:
                    metrics = all_metrics.get(condition, {'error': 'not found'})
                    eval_results[condition] = metrics
                    
                    if 'success_rate' in metrics:
                        reward_str = f", Reward: {metrics['mean_reward']:.4f}±{metrics.get('std_reward', 0):.4f}"
                        print(f"  {condition}: Success: {metrics['success_rate']:.1f}%{reward_str}")
                    else:
                        print(f"  {condition}: Error: {metrics.get('error', 'unknown')}")
            
            results['evaluations'].append(eval_results)
            save_results(results, os.path.join(output_dir, 'results.json'))
            print()
    
    # Save final model (use last training loss from results)
    last_training_loss = results['training_losses'][-1]['loss'] if results['training_losses'] else 0.0
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': policy.model.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'loss': last_training_loss,
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # Save best model (final save from memory)
    if best_model_state is not None:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'loss': best_loss,
        }, os.path.join(output_dir, 'best_model.pt'))
    
    # Save final results
    save_results(results, os.path.join(output_dir, 'results.json'))
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
