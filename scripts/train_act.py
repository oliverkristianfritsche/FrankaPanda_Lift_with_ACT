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

# Add ACT (original repo) to path. Repo root = parent of this script's dir;
# `act/` lives at <repo>/act (see README clone step). Portable across local,
# Launchable, and Modal — no hardcoded absolute paths.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "act"))
sys.path.insert(0, os.path.join(REPO_ROOT, "act", "detr"))
from policy import ACTPolicy


# ACT state/action dimensions for Franka
# qpos: 7 joint positions + 1 gripper = 8 dims
# actions: 8 dims (same as qpos)
STATE_DIM = 11  # joint_pos (7) + gripper (1) + goal (3)
QPOS_INDICES = list(range(0, 7))  # Joint positions from observation
GRIPPER_INDEX = 7  # Gripper from action (we'll use last action's gripper for qpos)
GOAL_INDICES = list(range(21, 24))  # Goal/target position (obs[21:24]) — the LIFT TARGET.
# Critical: the goal is an invisible randomized command (not rendered in the cameras), so
# without feeding it to ACT the policy is blind to where the cube should go — it can only
# lift to an average spot and never places at the goal (the cause of the persistent 0%).


def parse_args():
    parser = argparse.ArgumentParser(description="ACT Training")
    
    # Data
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to demo HDF5 file or glob pattern")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Separate HDF5 of held-out demos for validation-loss tracking. "
                             "Works WITH --sample_per_timestep (unlike --val_ratio, which splits the "
                             "training file and is ignored under per-timestep sampling). The val set "
                             "borrows the TRAIN normalizer, so no validation stats leak.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "logs", "act", "checkpoints"),
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
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping: stop if the selection loss (val loss when "
                             "--val_data/--val_ratio is set, else train loss) has not improved "
                             "for N epochs. 0 = disabled (train all --epochs).")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 avoids shared memory issues; used only with --lazy)")
    parser.add_argument("--lazy", action="store_true",
                        help="Read images from HDF5 on demand instead of pre-loading to RAM "
                             "(lets the full demo set fit a small-RAM box)")
    parser.add_argument("--augment", action="store_true",
                        help="Image augmentation (per-camera color jitter + mild random crop) "
                             "to reduce overfitting and improve visual generalization")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Hold out this fraction of demos as a validation set (per-episode mode). "
                             "Tracks val loss per epoch (cheap overfit signal) and selects the best "
                             "checkpoint by val loss, ACT-style. 0 = no validation split.")
    parser.add_argument("--num_eval_envs", type=int, default=16)
    parser.add_argument("--sample_per_timestep", action="store_true",
                        help="Sample every timestep instead of random timestep per episode (slower)")
    parser.add_argument("--trim_after_goal", type=int, default=-1,
                        help="Cut each demo N steps after the cube first reaches the goal "
                             "(<5cm). The oracle finishes by ~step 45 of 250, so ~82%% of every "
                             "demo is a static goal-hover that dominates the loss; trimming "
                             "re-weights training toward the approach/grasp phase where the "
                             "policy actually fails. -1 = off (keep full episodes).")
    
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from latest checkpoint in output_dir")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from an explicit checkpoint .pt (continues in that "
                             "checkpoint's directory; overrides the --resume newest-dir search "
                             "so an unrelated newer run can't shadow the one you mean to continue)")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Log training to Weights & Biases (needs WANDB_API_KEY in env)")
    parser.add_argument("--wandb_project", type=str, default="frankapanda-act",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team or username). Default None = the API key's default.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: the act_<timestamp> output dir name)")
    parser.add_argument("--wandb_log_every", type=int, default=0,
                        help="Also log batch loss to W&B every N iterations (0 = per-epoch "
                             "only). Restores faithful-recipe-style granular curves even "
                             "with large per-timestep epochs.")
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
        lazy: bool = False,  # read images from HDF5 on demand (fits the full set in RAM)
        augment: bool = False,  # image augmentation (color jitter + mild crop) for generalization
        val_ratio: float = 0.0,  # hold out last val_ratio of demos; split decided HERE so
                                 # normalization can use train demos only (no val leakage)
        normalizer: Optional[dict] = None,  # external TRAIN stats for a val set (no leakage)
        trim_after_goal: int = -1,  # cut demos N steps after first at-goal (<0 = off)
    ):
        self.chunk_size = chunk_size
        self.trim_after_goal = trim_after_goal
        self.sample_per_timestep = sample_per_timestep
        self.lazy = lazy
        self.augment = augment
        self.val_ratio = val_ratio
        self._ext_normalizer = normalizer  # if set, reuse these TRAIN stats instead of computing
        self.val_demos = set()  # filled below once num_demos is known
        if augment:
            import torchvision.transforms as _T
            self._cj = _T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self._h5cache = {}  # per-worker open HDF5 handles (lazy mode)

        self.qpos_data = []
        self.image_data = []  # Will be list of dicts: {cam_name: images} (eager mode only)
        self.action_data = []
        self.demo_refs = []   # (file_path, demo_key) per demo (lazy mode)
        
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

                    # Optionally drop the long static goal-hover tail: keep up to N steps
                    # past the first at-goal step so training mass concentrates on the
                    # approach/grasp/transport instead of ~200 steps of standing still.
                    if self.trim_after_goal >= 0:
                        dist = np.linalg.norm(full_obs[:, 18:21] - full_obs[:, 21:24], axis=1)
                        at_goal = np.where(dist < 0.05)[0]
                        if len(at_goal) > 0:
                            t_end = min(len(full_obs), int(at_goal[0]) + self.trim_after_goal)
                            full_obs = full_obs[:t_end]
                            actions_raw = actions_raw[:t_end]

                    # qpos = joint_pos (7) + gripper (1) = 8 dims
                    # Use joint positions from obs, gripper from action
                    joint_pos = full_obs[:, QPOS_INDICES].astype(np.float32)  # 7 dims
                    gripper = actions_raw[:, -1:].astype(np.float32)  # 1 dim from action
                    goal = full_obs[:, GOAL_INDICES].astype(np.float32)  # 3 dims — the lift target
                    qpos = np.concatenate([joint_pos, gripper, goal], axis=1)  # 11 dims (joints+gripper+goal)
                    
                    # Map each camera to its HDF5 dataset key (shared by both modes).
                    cam_keys = {}
                    for cam_name in self.camera_names:
                        if cam_name == 'camera' and 'images' in demo:
                            cam_keys[cam_name] = 'images'
                        elif f'images_{cam_name}' in demo:
                            cam_keys[cam_name] = f'images_{cam_name}'
                        else:
                            raise ValueError(f"No images found for camera '{cam_name}' in {fpath}/demo_{i}")

                    # Actions are already 8 dims
                    actions = actions_raw  # 8 dims

                    self.qpos_data.append(qpos)
                    self.action_data.append(actions)
                    if self.lazy:
                        # Defer image loading: remember where to read this demo's frames.
                        # (Trimming needs no special handling here: __getitem__ samples t
                        # from the trimmed qpos length, so frames past t_end are never read.)
                        self.demo_refs.append((fpath, f'demo_{i}', cam_keys))
                    else:
                        t_end = len(qpos)
                        self.image_data.append({c: demo[k][:t_end] for c, k in cam_keys.items()})
                    
                    total_demos += 1
                    total_timesteps += len(qpos)
                    pbar.set_postfix({'demos': total_demos, 'timesteps': total_timesteps})
        
        self.num_demos = total_demos

        # Decide the val split HERE (per-episode mode) so normalization uses TRAIN demos
        # ONLY -> zero validation leakage into the qpos/action mean+std.
        if self.val_ratio and self.val_ratio > 0 and not self.sample_per_timestep:
            n_val = max(1, int(round(self.num_demos * self.val_ratio)))
            self.val_demos = set(range(self.num_demos - n_val, self.num_demos))
        train_ids = [i for i in range(self.num_demos) if i not in self.val_demos]

        # Compute normalization stats on TRAIN demos only.
        all_qpos = np.concatenate([self.qpos_data[i] for i in train_ids], axis=0)
        all_actions = np.concatenate([self.action_data[i] for i in train_ids], axis=0)

        self.qpos_dim = all_qpos.shape[1]
        self.action_dim = all_actions.shape[1]
        
        self.qpos_mean = all_qpos.mean(axis=0)
        self.qpos_std = np.clip(all_qpos.std(axis=0), 1e-2, np.inf)  # Original ACT clips to 1e-2
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = np.clip(all_actions.std(axis=0), 1e-2, np.inf)  # Original ACT clips to 1e-2

        # A separate val set borrows the TRAIN normalizer: train/val share one scale and
        # zero validation stats leak into normalization.
        if self._ext_normalizer is not None:
            self.qpos_mean = self._ext_normalizer['qpos_mean']
            self.qpos_std = self._ext_normalizer['qpos_std']
            self.action_mean = self._ext_normalizer['action_mean']
            self.action_std = self._ext_normalizer['action_std']
        
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
        if not self.lazy:
            self.images_raw = self.image_data  # Just rename, no copy
            del self.image_data
        # Free the raw (unnormalized) qpos/action lists; keep qpos_data for valid-index reuse
        del self.action_data

        if sample_per_timestep:
            print(f"Training samples per epoch: {len(self.valid_indices)} (all timesteps)")
        else:
            print(f"Training samples per epoch: {self.num_demos} (like original ACT - random timestep per episode)")

        # Estimate memory usage (sum across all cameras)
        img_mem = 0
        if self.lazy:
            print(f"\nLoaded {self.num_demos} demos (LAZY: images read from HDF5 on demand, ~0 GB RAM)")
        else:
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
            # Called before normalization, so use the raw (still-present) qpos_data.
            demo_len = len(self.qpos_data[demo_idx])
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
    
    def _h5(self, path):
        # Open HDF5 handles lazily and cache per process. DataLoader workers are
        # forked AFTER __init__, so each worker builds its own cache here (h5py
        # handles can't be shared across processes).
        h = self._h5cache.get(path)
        if h is None:
            h = h5py.File(path, 'r')
            self._h5cache[path] = h
        return h

    def _augment_images(self, image):
        """Mild image augmentation for generalization (training only; eval uses raw
        frames): per-camera random resized crop (area scale 0.9-1.0) + color jitter.
        Independent per camera so each view sees its own jitter/shift."""
        import torchvision.transforms.functional as TF
        out = []
        for k in range(image.shape[0]):
            img = image[k]  # (C, H, W) float in [0,1]
            _, H, W = img.shape
            s = float(np.random.uniform(0.9, 1.0)) ** 0.5
            nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
            top = int(np.random.randint(0, H - nh + 1))
            left = int(np.random.randint(0, W - nw + 1))
            img = TF.resized_crop(img, top, left, nh, nw, [H, W], antialias=True)
            img = self._cj(img)
            out.append(img)
        return torch.stack(out, dim=0)

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
        # Lazy mode returns CPU tensors so DataLoader workers are allowed (CUDA
        # tensors can't cross the fork); train_epoch moves the batch to GPU. Eager
        # mode keeps the original "tensors already on GPU" behavior.
        dev = 'cpu' if self.lazy else self.device
        qpos = torch.from_numpy(self.qpos_normalized[demo_idx][t]).to(dev)

        # Get images from all cameras at timestep t
        # Stack as (num_cam, C, H, W) - the format ACT expects
        if self.lazy:
            fpath, demo_key, cam_keys = self.demo_refs[demo_idx]
            grp = self._h5(fpath)[demo_key]
            frame = {c: grp[cam_keys[c]][t] for c in self.camera_names}  # one HDF5 read per cam
        else:
            frame = {c: self.images_raw[demo_idx][c][t] for c in self.camera_names}
        cam_images = []
        for cam_name in self.camera_names:
            img_hwc = frame[cam_name]  # (H, W, C) uint8
            img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32) / 255.0  # (C, H, W)
            cam_images.append(img_chw)
        image = np.stack(cam_images, axis=0)  # (num_cam, C, H, W)
        image = torch.from_numpy(image).to(dev)
        if self.augment and demo_idx not in self.val_demos:
            image = self._augment_images(image)

        # Get actions from t to end of episode (original ACT does this)
        action_chunk = torch.from_numpy(self.actions_normalized[demo_idx][t:]).to(dev)
        action_len = len(action_chunk)

        # Pad to chunk_size with zeros (like original ACT)
        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool, device=dev)

        if action_len < self.chunk_size:
            pad_size = self.chunk_size - action_len
            padding = torch.zeros(pad_size, action_chunk.shape[1], device=dev)
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


def train_epoch(policy, dataloader, epoch, wandb_log_every=0):
    """Train for one epoch. wandb_log_every>0 also streams batch loss to W&B
    every N iterations (granular curves for large per-timestep epochs)."""
    policy.model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    num_batches = 0

    dev = next(policy.model.parameters()).device
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Eager mode: already on GPU (.to is a no-op). Lazy mode: move CPU->GPU here.
        qpos = batch['qpos'].to(dev, non_blocking=True)
        image = batch['image'].to(dev, non_blocking=True)
        actions = batch['actions'].to(dev, non_blocking=True)
        is_pad = batch['is_pad'].to(dev, non_blocking=True)

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

        if wandb_log_every > 0 and num_batches % wandb_log_every == 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"batch_loss": loss_dict['loss'].item(),
                               "batch_l1": loss_dict['l1'].item(),
                               "batch_kl": loss_dict['kl'].item()})
            except Exception:  # noqa: BLE001 — a W&B blip must not kill training
                pass

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


def validate(policy, dataloader):
    """Validation loss (l1 + kl) on the held-out demos — eval mode, no grad, no
    augmentation. Cheap per-epoch overfitting signal; also used for ACT-style
    best-checkpoint selection."""
    policy.model.eval()
    total_loss = total_l1 = total_kl = 0.0
    n = 0
    dev = next(policy.model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            qpos = batch['qpos'].to(dev, non_blocking=True)
            image = batch['image'].to(dev, non_blocking=True)
            actions = batch['actions'].to(dev, non_blocking=True)
            is_pad = batch['is_pad'].to(dev, non_blocking=True)
            loss_dict = policy(qpos, image, actions, is_pad)
            total_loss += loss_dict['loss'].item()
            total_l1 += loss_dict['l1'].item()
            total_kl += loss_dict['kl'].item()
            n += 1
    if n == 0:
        return None
    return {'loss': total_loss / n, 'l1': total_l1 / n, 'kl': total_kl / n}


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
        sys.executable,
        os.path.join(REPO_ROOT, 'scripts', 'evaluate_act.py'),
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
    resuming = args.resume or bool(args.resume_from)
    if resuming:
        if args.resume_from:
            # Explicit checkpoint: continue in ITS directory (overrides the newest-dir
            # search so an unrelated newer run can't shadow the one we mean to continue).
            latest_ckpt = args.resume_from
            output_dir = os.path.dirname(os.path.abspath(latest_ckpt))
            print(f"Resuming from explicit checkpoint: {latest_ckpt}")
        else:
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
        lazy=args.lazy,
        augment=args.augment,
        val_ratio=args.val_ratio,
        trim_after_goal=args.trim_after_goal,
    )

    # Lazy mode keeps tensors on CPU, so DataLoader workers can parallelize the
    # HDF5 reads (the bottleneck on this 2-vCPU box). Eager mode puts tensors on
    # the GPU in __getitem__, which forbids workers -> force num_workers=0.
    nw = args.num_workers if args.lazy else 0
    # Optional train/val split (per-episode mode): hold out the last val_ratio of demos
    # as validation (no augmentation on them). Lets us track val loss per epoch and pick
    # the best checkpoint by val loss, ACT-style. Normalization stays over all demos
    # (mean/std leakage is negligible and keeps train/val on the same scale).
    val_loader = None
    if args.val_data:
        # Separate held-out val FILE. Train on ALL of --data (so this works with
        # --sample_per_timestep, unlike --val_ratio); validate on --val_data using the
        # TRAIN normalizer (no leakage). Val uses per-episode sampling (one sample/demo)
        # so validating each epoch stays cheap.
        print(f"Loading validation set from {args.val_data} ...")
        val_dataset = ACTDataset(
            args.val_data,
            chunk_size=args.chunk_size,
            camera_names=dataset.camera_names,
            sample_per_timestep=False,
            lazy=args.lazy,
            augment=False,
            normalizer=dataset.get_normalizer_params(),
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=nw, pin_memory=(nw > 0),
                                persistent_workers=(nw > 0))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=nw, pin_memory=(nw > 0),
                                persistent_workers=(nw > 0))
        print(f"Separate val file: {dataset.num_demos} train / {val_dataset.num_demos} val demos")
    elif dataset.val_demos:  # dataset already split + used TRAIN-ONLY normalization
        from torch.utils.data import Subset
        val_idx = sorted(dataset.val_demos)
        train_idx = [i for i in range(dataset.num_demos) if i not in dataset.val_demos]
        dataloader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                                shuffle=True, num_workers=nw, pin_memory=(nw > 0),
                                persistent_workers=(nw > 0))
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                                shuffle=False, num_workers=nw, pin_memory=(nw > 0),
                                persistent_workers=(nw > 0))
        print(f"Train/val split: {len(train_idx)} train / {len(val_idx)} val demos")
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=(nw > 0),
            persistent_workers=(nw > 0),
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
        'state_dim': dataset.qpos_dim,      # 11 = joints(7) + gripper(1) + goal(3)
        'action_dim': dataset.action_dim,   # 8 = joints(7) + gripper(1); differs from qpos (goal-conditioned)
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
    if not resuming:
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        # Save normalizer
        normalizer = dataset.get_normalizer_params()
        torch.save(normalizer, os.path.join(output_dir, 'normalizer.pt'))

    # Optional Weights & Biases logging (non-fatal if it fails)
    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        _wb = dict(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=args.wandb_run_name or os.path.basename(output_dir),
                   config=config, resume="allow")
        try:
            wandb.init(**_wb)
            print(f"W&B logging to '{args.wandb_project}' as '{wandb.run.name}' "
                  f"(mode={os.environ.get('WANDB_MODE', 'online')})", flush=True)
        except Exception as e:  # noqa: BLE001 — a bad/expired key must NOT kill a multi-hour run
            print(f"W&B online init failed ({e}); falling back to OFFLINE (sync later).", flush=True)
            try:
                os.environ['WANDB_MODE'] = 'offline'
                wandb.init(**_wb)
                print("W&B logging OFFLINE.", flush=True)
            except Exception as e2:  # noqa: BLE001
                print(f"W&B offline init also failed ({e2}); continuing WITHOUT W&B "
                      "(results.json still written).", flush=True)
                use_wandb = False

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
    epochs_since_best = 0  # for --patience early stopping
    if resuming:
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
        metrics = train_epoch(policy, dataloader, epoch,
                              wandb_log_every=(args.wandb_log_every if use_wandb else 0))
        val_metrics = validate(policy, val_loader) if val_loader is not None else None

        entry = {'epoch': epoch, 'loss': metrics['loss'], 'l1': metrics['l1'], 'kl': metrics['kl']}
        if val_metrics is not None:
            entry.update({'val_loss': val_metrics['loss'], 'val_l1': val_metrics['l1'],
                          'val_kl': val_metrics['kl']})
        results['training_losses'].append(entry)

        vstr = f", val_l1={val_metrics['l1']:.4f}" if val_metrics is not None else ""
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, l1={metrics['l1']:.4f}, "
              f"kl={metrics['kl']:.4f}{vstr}")

        # Persist the loss history every epoch (eval_interval=0 otherwise only writes
        # results.json at the very end, so an interruption would lose the curve).
        save_results(results, os.path.join(output_dir, 'results.json'))

        if use_wandb:
            try:
                wlog = {"epoch": epoch, "loss": metrics["loss"],
                        "l1": metrics["l1"], "kl": metrics["kl"]}
                if val_metrics is not None:
                    wlog.update({"val_loss": val_metrics["loss"], "val_l1": val_metrics["l1"],
                                 "val_kl": val_metrics["kl"]})
                wandb.log(wlog)
            except Exception as e:  # noqa: BLE001 — a W&B blip must not kill a multi-hour run
                print(f"  (wandb.log failed, continuing: {e})", flush=True)

        # Track best model by VALIDATION loss (ACT-style) when available, else train loss.
        sel_loss = val_metrics['loss'] if val_metrics is not None else metrics['loss']
        if sel_loss < best_loss:
            best_loss = sel_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in policy.model.state_dict().items()}
            # Deep copy optimizer state (contains nested dicts with tensors)
            import copy
            best_optimizer_state = copy.deepcopy(policy.optimizer.state_dict())
            tag = "val_loss" if val_metrics is not None else "loss"
            print(f"  -> New best model ({tag}={best_loss:.4f})")
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        
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
        
        # Early stopping: stop when the selection loss hasn't improved for --patience epochs.
        if args.patience > 0 and epochs_since_best >= args.patience:
            print(f"Early stopping at epoch {epoch}: no improvement for {epochs_since_best} "
                  f"epochs (patience={args.patience}); best epoch {best_epoch} "
                  f"(loss={best_loss:.4f}).", flush=True)
            torch.save({'epoch': epoch,
                        'model_state_dict': policy.model.state_dict(),
                        'optimizer_state_dict': policy.optimizer.state_dict(),
                        'loss': metrics['loss']},
                       os.path.join(output_dir, f'checkpoint_{epoch:04d}.pt'))
            break

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
    
    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
