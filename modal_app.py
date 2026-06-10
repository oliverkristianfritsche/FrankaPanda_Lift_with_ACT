"""Modal CLI runners for the FrankaPanda lift + ACT pipeline (no dashboard needed).

Architecture (matches the repo's split):
  * demos / eval  -> need Isaac Lab (sim + camera rendering) -> official isaac-lab image
  * train (ACT)   -> needs only torch + the act/ repo        -> plain debian image

Everything hands off through one Modal Volume ("frankapanda-data") at /data:
  /data/demos/*.hdf5                        (from `demos`)
  /data/logs/act/checkpoints/act_*/...      (from `train`: best_model.pt, normalizer.pt, config.json)

------------------------------------------------------------------------------
ONE-TIME SETUP (free NGC account -> API key, then store as a Modal secret):
  # https://org.ngc.nvidia.com/account/api-keys  (username is literally $oauthtoken)
  modal secret create ngc-registry \
      REGISTRY_USERNAME='$oauthtoken' REGISTRY_PASSWORD='<YOUR_NGC_API_KEY>'

USAGE (run from this worktree root so the local code is copied in):
  modal run modal_app.py::gpu_check                      # plumbing check (needs ngc-registry secret to exist)
  modal run modal_app.py::demos --target-minutes 2       # Isaac: smoke (tiny dataset)
  modal run modal_app.py::demos --target-minutes 30      # Isaac: real >=30 min dataset
  modal run modal_app.py::train --epochs 10              # torch: smoke
  modal run modal_app.py::train --epochs 500             # torch: real
  modal run modal_app.py::eval                           # Isaac: eval latest ACT ckpt
  modal volume get frankapanda-data logs/act ./pulled    # download results locally
------------------------------------------------------------------------------
"""
import modal

# Path the repo is copied to inside the containers (avoid clashing with /workspace/isaaclab).
REPO_REMOTE = "/root/FrankaPanda_Lift_with_ACT"
PPO_CKPT = f"{REPO_REMOTE}/logs/skrl/franka_lift/2025-12-11_16-32-17_ppo_torch/checkpoints/best_agent.pt"
VOL = "/data"

# GPUs: A10G is Ampere with RT cores (good for headless camera rendering) and avoids
# the isaac-lab:2.1.0 RTX-50-series bug. ACT training is light; A10G is plenty.
ISAAC_GPU = "A10G"
TRAIN_GPU = "A10G"

app = modal.App("frankapanda-lift")
vol = modal.Volume.from_name("frankapanda-data", create_if_missing=True)

# Repo is tiny (~few MB incl. act/ and the PPO checkpoint) -> copy it into both images.
_IGNORE = ["**/.git", "**/__pycache__", "**/*.pyc", "**/.claude", "data", "media", "**/*.gif"]


def _with_repo(img):
    return img.add_local_dir(".", REPO_REMOTE, copy=True, ignore=_IGNORE)


# Isaac Lab image (Isaac Sim bundled). Pulling from nvcr.io needs NGC creds (see header).
isaac_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-lab:2.1.0",
        add_python="3.11",  # standalone python for Modal's harness; scripts use /isaac-sim/python.sh
        # NOTE: Modal hydrates every referenced secret at app load, so `ngc-registry` must
        # exist before ANY `modal run` of this app (even gpu_check). Create it once (see
        # header). Fine in practice — NGC is required for the Isaac steps regardless.
        secret=modal.Secret.from_name("ngc-registry"),
    )
    .env({
        "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ALLOW_ROOT": "1",
        # Expose the GPU's graphics/display capabilities (not just compute) so the Isaac
        # RTX/Vulkan renderer can enumerate the device for headless camera rendering.
        "NVIDIA_DRIVER_CAPABILITIES": "all", "NVIDIA_VISIBLE_DEVICES": "all",
    })
    .apt_install("vulkan-tools")  # provides vulkaninfo for the vulkan_check diagnostic
    # Copy the repo in BEFORE the editable install (pip -e needs source/Lift present at
    # build time). The `ls` makes the build log show the path if it ever breaks again.
    # If isaaclab sits behind isaaclab.sh rather than /isaac-sim/python.sh, switch runner.
    .add_local_dir(".", REPO_REMOTE, copy=True, ignore=_IGNORE)
    .run_commands(f"ls -la {REPO_REMOTE}/source && /isaac-sim/python.sh -m pip install -e {REPO_REMOTE}/source/Lift")
)

# Torch image for ACT training: no Isaac, no NGC.
train_image = _with_repo(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "numpy", "h5py", "opencv-python-headless",
        "einops", "ipython", "pyquaternion", "tqdm", "matplotlib", "packaging", "wandb",
    )
)


@app.function(image=train_image, gpu="T4", timeout=900)
def gpu_check():
    """Cheapest possible smoke test: confirm Modal scheduling + GPU + repo copy work."""
    import os, subprocess, torch
    print("=== gpu_check ===")
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    print("repo present:", os.path.exists(f"{REPO_REMOTE}/scripts/train_act.py"))
    print("act/ present:", os.path.exists(f"{REPO_REMOTE}/act/policy.py"))
    print("PPO ckpt present:", os.path.exists(PPO_CKPT))
    subprocess.run(["nvidia-smi"], check=False)


@app.function(image=isaac_image, gpu=ISAAC_GPU, timeout=600)
def vulkan_check():
    """Decisive test: can the Isaac image do GPU graphics (Vulkan) on Modal?

    If vulkaninfo lists the GPU as a device, headless RTX/camera rendering should
    work and the demos/eval failure was config. If it reports 0 GPUs, Modal isn't
    exposing graphics for this GPU and Isaac demos/eval cannot render here (ACT
    training is unaffected — that's pure CUDA compute).
    """
    import subprocess
    subprocess.run(["nvidia-smi"], check=False)
    print("=== Vulkan ICD files ===", flush=True)
    subprocess.run("ls -la /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/ 2>&1", shell=True, check=False)
    print("=== vulkaninfo --summary ===", flush=True)
    subprocess.run("vulkaninfo --summary 2>&1 | head -60", shell=True, check=False)


@app.function(image=isaac_image, gpu=ISAAC_GPU, volumes={VOL: vol}, timeout=4 * 60 * 60)
def demos(target_minutes: float = 2.0, num_envs: int = 32, condition: str = "baseline"):
    """Collect oracle demos with Isaac Lab -> /data/demos (deterministic mean policy)."""
    import subprocess
    cmd = [
        "/isaac-sim/python.sh", f"{REPO_REMOTE}/scripts/generate_demos.py",
        "--checkpoint", PPO_CKPT,
        "--condition", condition,
        "--num_envs", str(num_envs),
        "--target_minutes", str(target_minutes),
        "--output_dir", f"{VOL}/demos",
        "--headless", "--enable_cameras",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_REMOTE)
    vol.commit()


@app.function(image=train_image, gpu=TRAIN_GPU, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("wandb-secret")], timeout=8 * 60 * 60)
def train(epochs: int = 10, chunk_size: int = 100, batch_size: int = 8,
          wandb: bool = True, wandb_project: str = "frankapanda-act"):
    """Train ACT from /data/demos -> /data/logs/act (no Isaac; eval_interval=0).

    Logs to Weights & Biases by default (WANDB_API_KEY comes from the wandb-secret).
    Pass --no-wandb to disable.
    """
    import subprocess
    cmd = [
        "python", f"{REPO_REMOTE}/scripts/train_act.py",
        "--data", f"{VOL}/demos/*.hdf5",
        "--output_dir", f"{VOL}/logs/act/checkpoints",
        "--epochs", str(epochs),
        "--chunk_size", str(chunk_size),
        "--batch_size", str(batch_size),
        "--eval_interval", "0",  # no in-loop Isaac eval on this non-Isaac image
    ]
    if wandb:
        cmd += ["--wandb", "--wandb_project", wandb_project]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_REMOTE)
    vol.commit()


@app.function(image=isaac_image, gpu=ISAAC_GPU, volumes={VOL: vol}, timeout=2 * 60 * 60)
def eval(checkpoint: str = None, num_episodes: int = 50, num_envs: int = 16,
         chunk_size: int = 100, condition: str = "baseline"):
    """Evaluate an ACT checkpoint in Isaac Lab (defaults to the latest best_model.pt)."""
    import glob, subprocess
    if checkpoint is None:
        cks = sorted(glob.glob(f"{VOL}/logs/act/checkpoints/act_*/best_model.pt"))
        assert cks, "no ACT checkpoint found under /data/logs/act/checkpoints/act_*/best_model.pt"
        checkpoint = cks[-1]
    print("Using checkpoint:", checkpoint, flush=True)
    cmd = [
        "/isaac-sim/python.sh", f"{REPO_REMOTE}/scripts/evaluate_act.py",
        "--checkpoint", checkpoint,
        "--condition", condition,
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--chunk_size", str(chunk_size),
        "--headless",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_REMOTE)
    vol.commit()
