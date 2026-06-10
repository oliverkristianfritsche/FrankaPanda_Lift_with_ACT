# Runbook — two-camera ACT pipeline on an Isaac Launchable

Isaac Sim's Vulkan renderer (needed for camera capture) does not initialize on Modal's
compute GPUs, so the Isaac steps (**demos, eval, optional PPO retrain**) run on an
**NVIDIA Isaac Launchable** (Brev) which has Isaac Sim + working rendering preinstalled.
ACT **training** is pure CUDA compute and can run on the Launchable too (commands below)
or on any GPU box (e.g. Modal `modal run modal_app.py::train`).

All paths in the scripts are portable (`REPO_ROOT` derived from `__file__`), so no edits
are needed on the Launchable.

## Provision the GPU over the Brev CLI (no dashboard)
Brev (NVIDIA) manages provisioning + SSH from the terminal.
```bash
# Install + login (one-time, on your local machine)
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
brev login                       # browser OAuth; or `brev login --token <TOKEN>` for headless

# Start the CHEAPEST RTX GPU that renders (L4 is enough for 320x240 x 2 cams;
# step up to A10G / L40S only if you hit VRAM or throughput limits)
brev start isaac-act             # pick the GPU/instance type; see `brev start --help`
brev shell isaac-act             # SSH in (CLI-managed keys); or `ssh isaac-act`
brev refresh                     # sync instances created via the console into the CLI

# Stop billing when done
brev stop isaac-act              # or: brev delete isaac-act
```
Two ways to get Isaac onto the box:
- **Isaac Launchable template** — Isaac Sim + rendering preinstalled at `/isaac-sim`; deploy
  it, then `brev shell` and run §0–3 directly.
- **Plain GPU VM (fully CLI)** — `brev start` above, then run the official container:
  ```bash
  echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
  docker run --gpus all -it --network host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
    nvcr.io/nvidia/isaac-lab:2.1.0 bash
  ```
  Then run §0–3 inside the container (`/isaac-sim/python.sh` lives there). Unlike Modal, a
  real GPU VM exposes graphics/Vulkan, so the renderer works.

## 0. One-time setup on the Launchable
```bash
cd /workspace
# Clone THIS branch (act/ is committed in the repo — no separate act clone needed)
git clone -b two-camera-front-wrist \
  https://github.com/oliverkristianfritsche/FrankaPanda_Lift_with_ACT.git
cd FrankaPanda_Lift_with_ACT

# Install the task package into Isaac Sim's python
/isaac-sim/python.sh -m pip install -e source/Lift
# Extra deps used by ACT (the act/ repo) + logging
/isaac-sim/python.sh -m pip install einops ipython pyquaternion wandb opencv-python-headless

# Sanity check the env is importable
/isaac-sim/python.sh scripts/list_envs.py   # should list Template-Lift-Cube-Franka-*
```

## 1. Collect demos (2 cameras: `camera` scene + `wrist`)
PPO checkpoint is committed at the path below. Cameras + resolution (320x240 RGB) and the
deterministic-mean (smooth) action recording are already wired in.

```bash
CKPT=logs/skrl/franka_lift/2025-12-11_16-32-17_ppo_torch/checkpoints/best_agent.pt

# SMOKE first — a couple demos, then verify both camera views are present/correct
/isaac-sim/python.sh scripts/generate_demos.py \
  --checkpoint $CKPT --condition baseline --num_demos 2 --num_envs 4 --headless
/isaac-sim/python.sh scripts/view_demos.py --demo_file "data/demos/demos_baseline_*.hdf5" --list
#   -> confirm each demo has images_camera AND images_wrist with shape (T,240,320,3)
#   Visual framing of the WRIST cam: view_demos' live window needs a display. On a headless
#   box, scp the .hdf5 to a local machine and run view_demos there, or eyeball saved frames.
#   If the wrist view is poorly framed, tweak make_wrist_camera() pos/rot in
#   source/Lift/Lift/tasks/manager_based/lift/config/franka/visual_randomization.py and re-run.

# FULL dataset — >= 30 minutes of oracle demos (target stops on collected time)
/isaac-sim/python.sh scripts/generate_demos.py \
  --checkpoint $CKPT --condition baseline --target_minutes 30 --num_envs 32 --headless
#   ~500-600 episodes @ 50 Hz; bump --target_minutes for more. HDF5 lands in data/demos/.
```

## 2. Train ACT (Weights & Biases on)
```bash
export WANDB_API_KEY=<your key>          # or run `wandb login`; required — --wandb is fail-fast

# SMOKE first: 2 epochs to confirm the data pipeline + W&B logging before the long run
/isaac-sim/python.sh scripts/train_act.py \
  --data "data/demos/*.hdf5" --epochs 2 --batch_size 8 --eval_interval 0 --wandb
#   -> a run must appear at wandb.ai/<you>/frankapanda-act with loss/l1/kl curves

# FULL run
/isaac-sim/python.sh scripts/train_act.py \
  --data "data/demos/*.hdf5" \
  --epochs 500 --chunk_size 100 --batch_size 8 --eval_interval 0 --wandb
#   camera_names are auto-detected from the HDF5 (["camera","wrist"]); ACT auto-sizes.
#   Outputs: logs/act/checkpoints/act_<ts>/{best_model.pt, normalizer.pt, config.json}
```
(Alternative: run on Modal instead — `modal run modal_app.py::train --epochs 500` — pulls
demos from a Modal Volume; only useful if you also pushed the HDF5 there.)

## 3. Evaluate ACT in Isaac
```bash
/isaac-sim/python.sh scripts/evaluate_act.py \
  --checkpoint logs/act/checkpoints/act_*/best_model.pt \
  --condition baseline --num_episodes 50 --headless
#   success_rate = object within 5 cm of goal. eval_results_*.json written next to the ckpt.
```

## 4. (Optional) Retrain PPO for a smoother oracle
The smoothness curriculum is already bumped (action_rate -0.1->-0.5, joint_vel -0.1->-0.25).
Only do this if the mean-policy demos are still jittery.
```bash
/isaac-sim/python.sh scripts/skrl/train.py --task Template-Lift-Cube-Franka-v0 --headless
#   then point --checkpoint at the new best_agent.pt in step 1.
```

## Notes
- Jitter strategy is end-to-end / no inference filters: smooth oracle (mean-action demos,
  stronger jerk penalty) + (if needed) execute fewer steps per ACT chunk. See the audit.
- `--num_envs` is limited by GPU VRAM; reduce if you hit OOM during rendering.

## Cost optimization (Brev is pay-per-uptime)
Whole pipeline is ~a few dollars if you follow these:
- **Already in-repo:** 2 cameras (not 3) + 320x240 (not 640x480) = ~6x less render/data;
  `--target_minutes 30` stops exactly at target; `baseline` only (not all 4 = 4x); headless.
- **STOP the instance the moment you're done** — `brev stop isaac-act` (or `brev delete`).
  An idle GPU is by far the biggest money-waster — more than any compute choice.
- **Use the Isaac Launchable template** (Isaac preinstalled) instead of a plain VM that
  pulls the ~20 GB isaac-lab image — saves paid minutes every startup.
- **Cheapest RTX GPU that renders:** L4. Only step up if VRAM/throughput forces it.
- **Maximize `--num_envs`** (32, or 64 if VRAM allows): more parallel envs finish the
  30-min dataset in less wall-clock (better GPU utilization) = less paid time.
- **ACT training (~1 h, the only non-render step):** run on the same box (simplest), or to
  keep it off paid Brev hours use your free $30 Modal credit —
  `modal volume put frankapanda-data data/demos/*.hdf5 demos/` then
  `modal run modal_app.py::train`. ~15 GB hop, so only worth it to squeeze every cent.
- **Trim eval** if needed: `--num_episodes 25` instead of 50 halves eval render time.
- Collect once at `--target_minutes 30`; only re-collect if the wrist framing was wrong
  (that's why the 2-demo smoke + view check comes first).

## Parking / resuming the box (avoid the rebuild + stuck stops)
The expensive churn is **delete -> redeploy** (~15 min full rebuild) and Isaac **cold-boot
per script**. Avoid both: **stop, don't delete**, and keep one Isaac session for iteration.

Stopping preserves the built containers, the cloned repo, the pip install, the PPO
checkpoint, and the shader cache -> `resume` is ~1-3 min with NO rebuild and a faster boot.
Helper:
```bash
scripts/brev_box.sh park      # kill Isaac, then brev stop (halts GPU billing; keeps everything)
scripts/brev_box.sh resume    # brev start, wait READY, ensure containers + git pull
scripts/brev_box.sh status    # instance + container + isaac-process state
```
The helper kills the Isaac process *before* stopping because a live render job + an
unhealthy box once made a plain `brev stop` hang (flapping STOPPING<->STOPPED). Freeing the
GPU first makes the stop clean. Manual equivalent:
```bash
brev exec <inst> "docker exec vscode pkill -9 -f 'kit/python'"   # free the GPU
brev stop <inst>                                                  # halt billing (keeps disk)
brev start <inst>                                                 # later: ~1-3 min, no rebuild
```
Other start/stop savers:
- **One persistent Isaac session** for iterative renders (boot once, drive via a command
  file) instead of re-launching per task — see `scripts/cam_tune.py` / `scripts/batch_cam.py`.
- **ACT training imports no Isaac** (`--eval_interval 0`) -> zero boot cost; run without Sim.
- **Don't race the launchable's container build** on a fresh deploy: let the `vscode`
  container come up on its own before any `docker exec`, or you hit a "container name in
  use" build failure (cosmetic, but confusing).
