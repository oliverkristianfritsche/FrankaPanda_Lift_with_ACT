# Wrist-camera analysis — why the grasp is the bottleneck, and the fwd_b fix

Investigation of whether the **wrist camera** is what limits ACT grasp success on the
Franka lift-and-place task. All renders use the production D455 intrinsics (focal 1.93,
~90° HFOV, 240×320). Images referenced below live in this folder.

## 1. The bottleneck is the grasp, and the % is effectively grasp success
Final 50-episode eval (`results/final_eval_50ep/`): in every cell `lift` == `place`
(e.g. baseline RH 10/10, TA 11/11). A successful lift always completes the place, so the
reported place-rate **is** the grasp-and-lift success rate. Every failure is a failure to
grab + lift the cube. Since we deliberately do **not** feed privileged cube position, the
grasp depends entirely on the cameras localizing the cube vs the fingers → vision is the
prime suspect.

## 2. No train/eval camera mismatch (ruled out)
`camera_match_comparison.png`: the wrist + scene frames stored in the training demos
(`demos_baseline_20260608_095734.hdf5`, 360 demos, collected Jun 8) match the current
env render pixel-for-pixel. The wrist cam was committed as an eye-in-hand parallel mount
(`make_wrist_camera`, `pos=(0.05,0,-0.03)`), and demos were collected with it. So the 20%
is **not** a camera-mismatch artifact.

## 3. The old framing put the grasp at the bottom edge
`strip/wrist_filmstrip.png` (demo wrist cam through approach→grasp→lift): the cube is
visible from t≈19 on, but sits **small and low** in the frame — the gray hand-mount eats
the top ~half and the finger-contact point sits right at the bottom edge, the worst place
for the precision signal the policy needs.

## 4. Forward sweep → fwd_b chosen
`forward/forward_compare.png` and `strip_fwd/fwd_filmstrips_compare.png`: moving the cam
forward along its view axis (+Z in the hand frame) enlarges and centres the cube.
- `existing` z=-0.03: small/low (baseline).
- `fwd_a` z=0.00 (+3cm): bigger, centred, contact visible.
- **`fwd_b` z=0.03 (+6cm): biggest, clearest cube + contact through the whole grasp — CHOSEN.**
- `fwd_c` z=0.05 (+8cm): too close (mostly top face, form lost).
- `fwd_centered` x=0: fails (loses the cube; the +X offset is load-bearing).

## 5. Decision + next step
Adopted **fwd_b**: `make_wrist_camera` `pos` → `(0.05, 0.0, 0.03)`. Because the camera and
the demos are coupled, the real test is to **re-collect the 360 demos with this camera and
retrain ACT**, then re-run the 50-ep eval to see if grasp success rises above the ~20%
ceiling. (Other suspects — oracle precision, the vision→state info gap — remain, but the
framing was the cheapest high-leverage lever and is now improved.)

## Artifacts (keep for the report)
- `camera_match_comparison.png` — demo vs current, both cameras (no mismatch)
- `strip/wrist_filmstrip.png` — demo wrist cam through the grasp (old framing)
- `forward/forward_compare.png` — single-frame forward sweep
- `strip_fwd/fwd_filmstrips_compare.png` — existing/fwd_a/fwd_b filmstrips through the grasp
- `demo/`, `forward/fp_*.png`, `strip*/` — raw frames
- `../final_eval_50ep/SUMMARY.md` — the 50-ep matrix that established the 20% baseline
