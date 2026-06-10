# Grasp-recovery experiment — FINAL (2026-06-10)

## Headline (50 episodes)

- **FINAL MODEL: place 60% / lift 80%** — `actC_short_recovery` epoch 3, `--query_freq 10`
  (ckpt act_20260610_095605/checkpoint_0003.pt on the box). Trained on short task-dense
  recovery demos (31 demo-min: σ=0.3+0.45 noisy 25 min + clean 6 min; --episode_seconds 3.0
  --end_after_goal 25).
- Prior bests: actB ep3 56% lift / actA ep1 42% place (long demos). Baseline 16%;
  exact-recipe clean control under qf10: **7%** (re-planning alone confers nothing).
- Same checkpoints under temporal aggregation: ≤17% — TA suppresses learned corrections.
Full table: W&B `eval_grasp_recovery_sweeps` + results/grasp_recovery/artifacts/;
figures media/grasp_{taxonomy,results_bars,success_vs_epoch}.png.

The grasp ceiling was TWO stacked problems, both now identified:
1. **Data**: success-filtered deterministic-oracle demos contain no corrective behavior, so
   ACT's cm-scale approach errors were unrecoverable (closed-loop taxonomy: 42/50 failures =
   close at the right time/height but median 48 mm lateral off, cube knocked away, no retry;
   0/50 never reached the cube; successes landed within ~13 mm).
   Fixed by DART-style collection: execute oracle + OU noise on arm joints, store the clean
   oracle mean as label (360 recovery demos: 180 @ σ=0.3 / 98.9% oracle success,
   180 @ σ=0.45 / 92.3%).
2. **Inference**: temporal aggregation (ta_k=0.01, ACT-paper oldest-first weighting over up
   to 50 stale plans) averages away the learned reactive corrections. Under TA, BOTH new
   models sweep flat (0–17%) — the recovery behavior was in the weights but could not
   express. Receding horizon (query_freq 10) unlocks it.

Supporting diagnosis evidence (from before the experiment): teacher-forced on demos, the 22%
checkpoint predicts gripper-close timing within ±1 step (timing/vision exonerated) but its
arm L1 during the ballistic 0.46 s approach is ~2.8x the hover error; 82% of every 250-step
demo is static goal-hover (close at t≈23±2.6, at goal by t≈45).

## Results (30 episodes, temporal-agg vs query_freq=10)

| model (data) | ckpt | TA place/lift | qf10 place/lift |
|---|---|---|---|
| Run B: clean+noisy, trim_after_goal 30 | epoch 3 | 13% / 17% | **50% / 83%** |
| Run A: clean+noisy, untrimmed | epoch 1 | 13% / 13% | **47% / 50%** |
| Run A | epoch 4 | 27% / 27% | (not yet run) |

Trim/untrimmed trade-off: B grasps best (83%) but converts ~60% of grasps to placements
(trim cut the goal-hold teaching); A converts 94% (47/50) but grasps less. Likely best
combo: B's grasping + milder trim (e.g. trim_after_goal 60–80), or A trained longer.

## Pending (box stopped; resume checklist)

1. **50-episode confirmation** of B-ckpt3 @ qf10 (was mid-run when box stopped; relaunch:
   `eval_checkpoints.py --ckpt_dir logs/act/checkpoints/act_20260610_052853 --ckpt
   checkpoint_0003.pt --query_freq 10 --episodes 50`).
2. **Attribution control**: old baseline best (act_20260609_173403/best_model.pt) @ qf10 30
   eps — how much does qf10 alone recover without the new data?
3. qf10 epoch scans: B ckpt2/ckpt4, A ckpt4 + best (ep8).
4. query_freq sweep (5/10/25) and place-gap fix (milder trim retrain) if pushing higher.
5. Update README with the two-stacked-bugs story.

Run dirs on box: A=act_20260610_035108 (8 ep), B=act_20260610_052853 (16 ep + best).
W&B: actA_recovery_demos (vck8gyf3), actB_recovery_trim — team key in /workspace/repo/.wandb_key.
