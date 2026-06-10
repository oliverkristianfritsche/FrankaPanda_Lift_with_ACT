# Final ACT eval — 50 episodes, sequential, two inference modes

Settled the "train longer + augment" experiment against the early no-aug baseline.
All cells: single-env `scripts/eval_checkpoints.py`, 50 episodes, max_steps 250,
goal-conditioned. Receding-horizon uses `--query_freq 25` (chunk 50); temporal-agg
uses `--temporal_agg` (ACT-paper exp-weighted ensembling). place == lift in every
cell (a successful lift carries the cube to the goal here, so the two co-occur).

| 50 episodes | Receding-horizon (qf=25) | Temporal-agg |
|---|---|---|
| **Baseline** — `act_20260608_122109/checkpoint_0005.pt` (no-aug, epoch 5) | **20%** (10/50) | **22%** (11/50) |
| **Augment** — `act_20260609_030645/best_model.pt` (val-loss-selected, ~epoch 4000+) | **6%** (3/50) | **14%** (7/50) |

## Verdict
- **The baseline early checkpoint wins decisively in both inference modes.** The
  augmentation + long-training + validation-loss-selection pipeline produced a *worse*
  task policy than the simple epoch-5 no-aug checkpoint.
- Confirms the train-longer finding: grasp/place success peaks early and **degrades**
  with more training. Validation L1 loss is **not** a good selector for task success in
  this imitation-learning setup — `best_model.pt` had the lowest val loss (late epoch)
  but is the worst on the actual task.
- **Temporal aggregation is a small, free win** over receding-horizon: +2pp baseline,
  +8pp augment. Best measured config = baseline `checkpoint_0005.pt` under temporal-agg
  (22%).
- The 50-ep baseline (20–22%) is consistent with the earlier 27.5%/40-ep reading within
  binomial noise; the underlying baseline rate is ~20–27%.

## Keep / ship
`act_20260608_122109/checkpoint_0005.pt`, run with temporal aggregation.

## Note on parallel eval
A vectorized eval (`scripts/eval_parallel.py`, num_envs>1) was attempted to speed this
up. It does not help here: the env uses standard `CameraCfg` (each camera is a separate
viewpoint render → render cost is linear in env count, no amortization), and 50 envs ×
2 cameras exceeds the RTX descriptor-set pool (render wedges). Real speedup would require
converting to `TiledCamera` + re-validating image equivalence. Sequential eval
(~5 min/50-ep RH, ~9 min TA) is the method used.
