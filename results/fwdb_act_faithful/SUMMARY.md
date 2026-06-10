# fwd_b + faithful-ACT — grasp-reliability experiment

Branch `two-camera-front-wrist`. Data: 360 `fwd_b`-camera train demos
(`demos_baseline_20260609_125650.hdf5`) + 60 held-out val demos
(`demos_val/demos_baseline_20260609_151305.hdf5`).

## Recipe (faithful ACT)
Per-episode "thin" epochs, `chunk_size 100`, `batch_size 32`, `lr 1e-5`,
`lr_backbone 1e-5`, `kl_weight 10`, `2000 epochs`, `--patience 500` early stopping
(did not trigger — val loss kept improving), held-out `--val_data`. W&B:
`LiquidAI_Hackathon_05/frankapanda-grasp`, run `fwdb_act_faithful` (`8xs8dbnv`).
Eval: ACT temporal-agg (`--ta_k 0.01`), `query_freq = chunk (100)`.

## Grasp/lift success vs checkpoint (lift == place throughout)
| checkpoint | episodes | success |
|---|---|---|
| epoch 250  | 25 | 0%  |
| epoch 500  | 25 | 0%  |
| epoch 750  | 25 | 4%  |
| epoch 1000 | 25 | 4%  |
| epoch 1250 | 25 | 8%  |
| epoch 1500 | 25 | 8%  |
| epoch 1750 | 25 | 12% |
| epoch 2000 | 25 | 8%  |
| **best_model (epoch 1926)** | **50** | **22% (11/50)** |

Best-model successes are precise — the cube lands **1–38 mm** from the goal.

## Findings
- Faithful-ACT (lr 1e-5) climbs **slowly and monotonically** to ~22% — no early peak,
  unlike the tuned baseline (lr 5e-5, which peaks at epoch 5 then overfits).
- **22% ties the original-camera baseline** (22% temporal-agg @ 50 ep). The ~22%
  grasp ceiling held across cameras (orig ↔ fwd_b) and recipes (tuned ↔ faithful-ACT);
  only augmentation did worse (~14%).
- ACT norms are 80–95%, so ~22% is a **pipeline limit, not fundamental**. Failures
  localise to the **grasp**; placement is solved (mm-precise once grasped). Sim → no
  visual domain shift; 360 demos → not a quantity problem.

## Diagnosis & next step
The PPO oracle only ever demonstrates **perfect grasps → no recovery data**, so ACT
can't correct a ~1 cm approach error and misses. Next: **noise-injected "recovery"
demos** (perturb the closed-loop oracle so it corrects back), optionally with a
**shorter chunk (~20)** for grasp reactivity.
