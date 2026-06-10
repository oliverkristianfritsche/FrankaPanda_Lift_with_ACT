#!/usr/bin/env bash
# Oliver Fritsche
# June 7, 2026
# CS 7180 Advanced Perception
#
# Generate ALL README media in a single detached session on the Isaac box, so we
# only pay one SSH/launch round-trip (the box goes UNHEALTHY under repeated
# render+SSH load). Runs, in sequence: success-vs-epoch eval, the ACT triptych
# recording, the PPO oracle recording, and the analysis plots — then dumps the
# numbers the README needs (success curve + best/last loss) into this log.
#
#   brev exec <box> 'bash /workspace/FrankaPanda_Lift_with_ACT/scripts/make_media.sh' \
#       > make_media.log 2>&1 &
#
# Optional args: $1 = ACT checkpoint dir, $2 = PPO best_agent.pt (else auto-detect latest).
set -u
cd "$(dirname "$0")/.." || exit 1
unset CUDA_VISIBLE_DEVICES
PY=/isaac-sim/python.sh

ACT_DIR="${1:-$(ls -dt logs/act/checkpoints/act_* 2>/dev/null | head -1)}"
PPO_CKPT="${2:-$(ls -t logs/skrl/franka_lift/*/checkpoints/best_agent.pt 2>/dev/null | head -1)}"
EPISODES="${EPISODES:-8}"
echo "ACT_DIR=$ACT_DIR"
echo "PPO_CKPT=$PPO_CKPT"
echo "EPISODES=$EPISODES"
[ -n "$ACT_DIR" ] && [ -f "$ACT_DIR/best_model.pt" ] || { echo "NO_ACT_CKPT"; }

# --query_freq 25: re-predict every 25 steps (closed-loop). eval defaults to full-
# chunk (0), which fails; record_policy already defaults to 25.
QF="${QF:-25}"
echo "=== [1/4] eval_checkpoints (success vs epoch, query_freq=$QF) ==="
$PY scripts/eval_checkpoints.py --ckpt_dir "$ACT_DIR" --episodes "$EPISODES" --query_freq "$QF" || echo "EVAL_FAILED"

echo "=== [2/4] record ACT student (triptych + joint plot, query_freq=$QF) ==="
$PY scripts/record_policy.py --policy act --checkpoint "$ACT_DIR/best_model.pt" \
    --query_freq "$QF" --out media/act.mp4 || echo "ACT_REC_FAILED"

echo "=== [3/4] record PPO oracle (cinematic + joint plot) ==="
$PY scripts/record_policy.py --policy ppo --checkpoint "$PPO_CKPT" \
    --out media/ppo.mp4 || echo "PPO_REC_FAILED"

echo "=== [4/4] analysis plots ==="
$PY scripts/make_plots.py --ckpt_dir "$ACT_DIR" --demos "data/demos/*.hdf5" || echo "PLOTS_FAILED"

echo "=== artifacts ==="
ls -la media/ media/plots/ 2>/dev/null
echo "=== success_curve.json ==="
cat "$ACT_DIR/success_curve.json" 2>/dev/null
echo "=== loss summary (for README) ==="
$PY - "$ACT_DIR" <<'PYEOF' 2>/dev/null || true
import json, sys, os
d = os.path.join(sys.argv[1], "results.json")
tl = json.load(open(d)).get("training_losses", []) if os.path.exists(d) else []
if tl:
    best = min(tl, key=lambda x: x["loss"])
    print(f"EPOCHS={len(tl)} LAST_LOSS={tl[-1]['loss']:.4f} "
          f"BEST_LOSS={best['loss']:.4f} BEST_EPOCH={best['epoch']}")
else:
    print("NO_RESULTS_JSON")
PYEOF

echo "MAKE_MEDIA_ALL_DONE"
