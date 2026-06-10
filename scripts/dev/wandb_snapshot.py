# Oliver Fritsche
# June 10, 2026
# CS 7180 Advanced Perception

"""Render W&B runs as a local PNG for visual inspection — one comparable plot
instead of mismatched dashboard panels.

  WANDB_API_KEY=... python scripts/dev/wandb_snapshot.py \
      --entity LiquidAI_Hackathon_05 --project frankapanda-grasp \
      --group standardized --x samples_seen --metrics loss l1 --out /tmp/wandb_std.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", default=None, help="only runs in this W&B group")
    p.add_argument("--names", nargs="*", default=None, help="explicit run names")
    p.add_argument("--x", default="samples_seen")
    p.add_argument("--metrics", nargs="+", default=["loss"])
    p.add_argument("--logx", action="store_true")
    p.add_argument("--logy", action="store_true")
    p.add_argument("--out", default="/tmp/wandb_snapshot.png")
    a = p.parse_args()

    api = wandb.Api(timeout=30)
    runs = api.runs(f"{a.entity}/{a.project}",
                    filters={"group": a.group} if a.group else None)
    runs = [r for r in runs if (not a.names or r.name in a.names)]

    fig, axes = plt.subplots(1, len(a.metrics), figsize=(7 * len(a.metrics), 5))
    if len(a.metrics) == 1:
        axes = [axes]
    for r in runs:
        hist = r.history(keys=[a.x] + a.metrics, pandas=False)
        if not hist:
            continue
        hist = sorted(hist, key=lambda h: h.get(a.x) or 0)
        xs = [h[a.x] for h in hist if h.get(a.x) is not None]
        for ax, m in zip(axes, a.metrics):
            ys = [h.get(m) for h in hist if h.get(a.x) is not None]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), label=r.name, alpha=0.85)
    for ax, m in zip(axes, a.metrics):
        ax.set_xlabel(a.x)
        ax.set_ylabel(m)
        ax.set_title(m)
        if a.logx:
            ax.set_xscale("log")
        if a.logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
