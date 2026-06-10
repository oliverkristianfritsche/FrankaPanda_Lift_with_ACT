# Oliver Fritsche
# June 7, 2026
# CS 7180 Advanced Perception

"""Generate the analysis figures for the README from artifacts we already have on
disk (no Isaac, no GPU): the ACT training log, the success-rate-vs-epoch curve,
and the oracle demo dataset. Each figure is independent, so a missing input only
skips that one plot.

  python scripts/make_plots.py \
      --ckpt_dir logs/act/checkpoints/act_<ts> \
      --demos "data/demos/*.hdf5" --out_dir media/plots

Produces (into --out_dir):
  loss_curve.png        ACT training loss (total / L1 recon / KL) vs epoch
  success_curve.png     task success rate (%) vs training epoch
  action_smoothness.png oracle action trajectories + per-step jitter per joint
  dataset_overview.png  demos collected: episode-length + reward + joint coverage
"""
import argparse
import glob
import json
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A clean look that reads well in both GitHub light and dark themes.
for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    if style in plt.style.available:
        plt.style.use(style)
        break
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 13, "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
JOINT_COLORS = plt.cm.turbo(np.linspace(0.05, 0.95, 7))


def loss_curve(ckpt_dir, out_dir):
    path = os.path.join(ckpt_dir, "results.json")
    if not os.path.exists(path):
        print(f"[loss_curve] skip — no {path}")
        return
    with open(path) as f:
        results = json.load(f)
    losses = results.get("training_losses", [])
    if not losses:
        print("[loss_curve] skip — empty training_losses")
        return
    ep = np.array([d["epoch"] for d in losses])
    total = np.array([d["loss"] for d in losses])
    l1 = np.array([d.get("l1", np.nan) for d in losses])
    kl = np.array([d.get("kl", np.nan) for d in losses])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, total, color="#d62728", lw=2.4, label="total loss")
    ax.plot(ep, l1, color="#1f77b4", lw=1.8, label="L1 reconstruction")
    ax.plot(ep, kl, color="#2ca02c", lw=1.8, ls="--", label="KL divergence")
    bi = int(np.argmin(total))
    ax.scatter([ep[bi]], [total[bi]], color="#d62728", zorder=5, s=45)
    ax.annotate(f"best  {total[bi]:.3f} @ ep {ep[bi]}",
                (ep[bi], total[bi]), textcoords="offset points", xytext=(8, 10),
                fontsize=9, color="#d62728")
    if np.nanmax(total) / max(np.nanmin(total[total > 0]), 1e-6) > 30:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("ACT training — DETR-VAE objective")
    ax.legend(frameon=True)
    fig.tight_layout()
    p = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[loss_curve] wrote {p}  ({len(ep)} epochs, best loss {total[bi]:.4f})")


def success_curve(ckpt_dir, out_dir):
    path = os.path.join(ckpt_dir, "success_curve.json")
    if not os.path.exists(path):
        print(f"[success_curve] skip — no {path} (run eval_checkpoints.py first)")
        return
    with open(path) as f:
        data = json.load(f)
    # Only the per-epoch checkpoints (drop best_model unless it is the sole point).
    pts = [d for d in data if str(d.get("ckpt", "")).startswith("checkpoint_")] or data
    pts = sorted(pts, key=lambda d: d["epoch"])
    if not pts:
        print("[success_curve] skip — empty")
        return
    ep = np.array([d["epoch"] for d in pts])
    place = np.array([d.get("place_rate", d.get("success_rate", 0)) for d in pts])
    has_lift = any("lift_rate" in d for d in pts)
    lift = np.array([d.get("lift_rate", np.nan) for d in pts]) if has_lift else None
    n = pts[0].get("episodes", "?")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if has_lift:
        ax.plot(ep, lift, color="#2ca02c", lw=2.4, marker="s", ms=6, zorder=3, label="lift (cube off table)")
        ax.fill_between(ep, 0, lift, color="#2ca02c", alpha=0.08)
    ax.plot(ep, place, color="#9467bd", lw=2.6, marker="o", ms=7, zorder=4,
            label="place (within 5cm of goal)")
    ax.fill_between(ep, 0, place, color="#9467bd", alpha=0.12)
    for x, y in zip(ep, place):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color="#5b3a8c")
    ax.set_ylim(0, 105)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("success rate (%)")
    ax.set_title(f"Lift vs place success vs training ({n} eval episodes / checkpoint)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    p = os.path.join(out_dir, "success_curve.png")
    fig.savefig(p)
    plt.close(fig)
    extra = f", lift {lift[-1]:.0f}%" if has_lift else ""
    print(f"[success_curve] wrote {p}  (final place {place[-1]:.0f}%{extra} @ ep {ep[-1]})")


def _iter_demos(files, max_scan):
    """Yield (actions[T,8], joint_pos[T,7], length, total_reward) per demo."""
    seen = 0
    for fp in files:
        try:
            h5 = h5py.File(fp, "r")
        except Exception as e:  # noqa: BLE001
            print(f"[demos] cannot open {fp}: {e}")
            continue
        with h5:
            keys = sorted((k for k in h5.keys() if k.startswith("demo_")),
                          key=lambda k: int(k.split("_")[1]))
            for k in keys:
                if seen >= max_scan:
                    return
                g = h5[k]
                if "actions" not in g:
                    continue
                acts = np.asarray(g["actions"], dtype=np.float32)
                obs = np.asarray(g["observations"], dtype=np.float32) if "observations" in g else None
                jp = obs[:, 0:7] if obs is not None else acts[:, :7]
                length = int(g.attrs.get("length", len(acts)))
                rew = float(g.attrs.get("total_reward", np.nan))
                seen += 1
                yield acts, jp, length, rew


def action_smoothness(files, out_dir, max_scan):
    if not files:
        print("[action_smoothness] skip — no demo files")
        return
    sample_acts = None
    deltas = [[] for _ in range(7)]  # per-joint |Δ| across all scanned demos
    n = 0
    for acts, _jp, _len, _rew in _iter_demos(files, max_scan):
        if sample_acts is None and len(acts) > 5:
            sample_acts = acts
        d = np.abs(np.diff(acts[:, :7], axis=0))  # [T-1, 7]
        for j in range(7):
            deltas[j].extend(d[:, j].tolist())
        n += 1
    if n == 0:
        print("[action_smoothness] skip — no demos scanned")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    # (a) one demo's commanded joint trajectories — show they are smooth.
    t = np.arange(len(sample_acts))
    for j in range(7):
        ax1.plot(t, sample_acts[:, j], color=JOINT_COLORS[j], lw=1.8, label=f"j{j+1}")
    ax1.set_title("Oracle commanded joint targets (one demo)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("target (rad)")
    ax1.legend(ncol=4, fontsize=8, loc="upper center")
    # (b) per-step jitter distribution per joint — low = smooth.
    parts = ax2.violinplot([np.asarray(deltas[j]) for j in range(7)],
                           showmeans=True, showextrema=False)
    for j, b in enumerate(parts["bodies"]):
        b.set_facecolor(JOINT_COLORS[j])
        b.set_alpha(0.75)
    if "cmeans" in parts:
        parts["cmeans"].set_color("0.2")
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([f"j{j+1}" for j in range(7)])
    ax2.set_title(f"Per-step jitter |Δtarget| ({n} demos)")
    ax2.set_xlabel("joint")
    ax2.set_ylabel("|Δ| per step (rad)")
    fig.tight_layout()
    p = os.path.join(out_dir, "action_smoothness.png")
    fig.savefig(p)
    plt.close(fig)
    alld = np.concatenate([np.asarray(d) for d in deltas])
    print(f"[action_smoothness] wrote {p}  (median |Δ| {np.median(alld):.4f} rad/step)")


def dataset_overview(files, out_dir, max_scan, step_dt=0.02):  # env control rate = 50 Hz
    if not files:
        print("[dataset_overview] skip — no demo files")
        return
    lengths, rewards, jpos = [], [], []
    for _acts, jp, length, rew in _iter_demos(files, max_scan):
        lengths.append(length)
        rewards.append(rew)
        jpos.append(jp)
    if not lengths:
        print("[dataset_overview] skip — no demos scanned")
        return
    lengths = np.asarray(lengths)
    rewards = np.asarray(rewards)
    jpos = np.concatenate(jpos, axis=0)  # [sum T, 7]
    total_min = float(lengths.sum()) * step_dt / 60.0

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].hist(lengths, bins=20, color="#1f77b4", alpha=0.85, edgecolor="white")
    axs[0].axvline(lengths.mean(), color="#d62728", lw=2,
                   label=f"mean {lengths.mean():.0f}")
    axs[0].set_title("Episode length")
    axs[0].set_xlabel("steps")
    axs[0].set_ylabel("demos")
    axs[0].legend()

    finite = rewards[np.isfinite(rewards)]
    if finite.size:
        axs[1].hist(finite, bins=20, color="#2ca02c", alpha=0.85, edgecolor="white")
        axs[1].axvline(finite.mean(), color="#d62728", lw=2,
                       label=f"mean {finite.mean():.1f}")
        axs[1].legend()
    axs[1].set_title("Episode return")
    axs[1].set_xlabel("total reward")
    axs[1].set_ylabel("demos")

    parts = axs[2].violinplot([jpos[:, j] for j in range(7)], showextrema=True)
    for j, b in enumerate(parts["bodies"]):
        b.set_facecolor(JOINT_COLORS[j])
        b.set_alpha(0.7)
    axs[2].set_xticks(range(1, 8))
    axs[2].set_xticklabels([f"j{j+1}" for j in range(7)])
    axs[2].set_title("Joint-angle coverage")
    axs[2].set_xlabel("joint")
    axs[2].set_ylabel("position (rad)")

    fig.suptitle(f"Oracle demo dataset — {len(lengths)} demos scanned · "
                 f"~{total_min:.1f} min · {int(lengths.sum())} steps",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out_dir, "dataset_overview.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[dataset_overview] wrote {p}  ({len(lengths)} demos, ~{total_min:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=None,
                    help="ACT checkpoint dir (for results.json + success_curve.json)")
    ap.add_argument("--demos", default="data/demos/*.hdf5",
                    help="glob for demo HDF5 files")
    ap.add_argument("--out_dir", default="media/plots")
    ap.add_argument("--max_scan", type=int, default=80,
                    help="max demos to scan for the dataset/jitter plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.demos))
    print(f"demos matched: {len(files)} file(s) for '{args.demos}'")

    if args.ckpt_dir:
        loss_curve(args.ckpt_dir, args.out_dir)
        success_curve(args.ckpt_dir, args.out_dir)
    else:
        print("[loss/success] skip — no --ckpt_dir given")
    action_smoothness(files, args.out_dir, args.max_scan)
    dataset_overview(files, args.out_dir, args.max_scan)
    print("MAKE_PLOTS_DONE")


if __name__ == "__main__":
    main()
