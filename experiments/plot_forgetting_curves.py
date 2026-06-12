"""
experiments/plot_forgetting_curves.py
=======================================
Generates the two main comparison figures for the Phase 2 results.

fig5_forgetting_curves.png
  — 3 lines: baseline, EWC, replay
  — Mean forgetting ± 1 std shaded band at each timestep
  — Paired t-test significance markers between strategies

fig6_persona_forgetting.png
  — Grouped bar chart: 5 personas × 3 strategies at t=4
  — Shows which persona benefits most from each CL strategy

Run from project root:
    python experiments/plot_forgetting_curves.py

Inputs:
    experiments/results/baseline_forgetting.csv
    experiments/results/ewc_forgetting.csv
    experiments/results/replay_forgetting.csv

Outputs:
    experiments/figures/fig5_forgetting_curves.png
    experiments/figures/fig6_persona_forgetting.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# 0. STYLE SETUP
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

COLORS = {
    "baseline": "#E74C3C",   # red
    "ewc":      "#2ECC71",   # green
    "replay":   "#3498DB",   # blue
}
LABELS = {
    "baseline": "Baseline (Naive Overwrite)",
    "ewc":      "EWC (Elastic Weight Consolidation)",
    "replay":   "Replay Buffer",
}

# ---------------------------------------------------------------------------
# 1. LOAD CSVs
# ---------------------------------------------------------------------------
results_dir = "experiments/results"

df_base   = pd.read_csv(os.path.join(results_dir, "baseline_forgetting.csv"))
df_ewc    = pd.read_csv(os.path.join(results_dir, "ewc_forgetting.csv"))
df_replay = pd.read_csv(os.path.join(results_dir, "replay_forgetting.csv"))

print(f"[load] baseline  rows={len(df_base)}  cols={list(df_base.columns)}")
print(f"[load] ewc       rows={len(df_ewc)}   cols={list(df_ewc.columns)}")
print(f"[load] replay    rows={len(df_replay)} cols={list(df_replay.columns)}")

# ---------------------------------------------------------------------------
# 2. COMPUTE MEAN / STD TRAJECTORIES
# ---------------------------------------------------------------------------
timesteps = [0, 1, 2, 3, 4]

def trajectory(df):
    """Returns (means, stds) arrays of length 5 (including t=0 = 0)."""
    means = [0.0] + [df[f"t{t}"].mean() for t in range(1, 5)]
    stds  = [0.0] + [df[f"t{t}"].std()  for t in range(1, 5)]
    return np.array(means), np.array(stds)

means_base,   stds_base   = trajectory(df_base)
means_ewc,    stds_ewc    = trajectory(df_ewc)
means_replay, stds_replay = trajectory(df_replay)

print("\nMean forgetting @ t=4:")
print(f"  Baseline : {means_base[-1]:.4f}")
print(f"  EWC      : {means_ewc[-1]:.4f}   reduction = {(means_base[-1]-means_ewc[-1])/means_base[-1]*100:.1f}%")
print(f"  Replay   : {means_replay[-1]:.4f}   reduction = {(means_base[-1]-means_replay[-1])/means_base[-1]*100:.1f}%")

# ---------------------------------------------------------------------------
# 3. FIG 5: FORGETTING CURVES (main result)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

for (means, stds, key) in [
    (means_base,   stds_base,   "baseline"),
    (means_ewc,    stds_ewc,    "ewc"),
    (means_replay, stds_replay, "replay"),
]:
    col = COLORS[key]
    lbl = LABELS[key]
    ax.plot(timesteps, means, "o-", color=col, linewidth=2.2,
            markersize=6, label=lbl, zorder=3)
    ax.fill_between(timesteps,
                    means - stds,
                    means + stds,
                    color=col, alpha=0.15, zorder=2)

# --- Significance markers (paired t-test EWC vs baseline, Replay vs baseline at each t) ---
for t_idx, t in enumerate(range(1, 5)):
    b = df_base[f"t{t}"].values
    e = df_ewc[f"t{t}"].values
    r = df_replay[f"t{t}"].values

    _, p_ewc    = stats.ttest_rel(b, e)
    _, p_replay = stats.ttest_rel(b, r)

    y_top = max(means_base[t_idx+1] + stds_base[t_idx+1],
                means_ewc[t_idx+1]  + stds_ewc[t_idx+1],
                means_replay[t_idx+1] + stds_replay[t_idx+1]) + 0.04

    def sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    label_str = f"E:{sig_label(p_ewc)}  R:{sig_label(p_replay)}"
    ax.text(t, y_top, label_str, ha="center", va="bottom",
            fontsize=7.5, color="#555555")

ax.set_xlabel("Timestep", fontweight="bold")
ax.set_ylabel("Forgetting Score  (1 − cos sim to t=0)", fontweight="bold")
ax.set_title("Fig 5 — Catastrophic Forgetting: Baseline vs EWC vs Replay",
             fontweight="bold", pad=14)
ax.set_xticks(timesteps)
ax.set_xticklabels(["t=0\n(reference)", "t=1", "t=2", "t=3", "t=4"])
ax.set_ylim(-0.05, min(1.35, ax.get_ylim()[1] + 0.15))
ax.legend(loc="upper left", framealpha=0.85)

# Annotation: target lines
ax.axhline(0.6395, color=COLORS["ewc"],    linestyle="--", linewidth=1,
           alpha=0.6, label="_EWC target")
ax.axhline(0.6822, color=COLORS["replay"], linestyle="--", linewidth=1,
           alpha=0.6, label="_Replay target")
ax.text(4.08, 0.6395, "EWC\ntarget", color=COLORS["ewc"],    fontsize=7.5, va="center")
ax.text(4.08, 0.6822, "Replay\ntarget", color=COLORS["replay"], fontsize=7.5, va="center")

fig.tight_layout()
fig5_path = "experiments/figures/fig5_forgetting_curves.png"
fig.savefig(fig5_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[save] fig5 → {fig5_path}")

# ---------------------------------------------------------------------------
# 4. FIG 6: PERSONA FORGETTING BAR CHART
# ---------------------------------------------------------------------------
personas = ["ml", "robotics", "web", "data_science", "security"]
x        = np.arange(len(personas))
width    = 0.25

persona_means = {}
for key, df in [("baseline", df_base), ("ewc", df_ewc), ("replay", df_replay)]:
    persona_means[key] = [df[df["persona"] == p]["t4"].mean() for p in personas]

fig, ax = plt.subplots(figsize=(10, 5.5))

bars_b = ax.bar(x - width, persona_means["baseline"], width,
                color=COLORS["baseline"], label=LABELS["baseline"],
                alpha=0.85, edgecolor="white", linewidth=0.6)
bars_e = ax.bar(x,         persona_means["ewc"],      width,
                color=COLORS["ewc"],      label=LABELS["ewc"],
                alpha=0.85, edgecolor="white", linewidth=0.6)
bars_r = ax.bar(x + width, persona_means["replay"],   width,
                color=COLORS["replay"],   label=LABELS["replay"],
                alpha=0.85, edgecolor="white", linewidth=0.6)

# Labels on bars
for bars in [bars_b, bars_e, bars_r]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xlabel("Persona", fontweight="bold")
ax.set_ylabel("Mean Forgetting at t=4", fontweight="bold")
ax.set_title("Fig 6 — Per-Persona Forgetting at t=4: Strategy Comparison",
             fontweight="bold", pad=14)
ax.set_xticks(x)
ax.set_xticklabels([p.replace("_", "\n") for p in personas], fontsize=9.5)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0.0)

fig.tight_layout()
fig6_path = "experiments/figures/fig6_persona_forgetting.png"
fig.savefig(fig6_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[save] fig6 → {fig6_path}")

print("\n" + "=" * 60)
print("FIGURES COMPLETE")
print("=" * 60)
print("fig5_forgetting_curves.png  — main comparison (3 strategies)")
print("fig6_persona_forgetting.png — per-persona breakdown at t=4")
print("\nNext step: python social_space/cluster.py")
print("=" * 60)
