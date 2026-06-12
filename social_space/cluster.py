"""
social_space/cluster.py
========================
Cluster analysis of the user embedding space across timesteps.

Runs KMeans (k=5) on the embedding matrix at t=0, t=2, and t=4,
using the BASELINE embeddings (all_embeddings.npy).  Measures how
well the learned clusters recover the ground-truth persona labels.

Outputs:
    fig7_silhouette.png    — silhouette scores at t=0, t=2, t=4
    fig8_cluster_purity.png — cluster purity heatmap (cluster × persona)

Run from project root:
    python social_space/cluster.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
all_embeddings = np.load("embeddings/store/all_embeddings.npy")   # (50, 5, 384)

with open("embeddings/store/meta.json", "r") as f:
    meta = json.load(f)

personas_order = ["ml", "robotics", "web", "data_science", "security"]
persona_list   = [entry["persona"] for entry in meta]   # ground truth, len=50

# Map persona string → int label
persona_to_int = {p: i for i, p in enumerate(personas_order)}
true_labels    = np.array([persona_to_int[p] for p in persona_list])

print(f"[load] all_embeddings: {all_embeddings.shape}")
print(f"[load] ground truth labels: {len(true_labels)} users, 5 personas")

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def cluster_and_score(embeddings_2d, k=5):
    """
    Run KMeans(k) on a (50, 384) matrix.
    Returns (cluster_labels, silhouette, purity_matrix).
    purity_matrix: (k, 5) counts of each persona in each cluster.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(embeddings_2d)
    cluster_labels = km.labels_                          # shape (50,)

    sil = silhouette_score(embeddings_2d, cluster_labels)

    # Purity matrix: rows=clusters, cols=personas
    purity = np.zeros((k, len(personas_order)), dtype=int)
    for user_idx, c in enumerate(cluster_labels):
        p_int = true_labels[user_idx]
        purity[c, p_int] += 1

    return cluster_labels, sil, purity


# ---------------------------------------------------------------------------
# 3. RUN CLUSTERING AT t=0, t=2, t=4
# ---------------------------------------------------------------------------
TIMESTEPS = [0, 2, 4]
sil_scores = {}
purity_mats = {}

for t in TIMESTEPS:
    emb = all_embeddings[:, t, :]                        # (50, 384)
    labels, sil, purity = cluster_and_score(emb)
    sil_scores[t]   = sil
    purity_mats[t]  = purity
    print(f"  t={t}: silhouette={sil:.4f}  cluster sizes={Counter(labels)}")

# Cluster purity (scalar): fraction of points in majority class per cluster
def cluster_purity_scalar(purity_matrix):
    """Mean purity across clusters: majority class fraction per cluster."""
    totals = purity_matrix.sum(axis=1)                   # (k,)
    maj    = purity_matrix.max(axis=1)                   # (k,)
    return (maj / np.maximum(totals, 1)).mean()

for t in TIMESTEPS:
    cp = cluster_purity_scalar(purity_mats[t])
    print(f"  t={t}: cluster purity = {cp:.4f}")

# ---------------------------------------------------------------------------
# 4. FIG 7: SILHOUETTE SCORES (line chart)
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

fig, ax = plt.subplots(figsize=(7, 4.5))

sil_vals = [sil_scores[t] for t in TIMESTEPS]
ax.plot(TIMESTEPS, sil_vals, "D-", color="#9B59B6", linewidth=2.2,
        markersize=9, markerfacecolor="white", markeredgewidth=2.4,
        label="Silhouette Score (KMeans k=5)")
for t, s in zip(TIMESTEPS, sil_vals):
    ax.text(t, s + 0.012, f"{s:.3f}", ha="center", fontsize=10, fontweight="bold",
            color="#9B59B6")

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(4.1, 0.502, "good\nthreshold", color="gray", fontsize=8, va="bottom")

ax.set_xlabel("Timestep", fontweight="bold")
ax.set_ylabel("Silhouette Score", fontweight="bold")
ax.set_title("Fig 7 — Cluster Separation (Silhouette) Across Timesteps",
             fontweight="bold", pad=12)
ax.set_xticks(TIMESTEPS)
ax.set_xticklabels([f"t={t}" for t in TIMESTEPS])
ax.set_ylim(0, 1.0)
ax.legend(loc="lower left")

fig.tight_layout()
fig7_path = "experiments/figures/fig7_silhouette.png"
os.makedirs("experiments/figures", exist_ok=True)
fig.savefig(fig7_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[save] fig7 → {fig7_path}")

# ---------------------------------------------------------------------------
# 5. FIG 8: CLUSTER PURITY HEATMAP (at t=0)
# ---------------------------------------------------------------------------
# Show purity matrix at t=0 and t=4 side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

persona_labels_short = ["ML", "Robotics", "Web", "Data Sci", "Security"]

for ax_i, t in enumerate([0, 4]):
    pm = purity_mats[t].astype(float)
    # Normalise row-wise so colour = fraction
    row_sums = pm.sum(axis=1, keepdims=True)
    pm_norm  = pm / np.maximum(row_sums, 1)

    im = axes[ax_i].imshow(pm_norm, cmap="YlOrRd", vmin=0, vmax=1,
                           aspect="auto")

    # Annotation
    for ci in range(pm_norm.shape[0]):
        for pi in range(pm_norm.shape[1]):
            count = int(purity_mats[t][ci, pi])
            color = "white" if pm_norm[ci, pi] > 0.6 else "black"
            axes[ax_i].text(pi, ci, f"{count}", ha="center", va="center",
                            fontsize=11, color=color, fontweight="bold")

    axes[ax_i].set_xticks(range(5))
    axes[ax_i].set_xticklabels(persona_labels_short, fontsize=9.5)
    axes[ax_i].set_yticks(range(5))
    axes[ax_i].set_yticklabels([f"Cluster {i}" for i in range(5)], fontsize=9.5)
    axes[ax_i].set_xlabel("True Persona", fontweight="bold")
    cp = cluster_purity_scalar(purity_mats[t])
    axes[ax_i].set_title(f"t={t}  (purity={cp:.3f})", fontweight="bold")

fig.suptitle("Fig 8 — Cluster Purity Heatmap (KMeans k=5, row = cluster, cell = count)",
             fontweight="bold", y=1.02)
plt.colorbar(im, ax=axes, label="Fraction of cluster", shrink=0.8)

fig.tight_layout()
fig8_path = "experiments/figures/fig8_cluster_purity.png"
fig.savefig(fig8_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[save] fig8 → {fig8_path}")

print("\n" + "=" * 60)
print("CLUSTER ANALYSIS COMPLETE")
print("=" * 60)
print(f"t=0 silhouette : {sil_scores[0]:.4f}")
print(f"t=2 silhouette : {sil_scores[2]:.4f}")
print(f"t=4 silhouette : {sil_scores[4]:.4f}")
print("\nNext step: python social_space/recommend.py")
print("=" * 60)
