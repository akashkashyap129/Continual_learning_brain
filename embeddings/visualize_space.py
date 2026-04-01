# embeddings/visualize_space.py
# Produces 4 publication-ready figures showing the embedding space.
# Run from project root: python embeddings/visualize_space.py
#
# Figures saved to experiments/figures/
#   fig1_pca_t0.png             — PCA scatter at t=0 (initial state)
#   fig2_trajectories.png       — drift arrows t=0 → t=final
#   fig3_similarity_matrix.png  — pairwise cosine similarity heatmap
#   fig4_drift_over_time.png    — per-user drift from t=0 over time

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = Path("embeddings/store/all_embeddings.npy")
META_PATH       = Path("embeddings/store/meta.json")
FIGURES_DIR     = Path("experiments/figures")

# ── colour scheme — one colour per persona ────────────────────────────────────
PERSONA_COLORS = {
    "ml":           "#534AB7",
    "robotics":     "#D85A30",
    "web":          "#1D9E75",
    "data_science": "#BA7517",
    "security":     "#888780",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"Loaded embeddings : {embeddings.shape}  (users × timesteps × dims)")
    print(f"Loaded meta       : {len(meta)} users\n")
    return embeddings, meta


def get_colors(meta: list[dict]) -> list[str]:
    return [PERSONA_COLORS[m["persona"]] for m in meta]


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(matrix)


def style_ax(ax) -> None:
    ax.tick_params(labelsize=9, colors="#888780")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D3D1C7")
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.set_facecolor("#FAFAF8")
    ax.set_xlabel("PC 1", fontsize=10, color="#5F5E5A")
    ax.set_ylabel("PC 2", fontsize=10, color="#5F5E5A")


def legend_patches(ax) -> None:
    patches = [
        mpatches.Patch(color=c, label=p.replace("_", " ").title())
        for p, c in PERSONA_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=9, framealpha=0.95,
              edgecolor="#D3D1C7", loc="best")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — PCA scatter at t=0
# ─────────────────────────────────────────────────────────────────────────────

def fig1_pca_t0(embeddings: np.ndarray, meta: list[dict]) -> None:
    """
    Shows where every user sits in 2D space at the very start (t=0).
    If personas cluster together, the embeddings are already meaningful.
    """
    coords = pca_2d(embeddings[:, 0, :])
    colors = get_colors(meta)

    fig, ax = plt.subplots(figsize=(8, 6))
    style_ax(ax)

    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color=colors[i], s=80, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(meta[i]["user_id"], (x, y), fontsize=6,
                    color="#888780", xytext=(4, 4),
                    textcoords="offset points")

    legend_patches(ax)
    ax.set_title(
        "Embedding space at t=0  —  each dot is one user\n"
        "Colour = persona. Proximity = semantic similarity.",
        fontsize=11, fontweight="medium", pad=12)

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_pca_t0.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Trajectory arrows (t=0 → t=final)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_trajectories(embeddings: np.ndarray, meta: list[dict]) -> None:
    """
    Fits PCA on t=0 and t=final together so both share the same 2D axes.
    Draws an arrow from each user's starting position to their final position.
    Long arrows = large drift. Short arrows = stable user.
    """
    t0, tf = embeddings[:, 0, :], embeddings[:, -1, :]
    combined = np.vstack([t0, tf])
    all_2d   = PCA(n_components=2, random_state=42).fit_transform(combined)
    c0, cf   = all_2d[:len(meta)], all_2d[len(meta):]
    colors   = get_colors(meta)

    fig, ax = plt.subplots(figsize=(8, 6))
    style_ax(ax)

    for i in range(len(meta)):
        ax.scatter(*c0[i], color=colors[i], s=45, alpha=0.30,
                   edgecolors="none", zorder=2)
        ax.scatter(*cf[i], color=colors[i], s=80, alpha=0.90,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate("",
            xy=cf[i], xytext=c0[i],
            arrowprops=dict(arrowstyle="->", color=colors[i],
                            alpha=0.45, lw=1.0))

    legend_patches(ax)
    ax.set_title(
        "User drift  —  faded dot = t=0,  solid dot = t=final\n"
        "Arrow direction and length show how interests evolved.",
        fontsize=11, fontweight="medium", pad=12)

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_trajectories.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Cosine similarity heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig3_similarity_matrix(embeddings: np.ndarray, meta: list[dict]) -> None:
    """
    Pairwise cosine similarity between all users at t=0.
    Users are sorted by persona so diagonal blocks appear — these blocks
    show intra-persona similarity. Off-diagonal = cross-persona.
    A good embedding shows bright diagonal blocks and dark off-diagonal.
    """
    personas = [m["persona"] for m in meta]
    order    = sorted(range(len(meta)), key=lambda i: personas[i])
    t0_sorted = embeddings[order, 0, :]
    sim_matrix = cosine_similarity(t0_sorted)

    sorted_personas = [personas[o] for o in order]
    counts = {}
    for p in sorted_personas:
        counts[p] = counts.get(p, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.82)

    boundary = 0
    for persona, count in counts.items():
        boundary += count
        if boundary < len(meta):
            ax.axhline(boundary - 0.5, color="white", lw=1.5)
            ax.axvline(boundary - 0.5, color="white", lw=1.5)

    ticks, tlabels = [], []
    pos = 0
    for persona, count in counts.items():
        ticks.append(pos + count / 2 - 0.5)
        tlabels.append(persona.replace("_", " "))
        pos += count

    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=9, rotation=20)
    ax.set_yticks(ticks); ax.set_yticklabels(tlabels, fontsize=9)
    ax.set_title(
        "Pairwise cosine similarity — all 50 users at t=0\n"
        "Bright diagonal = same-persona similarity. Dark off-diagonal = cross-persona.",
        fontsize=11, fontweight="medium", pad=12)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_similarity_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-user drift from t=0 over time
# ─────────────────────────────────────────────────────────────────────────────

def fig4_drift_over_time(embeddings: np.ndarray, meta: list[dict]) -> None:
    """
    For each user at each timestep: drift = 1 - cosine_sim(embedding_t, embedding_t0)
    0.0 = identical to starting point (no drift / no forgetting)
    1.0 = completely different from starting point (total drift / total forgetting)

    THIS IS THE EXACT SAME METRIC USED IN PHASE 2.
    This figure is your baseline — Phase 2 will show whether EWC and
    replay buffer keep this drift lower than the naive baseline.
    """
    N, T, _ = embeddings.shape
    personas = [m["persona"] for m in meta]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Thin lines — individual users
    for i in range(N):
        t0_v = embeddings[i, 0:1, :]
        drifts = [
            1.0 - cosine_similarity(t0_v, embeddings[i, t:t+1, :])[0, 0]
            for t in range(T)
        ]
        ax.plot(range(T), drifts,
                color=PERSONA_COLORS[personas[i]], alpha=0.18, linewidth=0.9)

    # Thick lines — per-persona mean
    for persona, color in PERSONA_COLORS.items():
        idx = [i for i, m in enumerate(meta) if m["persona"] == persona]
        mean_drifts = []
        for t in range(T):
            ds = [1.0 - cosine_similarity(
                      embeddings[i, 0:1, :],
                      embeddings[i, t:t+1, :])[0, 0]
                  for i in idx]
            mean_drifts.append(np.mean(ds))
        ax.plot(range(T), mean_drifts, color=color, linewidth=2.5,
                label=persona.replace("_", " ").title(), zorder=5)

    ax.set_xticks(range(T))
    ax.set_xticklabels([f"t={t}" for t in range(T)])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Drift from t=0  (1 − cosine similarity)", fontsize=11)
    ax.set_title(
        "Interest drift from initial state over time\n"
        "Thin = individual users. Thick = persona average. "
        "This curve is your Phase 2 forgetting baseline.",
        fontsize=11, fontweight="medium", pad=12)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#D3D1C7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAF8")
    ax.grid(axis="y", color="#E8E6DF", linewidth=0.8, zorder=0)

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_drift_over_time.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative summary — numbers to present to your panel
# ─────────────────────────────────────────────────────────────────────────────

def print_quantitative_summary(embeddings: np.ndarray,
                                meta: list[dict]) -> None:
    personas = [m["persona"] for m in meta]
    t0       = embeddings[:, 0, :]
    N        = len(meta)

    print("\n" + "═" * 60)
    print("  QUANTITATIVE RESULTS  —  present these to your panel")
    print("═" * 60)

    # ── 1. Cluster separation ─────────────────────────────────────────────
    same_sims, diff_sims = [], []
    for i in range(N):
        for j in range(i + 1, N):
            sim = cosine_similarity(t0[i:i+1], t0[j:j+1])[0, 0]
            if personas[i] == personas[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    sep = np.mean(same_sims) / np.mean(diff_sims)
    print(f"\n  1. Cluster separation at t=0")
    print(f"     Same-persona avg similarity   : {np.mean(same_sims):.4f}")
    print(f"     Cross-persona avg similarity  : {np.mean(diff_sims):.4f}")
    print(f"     Separation ratio              : {sep:.2f}x")
    print(f"     Interpretation: same-persona users are {sep:.1f}x more")
    print(f"     similar to each other than to users in other personas.")

    # ── 2. Per-persona final drift ────────────────────────────────────────
    print(f"\n  2. Average drift by final timestep (1 − cosine sim from t=0)")
    all_drifts = []
    for persona in PERSONA_COLORS:
        idx = [i for i, m in enumerate(meta) if m["persona"] == persona]
        drifts = [
            1.0 - cosine_similarity(
                embeddings[i, 0:1, :],
                embeddings[i, -1:, :])[0, 0]
            for i in idx
        ]
        all_drifts.extend(drifts)
        print(f"     {persona:<20} : {np.mean(drifts):.4f}  "
              f"(range {np.min(drifts):.3f}–{np.max(drifts):.3f})")

    # ── 3. Overall drift ──────────────────────────────────────────────────
    print(f"\n  3. Overall drift (all 50 users, t=0 → t=final)")
    print(f"     Mean   : {np.mean(all_drifts):.4f}")
    print(f"     Std    : {np.std(all_drifts):.4f}")
    print(f"     Min    : {np.min(all_drifts):.4f}")
    print(f"     Max    : {np.max(all_drifts):.4f}")

    # ── 4. What this means for Phase 2 ───────────────────────────────────
    baseline = np.mean(all_drifts)
    print(f"\n  4. Phase 2 target")
    print(f"     Baseline drift (naive overwrite) : ~{baseline:.4f}")
    print(f"     EWC target                       : < {baseline * 0.75:.4f}  (25% less)")
    print(f"     Replay buffer target             : < {baseline * 0.80:.4f}  (20% less)")
    print(f"\n  If EWC and replay drift is below these targets,")
    print(f"  your continual learning claim is proven.")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, meta = load_data()

    print("Generating figures...\n")
    fig1_pca_t0(embeddings, meta)
    fig2_trajectories(embeddings, meta)
    fig3_similarity_matrix(embeddings, meta)
    fig4_drift_over_time(embeddings, meta)

    print_quantitative_summary(embeddings, meta)

    print(f"Done. Open experiments/figures/ to view all 4 figures.")