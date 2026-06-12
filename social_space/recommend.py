"""
social_space/recommend.py
==========================
User similarity network and recommendation demo.

Builds a 50×50 cosine similarity matrix from the t=4 baseline embeddings.
Thresholds the matrix to form a user graph (edges where sim > 0.85).
Draws the graph using NetworkX, coloured by persona.

Also prints a demo: given a sample user, find top-5 similar users
and predict next topics from the most-similar user's t=4 topic list.

Outputs:
    fig9_similarity_network.png

Run from project root:
    python social_space/recommend.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
all_embeddings = np.load("embeddings/store/all_embeddings.npy")   # (50, 5, 384)

with open("embeddings/store/meta.json", "r") as f:
    meta = json.load(f)

with open("data/raw/synthetic_users.json", "r") as f:
    user_data = json.load(f)

# Build a lookup: user_id → topic timeline
user_timeline = {u["user_id"]: u["timeline"] for u in user_data}

user_ids  = [entry["user_id"]  for entry in meta]
personas  = [entry["persona"]  for entry in meta]

print(f"[load] all_embeddings: {all_embeddings.shape}")
print(f"[load] {len(meta)} users, {len(user_timeline)} timelines")

# ---------------------------------------------------------------------------
# 2. BUILD SIMILARITY MATRIX AT t=4
# ---------------------------------------------------------------------------
emb_t4 = all_embeddings[:, 4, :]       # (50, 384) — unit normalised

# Cosine similarity matrix (unit vectors → dot product)
sim_matrix = emb_t4 @ emb_t4.T          # (50, 50)
print(f"[sim] matrix shape: {sim_matrix.shape}")
print(f"[sim] mean off-diagonal: {(sim_matrix.sum() - 50) / (50*49):.4f}")

# ---------------------------------------------------------------------------
# 3. BUILD GRAPH
# ---------------------------------------------------------------------------
THRESHOLD = 0.85
G = nx.Graph()

for i, uid in enumerate(user_ids):
    G.add_node(uid, persona=personas[i], idx=i)

edge_count = 0
for i in range(50):
    for j in range(i + 1, 50):
        sim = float(sim_matrix[i, j])
        if sim >= THRESHOLD:
            G.add_edge(user_ids[i], user_ids[j], weight=sim)
            edge_count += 1

print(f"[graph] nodes={G.number_of_nodes()}  edges={edge_count}  (threshold={THRESHOLD})")

# If no edges at all, lower threshold automatically
if edge_count == 0:
    print("[graph] No edges at 0.85, auto-lowering threshold to 0.70")
    THRESHOLD = 0.70
    for i in range(50):
        for j in range(i + 1, 50):
            sim = float(sim_matrix[i, j])
            if sim >= THRESHOLD:
                G.add_edge(user_ids[i], user_ids[j], weight=sim)
    edge_count = G.number_of_edges()
    print(f"[graph] Retry: edges={edge_count}")

# ---------------------------------------------------------------------------
# 4. FIG 9: DRAW THE NETWORK
# ---------------------------------------------------------------------------
PERSONA_COLORS = {
    "ml":           "#E74C3C",
    "robotics":     "#3498DB",
    "web":          "#2ECC71",
    "data_science": "#F39C12",
    "security":     "#9B59B6",
}

node_colors = [PERSONA_COLORS[G.nodes[n]["persona"]] for n in G.nodes]

# Use spring layout for nice clustering
pos = nx.spring_layout(G, seed=42, k=1.4)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "figure.dpi":       150,
})

fig, ax = plt.subplots(figsize=(11, 9))

# Draw edges (weight → line width)
edges     = G.edges(data=True)
weights   = [d["weight"] for _, _, d in edges] if edge_count > 0 else []
edge_lw   = [max(0.4, (w - THRESHOLD) / (1.0 - THRESHOLD) * 3) for w in weights]

nx.draw_networkx_edges(G, pos, ax=ax,
                       width=edge_lw if edge_lw else 0.4,
                       edge_color="#BBBBBB", alpha=0.6)

# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors,
                       node_size=280,
                       alpha=0.92)

# Clean labels: just the user number
short_labels = {n: n.replace("u0", "").replace("u", "") for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=short_labels, ax=ax,
                        font_size=6, font_color="white", font_weight="bold")

# Legend
legend_patches = [
    mpatches.Patch(color=PERSONA_COLORS[p], label=p.replace("_", " ").title())
    for p in PERSONA_COLORS
]
ax.legend(handles=legend_patches, loc="lower left",
          title="Persona", title_fontsize=10, fontsize=9,
          framealpha=0.85)

ax.set_title(
    f"Fig 9 — User Similarity Network at t=4  "
    f"(sim ≥ {THRESHOLD:.2f}, {edge_count} edges)",
    fontweight="bold", pad=14
)
ax.axis("off")
fig.tight_layout()
fig9_path = "experiments/figures/fig9_similarity_network.png"
os.makedirs("experiments/figures", exist_ok=True)
fig.savefig(fig9_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[save] fig9 → {fig9_path}")

# ---------------------------------------------------------------------------
# 5. RECOMMENDATION DEMO
# ---------------------------------------------------------------------------
def top_similar_users(query_uid, top_k=5):
    """Return top-k most similar users to query_uid at t=4."""
    idx  = user_ids.index(query_uid)
    sims = sim_matrix[idx].copy()
    sims[idx] = -1.0                        # exclude self
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [(user_ids[i], float(sims[i]), personas[i]) for i in top_idxs]

def predict_next_topics(similar_uid):
    """Use the most similar user's latest topics as next-topic prediction."""
    timeline = user_timeline[similar_uid]
    latest   = max(timeline, key=lambda x: x["timestep"])
    return latest["topics"]

DEMO_USERS = ["u001", "u003", "u010"]

print("\n" + "=" * 60)
print("RECOMMENDATION DEMO")
print("=" * 60)

for demo_uid in DEMO_USERS:
    p = personas[user_ids.index(demo_uid)]
    print(f"\nQuery user: {demo_uid}  (persona={p})")
    top = top_similar_users(demo_uid, top_k=5)

    print(f"  {'User':<8}  {'Similarity':>10}  {'Persona'}")
    print(f"  {'-'*40}")
    for rec_uid, sim, rec_persona in top:
        print(f"  {rec_uid:<8}  {sim:>10.4f}  {rec_persona}")

    if top:
        best_uid = top[0][0]
        next_topics = predict_next_topics(best_uid)
        print(f"\n  → Next-topic prediction (from {best_uid}'s latest interests):")
        print(f"    {next_topics}")

print("\n" + "=" * 60)
print("SOCIAL SPACE COMPLETE — all 9 figures generated")
print("=" * 60)
