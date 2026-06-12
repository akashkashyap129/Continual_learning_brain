"""
continual_learning/replay_buffer.py
=====================================
Experience Replay Buffer strategy for continual user embedding updates.

Instead of discarding the past completely, a fixed-capacity FIFO buffer stores
recent embeddings.  At each update the new embedding is blended with the
buffer's mean to preserve past representations:

    memory  = np.mean(buffer)            # mean of buffered embeddings
    updated = (1 - alpha) * new + alpha * memory
    updated = updated / ||updated||       # re-normalise to unit sphere

Grid search:
    capacity ∈ [1, 2, 3]   (max items in buffer)
    alpha    ∈ [0.2, 0.3, 0.5]  (weight given to memory vs new)
    → 9 combinations total

Best (capacity, alpha) = lowest mean forgetting at t=4 across 50 users.
Target: mean t=4 < 0.6822  (20% reduction from baseline 0.8527)

Run from project root:
    python continual_learning/replay_buffer.py

Output:
    experiments/results/replay_forgetting.csv
"""

import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
all_embeddings = np.load("embeddings/store/all_embeddings.npy")   # (50, 5, 384)
print(f"[load] all_embeddings shape: {all_embeddings.shape}")

with open("embeddings/store/meta.json", "r") as f:
    meta = json.load(f)
print(f"[load] meta loaded — {len(meta)} users")

snapshot_dir = "embeddings/store/snapshots"
snapshots = {}
for entry in meta:
    user_id  = entry["user_id"]
    snapshots[user_id] = np.load(os.path.join(snapshot_dir, f"{user_id}_t0.npy"))
print(f"[load] snapshots loaded — {len(snapshots)} baseline vectors")

# ---------------------------------------------------------------------------
# 2. REPLAY BUFFER GRID SEARCH
# ---------------------------------------------------------------------------
CAPACITIES = [1, 2, 3]
ALPHAS     = [0.2, 0.3, 0.5]

def run_replay(capacity, alpha):
    """Run replay buffer for a given (capacity, alpha). Returns DataFrame."""
    results = []
    for entry in meta:
        user_idx    = entry["index"]
        user_id     = entry["user_id"]
        persona     = entry["persona"]
        snapshot_t0 = snapshots[user_id]          # shape (384,)

        row     = {"user_id": user_id, "persona": persona}
        current = all_embeddings[user_idx, 0, :].copy()
        buffer  = []                               # FIFO list of embeddings

        for t in range(1, 5):
            # Add current to buffer BEFORE updating it
            buffer.append(current.copy())
            if len(buffer) > capacity:
                buffer.pop(0)                     # evict oldest (FIFO)

            new_emb = all_embeddings[user_idx, t, :].copy()

            # Memory = mean of buffered embeddings
            memory = np.mean(np.stack(buffer, axis=0), axis=0)  # shape (384,)

            # Blend new with memory
            updated = (1.0 - alpha) * new_emb + alpha * memory

            # Re-normalise to unit sphere
            norm = np.linalg.norm(updated)
            if norm > 1e-8:
                updated = updated / norm
            else:
                updated = new_emb.copy()          # fallback (degenerate)

            current = updated

            forgetting_t = 1.0 - float(np.dot(current, snapshot_t0))
            row[f"t{t}"] = round(forgetting_t, 4)

        results.append(row)

    return pd.DataFrame(results, columns=["user_id", "persona", "t1", "t2", "t3", "t4"])


print("\n" + "=" * 70)
print("REPLAY BUFFER GRID SEARCH")
print("=" * 70)
print(f"{'Cap':>4}  {'Alpha':>6}  {'Mean t=1':>8}  {'Mean t=2':>8}  {'Mean t=3':>8}  {'Mean t=4':>8}")
print("-" * 65)

best_cap   = None
best_alpha = None
best_mean  = float("inf")
best_df    = None

for cap in CAPACITIES:
    for alpha in ALPHAS:
        df_rb  = run_replay(cap, alpha)
        means  = [df_rb[f"t{t}"].mean() for t in range(1, 5)]
        mean_t4 = means[3]
        print(f"  {cap:>2}    {alpha:>5.1f}  {means[0]:>8.4f}  {means[1]:>8.4f}  {means[2]:>8.4f}  {mean_t4:>8.4f}")

        if mean_t4 < best_mean:
            best_mean  = mean_t4
            best_cap   = cap
            best_alpha = alpha
            best_df    = df_rb

print("-" * 65)
print(f"\nBest: capacity={best_cap}, alpha={best_alpha}  →  mean t=4: {best_mean:.4f}")

target = 0.6822
status = "PASS ✓" if best_mean < target else "MISS ✗ (check data)"
print(f"Target (< {target}):   {status}")

# ---------------------------------------------------------------------------
# 3. SAVE BEST RESULTS
# ---------------------------------------------------------------------------
output_dir  = "experiments/results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "replay_forgetting.csv")
best_df.to_csv(output_path, index=False)
print(f"\n[save] replay_forgetting.csv written → {output_path}")
print(f"       rows: {len(best_df)}   columns: {list(best_df.columns)}")

# ---------------------------------------------------------------------------
# 4. DETAILED SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"REPLAY FORGETTING SUMMARY  (capacity={best_cap}, alpha={best_alpha})")
print("=" * 60)

personas = ["ml", "robotics", "web", "data_science", "security"]
print(f"\n{'Persona':<15}  {'Mean t=4':>8}  {'Min':>6}  {'Max':>6}  N")
print("-" * 50)
for p in personas:
    subset = best_df[best_df["persona"] == p]["t4"]
    print(f"{p:<15}  {subset.mean():>8.4f}  {subset.min():>6.4f}  {subset.max():>6.4f}  {len(subset)}")

overall_t4 = best_df["t4"]
print("-" * 50)
print(f"{'OVERALL':<15}  {overall_t4.mean():>8.4f}  {overall_t4.min():>6.4f}  {overall_t4.max():>6.4f}  {len(overall_t4)}")
print(f"\nOverall std at t=4:  {overall_t4.std():.4f}")

baseline_mean = 0.8527
reduction_pct = (baseline_mean - overall_t4.mean()) / baseline_mean * 100
print(f"\nReduction vs baseline: {reduction_pct:.1f}%  (target: ≥20%)")

print("\nMean forgetting progression (all users):")
print("  t=0: 0.0000")
for t in range(1, 5):
    print(f"  t={t}: {best_df[f't{t}'].mean():.4f}")

print("\n" + "=" * 60)
print("Next step: python experiments/plot_forgetting_curves.py")
print("=" * 60)
