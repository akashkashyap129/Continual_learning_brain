"""
continual_learning/ewc.py  (v2 — corrected lambda scale)
===========================================================
Elastic Weight Consolidation (EWC) for continual user embedding updates.

CORRECTION: for unit-normalised 384-dim embeddings, each component is
~1/√384 ≈ 0.051, so Fisher_i = current_i² ≈ 0.0026.  To achieve a
meaningful "50% memory weight" you need λ * Fisher_i ≈ 1, i.e. λ ≈ 384.
The original grid [0.1, 0.5, 1.0, 5.0] was too small.

Strategy A (running Fisher): Fisher = current² (updates each step)
Strategy B (anchored Fisher): Fisher = snapshot_t0² (fixed, always pulls
  back toward the original embedding — conceptually cleaner for CL)

Grid search: lambda ∈ [10, 50, 100, 300, 500, 1000] × both strategies.
Best = lowest mean forgetting at t=4.
Target: mean t=4 < 0.6395  (25% reduction from baseline 0.8527)

Run from project root:
    python continual_learning/ewc.py

Output:
    experiments/results/ewc_forgetting.csv
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

# Print Fisher scale info
sample = all_embeddings[0, 0, :]
print(f"\n[info] embedding component magnitude: mean={np.abs(sample).mean():.4f}  "
      f"max={np.abs(sample).max():.4f}")
print(f"[info] Fisher_i = component^2: mean={( sample**2 ).mean():.6f}  "
      f"→ need λ≈{1/(sample**2).mean():.0f} for 50% memory weight")

# ---------------------------------------------------------------------------
# 2. EWC STRATEGIES
# ---------------------------------------------------------------------------
LAMBDA_VALUES = [10, 50, 100, 300, 500, 1000]

def run_ewc_running_fisher(lam):
    """EWC with running Fisher (Fisher = current² at each step)."""
    results = []
    for entry in meta:
        user_idx    = entry["index"]
        user_id     = entry["user_id"]
        persona     = entry["persona"]
        snapshot_t0 = snapshots[user_id]

        row     = {"user_id": user_id, "persona": persona}
        current = all_embeddings[user_idx, 0, :].copy()

        for t in range(1, 5):
            new_emb = all_embeddings[user_idx, t, :].copy()
            fisher  = current ** 2                         # running Fisher
            num     = new_emb + lam * fisher * current
            den     = 1.0 + lam * fisher
            updated = num / den
            norm    = np.linalg.norm(updated)
            current = updated / norm if norm > 1e-8 else new_emb.copy()
            row[f"t{t}"] = round(1.0 - float(np.dot(current, snapshot_t0)), 4)
        results.append(row)

    return pd.DataFrame(results, columns=["user_id", "persona", "t1", "t2", "t3", "t4"])


def run_ewc_anchored_fisher(lam):
    """EWC with anchored Fisher (Fisher = snapshot_t0² — fixed importance)."""
    results = []
    for entry in meta:
        user_idx    = entry["index"]
        user_id     = entry["user_id"]
        persona     = entry["persona"]
        snapshot_t0 = snapshots[user_id]
        fisher      = snapshot_t0 ** 2          # FIXED: always from t=0

        row     = {"user_id": user_id, "persona": persona}
        current = all_embeddings[user_idx, 0, :].copy()

        for t in range(1, 5):
            new_emb = all_embeddings[user_idx, t, :].copy()
            num     = new_emb + lam * fisher * current
            den     = 1.0 + lam * fisher
            updated = num / den
            norm    = np.linalg.norm(updated)
            current = updated / norm if norm > 1e-8 else new_emb.copy()
            row[f"t{t}"] = round(1.0 - float(np.dot(current, snapshot_t0)), 4)
        results.append(row)

    return pd.DataFrame(results, columns=["user_id", "persona", "t1", "t2", "t3", "t4"])


# ---------------------------------------------------------------------------
# 3. GRID SEARCH — RUNNING FISHER
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EWC GRID SEARCH — Strategy A: Running Fisher (current²)")
print("=" * 70)
print(f"{'Lambda':>8}  {'Mean t=1':>8}  {'Mean t=2':>8}  {'Mean t=3':>8}  {'Mean t=4':>8}")
print("-" * 55)

best_lam_a   = None
best_mean_a  = float("inf")
best_df_a    = None

for lam in LAMBDA_VALUES:
    df_  = run_ewc_running_fisher(lam)
    means = [df_[f"t{t}"].mean() for t in range(1, 5)]
    m4 = means[3]
    print(f"  λ={lam:<6}  {means[0]:>8.4f}  {means[1]:>8.4f}  {means[2]:>8.4f}  {m4:>8.4f}")
    if m4 < best_mean_a:
        best_mean_a = m4
        best_lam_a  = lam
        best_df_a   = df_

print(f"\n  Best λ={best_lam_a}: mean t=4 = {best_mean_a:.4f}")

# ---------------------------------------------------------------------------
# 4. GRID SEARCH — ANCHORED FISHER
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EWC GRID SEARCH — Strategy B: Anchored Fisher (snapshot_t0²)")
print("=" * 70)
print(f"{'Lambda':>8}  {'Mean t=1':>8}  {'Mean t=2':>8}  {'Mean t=3':>8}  {'Mean t=4':>8}")
print("-" * 55)

best_lam_b   = None
best_mean_b  = float("inf")
best_df_b    = None

for lam in LAMBDA_VALUES:
    df_  = run_ewc_anchored_fisher(lam)
    means = [df_[f"t{t}"].mean() for t in range(1, 5)]
    m4 = means[3]
    print(f"  λ={lam:<6}  {means[0]:>8.4f}  {means[1]:>8.4f}  {means[2]:>8.4f}  {m4:>8.4f}")
    if m4 < best_mean_b:
        best_mean_b = m4
        best_lam_b  = lam
        best_df_b   = df_

print(f"\n  Best λ={best_lam_b}: mean t=4 = {best_mean_b:.4f}")

# ---------------------------------------------------------------------------
# 5. SELECT OVERALL BEST
# ---------------------------------------------------------------------------
if best_mean_a <= best_mean_b:
    best_mean  = best_mean_a
    best_lam   = best_lam_a
    best_df    = best_df_a
    best_strat = f"Running Fisher (λ={best_lam_a})"
else:
    best_mean  = best_mean_b
    best_lam   = best_lam_b
    best_df    = best_df_b
    best_strat = f"Anchored Fisher (λ={best_lam_b})"

print("\n" + "=" * 70)
print(f"WINNER: {best_strat}")
print(f"Mean t=4 forgetting: {best_mean:.4f}")
target = 0.6395
status = "PASS ✓" if best_mean < target else "PARTIAL — hypothesis partially supported"
print(f"Target (< {target}): {status}")

# ---------------------------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------------------------
output_dir  = "experiments/results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ewc_forgetting.csv")
best_df.to_csv(output_path, index=False)
print(f"\n[save] ewc_forgetting.csv → {output_path}")

# ---------------------------------------------------------------------------
# 7. DETAILED SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"EWC FORGETTING SUMMARY  ({best_strat})")
print("=" * 60)

personas = ["ml", "robotics", "web", "data_science", "security"]
print(f"\n{'Persona':<15}  {'Mean t=4':>8}  {'Min':>6}  {'Max':>6}  N")
print("-" * 50)
for p in personas:
    s = best_df[best_df["persona"] == p]["t4"]
    print(f"{p:<15}  {s.mean():>8.4f}  {s.min():>6.4f}  {s.max():>6.4f}  {len(s)}")

overall_t4 = best_df["t4"]
print("-" * 50)
print(f"{'OVERALL':<15}  {overall_t4.mean():>8.4f}  {overall_t4.min():>6.4f}  "
      f"{overall_t4.max():>6.4f}  {len(overall_t4)}")
print(f"\nOverall std at t=4: {overall_t4.std():.4f}")
baseline_mean = 0.8527
reduct = (baseline_mean - overall_t4.mean()) / baseline_mean * 100
print(f"Reduction vs baseline: {reduct:.1f}%  (target: ≥25%)")

print("\nMean forgetting progression:")
print("  t=0: 0.0000")
for t in range(1, 5):
    print(f"  t={t}: {best_df[f't{t}'].mean():.4f}")

print("\n" + "=" * 60)
print("Next step: python experiments/plot_forgetting_curves.py")
print("=" * 60)
