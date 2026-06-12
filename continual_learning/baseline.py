"""
continual_learning/baseline.py
================================
Baseline strategy: naive sequential overwrite (control condition).
At each timestep the current embedding is simply replaced by the new one —
no memory, no protection, no consolidation.  This is the worst-case upper
bound on catastrophic forgetting and serves as the control against which
EWC and replay buffer will be compared.

Run from project root:
    python continual_learning/baseline.py

Output:
    experiments/results/baseline_forgetting.csv
"""

import os               # for path manipulation and directory creation
import json             # to parse meta.json (user → persona mapping)
import numpy as np      # all numerical work: loading .npy files, dot products
import pandas as pd     # building and saving the results CSV

# ---------------------------------------------------------------------------
# 1. LOAD THE MASTER EMBEDDING ARRAY
# ---------------------------------------------------------------------------
# Shape: (50, 5, 384)
#   axis 0 → user index (0–49, maps to u001–u050)
#   axis 1 → timestep   (0–4)
#   axis 2 → embedding dimension (0–383)
# All vectors are already unit-normalised (L2 norm = 1.0).
all_embeddings = np.load("embeddings/store/all_embeddings.npy")
print(f"[load] all_embeddings shape: {all_embeddings.shape}")   # (50, 5, 384)

# ---------------------------------------------------------------------------
# 2. LOAD META (user_id ↔ array index ↔ persona)
# ---------------------------------------------------------------------------
# meta.json is a list of dicts:
#   [{"index": 0, "user_id": "u001", "persona": "ml"}, ...]
with open("embeddings/store/meta.json", "r") as f:
    meta = json.load(f)

# Build two quick-lookup dicts so we never have to scan the list repeatedly.
# index_to_meta[0]  → {"index": 0, "user_id": "u001", "persona": "ml"}
index_to_meta = {entry["index"]: entry for entry in meta}
print(f"[load] meta loaded — {len(meta)} users")

# ---------------------------------------------------------------------------
# 3. LOAD T=0 SNAPSHOTS (immutable baselines)
# ---------------------------------------------------------------------------
# These files were written once during Phase 1 and must NEVER be modified.
# Each file: shape (384,), dtype float32
# They represent each user's original interest embedding at the very first
# timestep.  Forgetting is measured as drift away from this point.

snapshot_dir = "embeddings/store/snapshots"
snapshots = {}   # {user_id: np.array shape (384,)}

for entry in meta:
    user_id  = entry["user_id"]                              # e.g. "u001"
    filepath = os.path.join(snapshot_dir, f"{user_id}_t0.npy")
    snapshots[user_id] = np.load(filepath)

print(f"[load] snapshots loaded — {len(snapshots)} baseline vectors")

# ---------------------------------------------------------------------------
# 4. RUN THE BASELINE SIMULATION
# ---------------------------------------------------------------------------
# Strategy: pure overwrite.
#   At every timestep, current_embedding = all_embeddings[user_idx, t, :]
#   No blending, no penalty, no memory.  The old embedding is simply gone.
#
# Forgetting formula (works because all vectors are unit-normalised):
#   forgetting_t = 1 - cosine_similarity(current, snapshot_t0)
#               = 1 - np.dot(current, snapshot_t0)
#
# A value of 0.0 means the current embedding is identical to t=0.
# A value of 1.0 means they are orthogonal (completely unrelated directions).
# Values slightly above 1.0 are possible when cosine similarity is mildly
# negative — this is normal and expected (see project notes).

results = []   # each element will become one row in the output CSV

for entry in meta:
    user_idx = entry["index"]     # position in all_embeddings axis 0
    user_id  = entry["user_id"]   # string like "u001"
    persona  = entry["persona"]   # one of: ml, robotics, web, data_science, security

    # Immutable reference point for this user
    snapshot_t0 = snapshots[user_id]   # shape (384,)

    # Row accumulator: we will append one forgetting score per timestep
    row = {"user_id": user_id, "persona": persona}

    # current_embedding starts at t=0 (same as snapshot, so forgetting = 0)
    current_embedding = all_embeddings[user_idx, 0, :].copy()

    for t in range(1, 5):   # timesteps 1, 2, 3, 4
        # --- BASELINE UPDATE: pure overwrite ---
        # The new embedding for this timestep completely replaces the old one.
        current_embedding = all_embeddings[user_idx, t, :].copy()

        # --- FORGETTING SCORE ---
        # np.dot(a, b) = cosine similarity when both vectors have norm 1.
        forgetting_t = 1.0 - float(np.dot(current_embedding, snapshot_t0))

        # Store rounded to 4 decimal places (matches target CSV format)
        row[f"t{t}"] = round(forgetting_t, 4)

    results.append(row)

print(f"[simulation] baseline complete — {len(results)} users processed")

# ---------------------------------------------------------------------------
# 5. SAVE RESULTS TO CSV
# ---------------------------------------------------------------------------
# Output path: experiments/results/baseline_forgetting.csv
# Columns: user_id, persona, t1, t2, t3, t4
# 50 rows, one per user

output_dir = "experiments/results"
os.makedirs(output_dir, exist_ok=True)   # create if missing (harmless if exists)

output_path = os.path.join(output_dir, "baseline_forgetting.csv")

df = pd.DataFrame(results, columns=["user_id", "persona", "t1", "t2", "t3", "t4"])
df.to_csv(output_path, index=False)

print(f"[save] CSV written to: {output_path}")
print(f"       rows: {len(df)}   columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 6. PRINT SUMMARY TABLE
# ---------------------------------------------------------------------------
# Verify results match known values from Phase 1.
# Expected: overall mean at t=4 ≈ 0.8527, std ≈ 0.1294

print("\n" + "=" * 60)
print("BASELINE FORGETTING SUMMARY")
print("=" * 60)

# Per-persona mean forgetting at t=4
personas = ["ml", "robotics", "web", "data_science", "security"]
print(f"\n{'Persona':<15}  {'Mean t=4':>8}  {'Min':>6}  {'Max':>6}  N")
print("-" * 50)
for persona in personas:
    subset = df[df["persona"] == persona]["t4"]
    print(
        f"{persona:<15}  {subset.mean():>8.4f}  "
        f"{subset.min():>6.4f}  {subset.max():>6.4f}  {len(subset)}"
    )

# Overall statistics
overall_t4 = df["t4"]
print("-" * 50)
print(f"{'OVERALL':<15}  {overall_t4.mean():>8.4f}  "
      f"{overall_t4.min():>6.4f}  {overall_t4.max():>6.4f}  {len(overall_t4)}")
print(f"\nOverall std at t=4:  {overall_t4.std():.4f}")

# Sanity check
target_mean = 0.8527
tolerance   = 0.02    # ±0.02 is acceptable given synthetic data variability
actual_mean = overall_t4.mean()
status = "PASS ✓" if abs(actual_mean - target_mean) < tolerance else "CHECK ✗"
print(f"\nSanity check (expected ~{target_mean}): {actual_mean:.4f}  [{status}]")

# Mean trajectory across all timesteps (shows progression of forgetting)
print("\nMean forgetting progression (all users):")
print(f"  t=0: 0.0000 (by definition — no drift yet)")
for t in range(1, 5):
    col_mean = df[f"t{t}"].mean()
    print(f"  t={t}: {col_mean:.4f}")

print("\n" + "=" * 60)
print("Next step: python continual_learning/ewc.py")
print("=" * 60)
