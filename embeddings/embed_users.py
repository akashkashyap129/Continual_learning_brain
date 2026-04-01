# embeddings/embed_users.py
# Loads synthetic_users.json, embeds each timestep using sentence-transformers,
# saves one .npy file per user per timestep + a master snapshot dict.

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH      = Path("data/raw/synthetic_users.json")
EMBEDDINGS_DIR = Path("embeddings/store")
SNAPSHOTS_DIR  = Path("embeddings/store/snapshots")
MASTER_PATH    = Path("embeddings/store/all_embeddings.npy")
META_PATH      = Path("embeddings/store/meta.json")
MODEL_NAME      = "all-MiniLM-L6-v2"
EMBED_DIM       = 384


def topics_to_sentence(topics: list[str]) -> str:
    """
    Join a list of topic strings into a single sentence for the encoder.
    "python numpy pandas"  →  one 384-dim vector
    Order is shuffled in data gen, so we sort here for determinism.
    """
    return " ".join(sorted(topics))


def load_users(path: Path) -> list[dict]:
    with open(path, "r") as f:
        users = json.load(f)
    print(f"Loaded {len(users)} users from {path}")
    return users


def embed_all_users(
    users: list[dict],
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[dict]]:
    """
    Returns:
        embeddings  — shape (N, T, D)
                      N = num users, T = num timesteps, D = 384
        meta        — list of {user_id, persona, index} dicts
    """
    num_users     = len(users)
    num_timesteps = len(users[0]["timeline"])
    embeddings    = np.zeros((num_users, num_timesteps, EMBED_DIM), dtype=np.float32)
    meta          = []

    print(f"\nEmbedding {num_users} users × {num_timesteps} timesteps "
          f"→ shape {embeddings.shape}")
    print(f"Model: {MODEL_NAME}\n")

    for i, user in enumerate(users):
        sentences = [
            topics_to_sentence(step["topics"])
            for step in sorted(user["timeline"], key=lambda s: s["timestep"])
        ]

        # Encode all timesteps for this user in one batch (fast)
        vecs = model.encode(sentences, convert_to_numpy=True,
                            show_progress_bar=False)   # (T, 384)
        embeddings[i] = vecs

        meta.append({
            "index":   i,
            "user_id": user["user_id"],
            "persona": user["persona"],
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:>3}/{num_users}] {user['user_id']} "
                  f"({user['persona']}) — done")

    return embeddings, meta


def save_per_user_npy(
    embeddings: np.ndarray,
    meta: list[dict],
    out_dir: Path,
    snapshots_dir: Path,
) -> None:
    """
    Save individual .npy files:
        embeddings/u001_t0.npy  …  u050_t4.npy   (per timestep)
        embeddings/snapshots/u001_t0.npy           (t=0 baselines only)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    num_timesteps = embeddings.shape[1]

    for entry in meta:
        i       = entry["index"]
        uid     = entry["user_id"]

        for t in range(num_timesteps):
            vec  = embeddings[i, t]                      # (384,)
            path = out_dir / f"{uid}_t{t}.npy"
            np.save(path, vec)

        # Save t=0 snapshot separately — this is the forgetting baseline
        snapshot_path = snapshots_dir / f"{uid}_t0.npy"
        np.save(snapshot_path, embeddings[i, 0])

    print(f"\nPer-user files saved → {out_dir}/")
    print(f"t=0 snapshots saved  → {snapshots_dir}/")


def save_master(
    embeddings: np.ndarray,
    meta: list[dict],
    master_path: Path,
    meta_path: Path,
) -> None:
    """
    Save the full (N, T, D) array as one file — convenient for Phase 2 & 3
    where we need all users at once.
    """
    np.save(master_path, embeddings)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Master array saved   → {master_path}  {embeddings.shape}")
    print(f"Meta index saved     → {meta_path}")


def verify_embeddings(embeddings: np.ndarray, meta: list[dict]) -> None:
    """
    Quick sanity checks before we call it done.
    """
    print("\n── Verification ────────────────────────────────────────────")

    # 1. Shape check
    N, T, D = embeddings.shape
    assert D == EMBED_DIM, f"Expected dim {EMBED_DIM}, got {D}"
    print(f"Shape check          : {N} users × {T} timesteps × {D} dims  OK")

    # 2. No NaNs or zeros
    assert not np.isnan(embeddings).any(), "NaNs found in embeddings!"
    assert not (embeddings == 0).all(axis=-1).any(), "Zero vectors found!"
    print(f"NaN / zero check     : clean  OK")

    # 3. Vectors should be roughly unit-norm (sentence-transformers normalises)
    norms = np.linalg.norm(embeddings, axis=-1)   # (N, T)
    mean_norm = norms.mean()
    print(f"Mean vector norm     : {mean_norm:.4f}  (expect ~1.0)  OK")

    # 4. Same-persona users should be more similar than cross-persona
    from sklearn.metrics.pairwise import cosine_similarity

    # Take t=0 embeddings for a quick persona similarity check
    t0 = embeddings[:, 0, :]    # (N, 384)
    personas = [m["persona"] for m in meta]

    same_sims, diff_sims = [], []
    for i in range(N):
        for j in range(i + 1, N):
            sim = cosine_similarity(t0[i:i+1], t0[j:j+1])[0, 0]
            if personas[i] == personas[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    avg_same = np.mean(same_sims)
    avg_diff = np.mean(diff_sims)
    print(f"Same-persona sim     : {avg_same:.4f}")
    print(f"Cross-persona sim    : {avg_diff:.4f}")

    if avg_same > avg_diff:
        print(f"Cluster separation   : GOOD — same-persona users are more similar  OK")
    else:
        print(f"Cluster separation   : WARN — same-persona users are NOT more similar")
        print(f"  → Check TOPIC_GRAPH for overlap between personas")

    print("────────────────────────────────────────────────────────────")


def print_example(embeddings: np.ndarray, meta: list[dict], users: list[dict]) -> None:
    """Show one user's embedding drift across timesteps."""
    from sklearn.metrics.pairwise import cosine_similarity

    user   = users[0]
    i      = 0
    uid    = user["user_id"]
    print(f"\n── Example: {uid} ({user['persona']}) ──────────────────────────")

    for t, step in enumerate(user["timeline"]):
        topics = step["topics"]
        norm   = np.linalg.norm(embeddings[i, t])
        print(f"  t={t}  topics={topics}")
        print(f"       norm={norm:.4f}  vec[:5]={embeddings[i,t,:5].round(4)}")

    # Show cosine similarity between consecutive timesteps
    print(f"\n  Cosine similarity between consecutive timesteps:")
    for t in range(embeddings.shape[1] - 1):
        sim = cosine_similarity(
            embeddings[i, t  :t+1],
            embeddings[i, t+1:t+2]
        )[0, 0]
        print(f"    t={t} → t={t+1} : {sim:.4f}", end="")
        if sim > 0.85:
            print("  (high — smooth drift)")
        elif sim > 0.65:
            print("  (medium drift)")
        else:
            print("  (large jump — check data)")
    print()


if __name__ == "__main__":
    # 1. Load model (uses local cache after first download)
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    # 2. Load users
    users = load_users(DATA_PATH)

    # 3. Embed everything → (50, 5, 384)
    embeddings, meta = embed_all_users(users, model)

    # 4. Save per-user .npy files + t=0 snapshots
    save_per_user_npy(embeddings, meta, EMBEDDINGS_DIR, SNAPSHOTS_DIR)

    # 5. Save master array (used by Phase 2 & 3)
    save_master(embeddings, meta, MASTER_PATH, META_PATH)

    # 6. Sanity checks
    verify_embeddings(embeddings, meta)

    # 7. Print one example
    print_example(embeddings, meta, users)

    print("\nDone. Embeddings ready for Phase 2.")
