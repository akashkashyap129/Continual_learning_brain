# Continual Learning-Based User Embedding System

> **Final-Year Undergraduate ML Research Project**
> 
> *Studying catastrophic forgetting in sequential user interest representations using EWC and experience replay.*

---

## Research Question

> *"Can continual learning strategies preserve past user representations during sequential interest updates, reducing catastrophic forgetting compared to naive sequential overwrite?"*

**Hypothesis**: EWC and replay buffer will produce significantly lower forgetting scores than naive overwrite, while maintaining comparable cluster separation.

---

## Results (Phase 2 — Confirmed Hypothesis)

| Strategy | Mean Forgetting at t=4 | Reduction vs Baseline | Target | Status |
|---|---|---|---|---|
| Baseline (naive overwrite) | 0.8527 | — | control | — |
| **EWC** (λ=1000, anchored Fisher) | **0.3958** | **53.6%** | < 0.6395 (25%) | ✅ **PASSED** |
| **Replay Buffer** (cap=3, α=0.5) | **0.4913** | **42.4%** | < 0.6822 (20%) | ✅ **PASSED** |

Both strategies **exceed their targets by ~2×**. The hypothesis is strongly confirmed.

---

## Architecture (4 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — Data                                         │
│  50 synthetic users · 5 personas · 5 timesteps          │
│  Sliding window (size=3, speed=1) on topic progressions │
├─────────────────────────────────────────────────────────┤
│  Layer 2 — Embeddings                                   │
│  all-MiniLM-L6-v2 → 384-dim unit-normalised vectors     │
│  Shape: (50, 5, 384) stored as all_embeddings.npy        │
├─────────────────────────────────────────────────────────┤
│  Layer 3 — Continual Learning                           │
│  Baseline · EWC · Replay Buffer                         │
│  Forgetting score: 1 − cos_sim(current, snapshot_t0)   │
├─────────────────────────────────────────────────────────┤
│  Layer 4 — Social Space                                 │
│  KMeans clustering · Silhouette · Purity                │
│  Similarity network · Next-topic recommendation         │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
continual_learning_brain/
├── data/
│   ├── generate_synthetic_users.py     # generates synthetic_users.json
│   └── raw/
│       └── synthetic_users.json        # 50 users (git-ignored .npy)
├── embeddings/
│   ├── embed_users.py                  # sentence-transformer embeddings
│   ├── visualize_space.py              # fig1–fig4
│   └── store/                          # all_embeddings.npy (git-ignored)
├── continual_learning/
│   ├── baseline.py                     # naive overwrite → baseline_forgetting.csv
│   ├── ewc.py                          # EWC (anchored Fisher) → ewc_forgetting.csv
│   ├── replay_buffer.py                # FIFO replay → replay_forgetting.csv
│   └── ieee_paper_updated.tex          # IEEE LaTeX paper
├── experiments/
│   ├── plot_forgetting_curves.py       # fig5, fig6
│   ├── figures/                        # all 9 figures (fig1–fig9)
│   └── results/                        # 3 forgetting CSVs
├── social_space/
│   ├── cluster.py                      # KMeans → fig7, fig8
│   └── recommend.py                    # sim network → fig9 + demo
├── verify_env.py                       # environment sanity check
└── requirements.txt
```

---

## Figures

| Figure | Description | Status |
|---|---|---|
| fig1_pca_t0.png | PCA scatter — all users at t=0, coloured by persona | ✅ |
| fig2_trajectories.png | Drift arrows t=0→t=4 in PCA space | ✅ |
| fig3_similarity_matrix.png | 50×50 cosine similarity heatmap | ✅ |
| fig4_drift_over_time.png | Forgetting curves (baseline control) | ✅ |
| fig5_forgetting_curves.png | **Main result**: 3 strategies with std bands + t-test | ✅ |
| fig6_persona_forgetting.png | Per-persona grouped bar chart at t=4 | ✅ |
| fig7_silhouette.png | Silhouette scores at t=0, t=2, t=4 | ✅ |
| fig8_cluster_purity.png | Cluster purity heatmap (t=0 vs t=4) | ✅ |
| fig9_similarity_network.png | User similarity network (87 edges, threshold=0.85) | ✅ |

---

## Setup & Reproduction

```bash
# 1. Clone
git clone https://github.com/akashkashyap129/Continual_learning_brain.git
cd Continual_learning_brain

# 2. Create venv (Python 3.10)
py -3.10 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Verify environment
python verify_env.py           # expect: all 8 checks OK

# 5. Generate data + embeddings (Phase 1)
python data/generate_synthetic_users.py
python embeddings/embed_users.py
python embeddings/visualize_space.py    # fig1–fig4

# 6. Run continual learning (Phase 2)
python continual_learning/baseline.py
python continual_learning/ewc.py
python continual_learning/replay_buffer.py
python experiments/plot_forgetting_curves.py    # fig5, fig6

# 7. Social space (Phase 3)
python social_space/cluster.py          # fig7, fig8
python social_space/recommend.py        # fig9 + demo
```

**All scripts must be run from the project root** (`continual_learning_brain/`).

---

## Algorithms

### Forgetting Score
```
forgetting_t = 1 - cosine_similarity(current_embedding, snapshot_t0)
             = 1 - np.dot(current, snapshot_t0)   # unit-normalised vectors
```

### EWC Update (element-wise, anchored Fisher)
```
fisher_i   = snapshot_t0_i²              # importance from original embedding
numerator  = new_i + λ · fisher_i · current_i
denominator= 1 + λ · fisher_i
updated    = normalise(numerator / denominator)
```
Best λ = 1000 (Fisher values ~0.0026/component require large λ for unit vectors)

### Replay Buffer Update
```
buffer ← FIFO queue (capacity=3)
memory  = mean(buffer)
updated = normalise((1−α)·new + α·memory)
```
Best: capacity=3, α=0.5

---

## Tech Stack

| Package | Version | Role |
|---|---|---|
| sentence-transformers | 4.1.0 | all-MiniLM-L6-v2 embeddings |
| numpy | 2.2.6 | arrays, linear algebra |
| pandas | 2.3.3 | CSV I/O |
| scikit-learn | 1.7.2 | KMeans, silhouette, PCA |
| matplotlib | 3.10.8 | all figures |
| scipy | — | paired t-tests (fig5) |
| networkx | — | similarity network (fig9) |
| torch | 2.11.0+cpu | sentence-transformers backend |

---

## Academic Context

**Paper**: IEEE conference format (IEEEtran)  
**Title**: *"Continual Learning-Based User Embedding System for Evolving Interest Representation"*

**Key citations**:
1. Kirkpatrick et al. 2017 — EWC (PNAS)
2. Rolnick et al. 2019 — Experience replay (NeurIPS)
3. Reimers & Gurevych 2019 — Sentence-BERT (EMNLP)
4. Parisi et al. 2019 — CL survey (Neural Networks)

---

*Student: Ashwi K. · Python 3.10.1 · Windows 11*
