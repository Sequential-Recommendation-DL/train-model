# NeuMF Training Pipeline — Implementation Plan

**Branch:** feat/NeuMF
**Complexity:** Medium (~8 files, ~550 lines)
**Reference:** He et al., 2017 — "Neural Collaborative Filtering"

---

## Current State

- `data/raw/explicit/electronics.csv` and `musical_instrument.csv` exist
  - Columns: `user_id`, `parent_asin`, `rating`, `timestamp`
- `main.py` (HuggingFace downloader) — **complete**, no changes needed
- All `src/` files are empty placeholders

---

## Pipeline Overview

```
[Raw Data]
    │
    ▼
Phase 0 ── Raw Data Collection ──────────────── main.py (DONE)
    │
    ▼
Phase 1 ── Validate & Clean Data ─────────────── detect missing values, duplicates,
    │                                             inconsistencies, invalid formats; drop/fix
    ▼
Phase 2 ── Preprocessing & Feature Engineering ── encode IDs, implicit feedback,
    │                                              negative sampling, build dataset
    ▼
Phase 3 ── Data Splitting ────────────────────── leave-one-out → train / val / test
    │
    ▼
Phase 4 ── Define NeuMF Model ────────────────── GMF + MLP branches → fused output
    │
    ▼
Phase 5 ── Train Model ───────────────────────── training loop; val monitors HR@10
    │                                             to checkpoint best weights
    ▼
Phase 6 ── Analyze & Evaluate ────────────────── test set HR@K, NDCG@K
    │
    ▼
Phase 7 ── Save / Load ───────────────────────── persist checkpoint; reload for inference
    │
    ▼
 Acceptable? ──Yes──▶ Deploy / Use model
    │
   No
    │
    └──▶ loop back to Phase 1 (improve data) or Phase 5 (retrain with new config)
```

---

## NeuMF Architecture

```
GMF branch:   user_emb(gmf_dim) ⊙ item_emb(gmf_dim)       →  gmf_out
MLP branch:   concat[user_emb, item_emb]  →  FC[256→128→64→32]  →  mlp_out
NeuMF fusion: concat[gmf_out, mlp_out]  →  Linear  →  sigmoid
```

---

## Phase Details

### Phase 0 — Raw Data Collection ✅
**File:** `main.py`

Downloads `Electronics` and `Musical_Instruments` JSONL from HuggingFace and writes:
- `data/raw/explicit/electronics.csv`
- `data/raw/explicit/musical_instrument.csv`

Columns: `user_id`, `parent_asin`, `rating`, `timestamp`. No changes needed.

---

### Phase 1 — Validate & Clean Data
**Files:** `src/data/load_data.py`, `src/data/preprocess.py`

**Load:**
- `load_csv(path) -> pd.DataFrame`
- `load_all(data_dir) -> pd.DataFrame` — merges both CSVs

**Validate** (detect and report):
- Missing values in any required column (`user_id`, `parent_asin`, `rating`)
- Duplicate `(user_id, parent_asin)` pairs
- Invalid `rating` values (outside expected range)
- Invalid / null `timestamp`

**Clean** (fix and filter):
- Drop rows with missing required fields
- Deduplicate by keeping the latest interaction per `(user_id, parent_asin)`
- Drop invalid ratings
- Filter cold users/items: keep only those with `min_interactions ≥ 5`

---

### Phase 2 — Preprocessing & Feature Engineering
**Files:** `src/data/preprocess.py`, `src/features/build_features.py`

**Preprocessing:**
- Encode `user_id` and `parent_asin` to contiguous integers (0-indexed `LabelEncoder`)
- Convert to implicit feedback: any interaction → label `1`

**Feature Engineering:**
- Build per-user positive item set for fast negative lookup
- `negative_sample(train_df, num_users, num_items, num_neg=4) -> pd.DataFrame`
  - Samples 4 negatives per positive during training
- `NCFDataset(torch.utils.data.Dataset)` — yields `(user_idx, item_idx, label)` tensors
- Val/test candidates: 1 positive + 99 random negatives per user (standard NCF protocol)

---

### Phase 3 — Data Splitting
**File:** `src/data/preprocess.py`

**Leave-one-out** split per user (sorted by `timestamp`):
- **Test**: each user's chronologically last interaction
- **Val**: second-to-last interaction
- **Train**: all remaining interactions

Returns `(train_df, val_df, test_df, num_users, num_items)`

---

### Phase 4 — Define NeuMF Model
**File:** `src/models/neumf.py` *(new)*

| Hyperparameter | Value |
|---|---|
| `gmf_dim` | 64 |
| `mlp_dim` | 64 (per branch) |
| `mlp_layers` | [256, 128, 64, 32] |
| `dropout` | 0.0 |

---

### Phase 5 — Train Model
**File:** `src/models/train.py`

- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam(lr=1e-3)`
- Each epoch: train on negatively-sampled data → evaluate HR@10 on val set → checkpoint if improved
- Returns `(trained_model, history)`

| Hyperparameter | Value |
|---|---|
| `batch_size` | 256 |
| `epochs` | 20 |
| `lr` | 1e-3 |
| `num_neg_train` | 4 |

---

### Phase 6 — Analyze & Evaluate
**File:** `src/models/predict.py`

- For each user: score 100 candidates (1 positive + 99 negatives), rank by predicted score
- `hit_rate_at_k(ranked, k) -> float` — 1 if positive in top-K
- `ndcg_at_k(ranked, k) -> float` — log₂-discounted cumulative gain
- `evaluate(model, test_data, k=10, ...) -> dict[str, float]`

**Acceptance threshold:** `HR@10 ≥ 0.60` and `NDCG@10 ≥ 0.35`
If not met → loop back to Phase 1 (data quality) or Phase 5 (tune hyperparameters).

---

### Phase 7 — Save / Load
**File:** `src/models/save_load.py`

- `save_model(model, path, metadata=None)` — saves `{state_dict, config, metadata}` to `models/`
- `load_model(path, model_class, **config) -> nn.Module` — reconstructs from config + loads weights with `weights_only=True`

---

### Phase 8 — Pipeline Orchestration
**File:** `src/pipeline/pipeline.py`

Wires all phases end-to-end:
```
load_all() → validate_and_clean() → preprocess() → split() →
negative_sample() → train() → evaluate() → save_model()
```
Prints final `HR@10` and `NDCG@10`. Exits with non-zero if acceptance threshold not met.

---

## Files to Change

| File | Action | Why |
|---|---|---|
| `src/data/load_data.py` | IMPLEMENT | Load and merge CSVs |
| `src/data/preprocess.py` | IMPLEMENT | Validate, clean, encode, split |
| `src/features/build_features.py` | IMPLEMENT | Negative sampling, NCFDataset |
| `src/models/neumf.py` | CREATE | GMF + MLP + NeuMF `nn.Module` |
| `src/models/train.py` | IMPLEMENT | Training loop with val checkpointing |
| `src/models/predict.py` | IMPLEMENT | HR@K, NDCG@K evaluation |
| `src/models/save_load.py` | IMPLEMENT | torch.save / torch.load helpers |
| `src/pipeline/pipeline.py` | IMPLEMENT | End-to-end orchestration |

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Users with < 3 interactions can't fill train+val+test | Medium | `min_interactions=5` guarantees enough rows |
| Electronics CSV has millions of rows → memory pressure | High | Filter cold users/items before encoding |
| Negative sampling collides with a known positive | Low | Track per-user positive set during sampling |
| Loss NaN / divergence | Low | Use `BCEWithLogitsLoss`, verify embedding init |

---

## Run

```bash
python -m src.pipeline.pipeline
```

---

## Progress

- [ ] Phase 1 — Validate & Clean Data
- [ ] Phase 2 — Preprocessing & Feature Engineering
- [ ] Phase 3 — Data Splitting
- [ ] Phase 4 — Define NeuMF Model
- [ ] Phase 5 — Train Model
- [ ] Phase 6 — Analyze & Evaluate
- [ ] Phase 7 — Save / Load
- [ ] Phase 8 — Pipeline Orchestration
