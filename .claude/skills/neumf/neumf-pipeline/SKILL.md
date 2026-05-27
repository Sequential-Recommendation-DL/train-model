---
name: neumf-pipeline
description: "Use when implementing, debugging, or extending the NeuMF training pipeline. Covers all 8 phases: validate & clean → preprocess → split → model → train → evaluate → save/load → orchestrate. Examples: \"implement Phase 1\", \"train the model\", \"why is HR@10 low?\", \"save and reload the model\""
---

# NeuMF Training Pipeline

Reference: `PLAN.md` | He et al., 2017 — "Neural Collaborative Filtering"

## When to Use

- "Implement Phase N" — step through pipeline phases
- "Train the NeuMF model"
- "Why is HR@10 / NDCG@10 too low?"
- "How do I save and reload the model?"
- "Add validation / cleaning logic"
- Debugging data issues, training instability, or evaluation bugs

---

## Pipeline at a Glance

```
main.py (DONE) → Phase 1: Validate & Clean
                       ↓
              Phase 2: Preprocess & Features
                       ↓
              Phase 3: Split (train/val/test)
                       ↓
              Phase 4: NeuMF Model Definition
                       ↓
              Phase 5: Train  ←──────────────────┐
                       ↓                         │
              Phase 6: Evaluate  ── fail ────────┘
                       ↓ pass
              Phase 7: Save / Load
                       ↓
              Phase 8: Pipeline Orchestration
```

---

## Implementation Checklist

```
- [ ] Phase 1 — src/data/load_data.py + src/data/preprocess.py (validate & clean)
- [ ] Phase 2 — src/data/preprocess.py + src/features/build_features.py
- [ ] Phase 3 — src/data/preprocess.py (leave-one-out split)
- [ ] Phase 4 — src/models/neumf.py (GMF + MLP + NeuMF)
- [ ] Phase 5 — src/models/train.py (training loop)
- [ ] Phase 6 — src/models/predict.py (HR@K, NDCG@K)
- [ ] Phase 7 — src/models/save_load.py (torch.save / torch.load)
- [ ] Phase 8 — src/pipeline/pipeline.py (orchestration)
```

---

## Phase 1 — Validate & Clean Data

**Files:** `src/data/load_data.py`, `src/data/preprocess.py`

### Key functions

```python
# load_data.py
def load_csv(path: str) -> pd.DataFrame: ...
def load_all(data_dir: str) -> pd.DataFrame: ...   # merges both CSVs

# preprocess.py
def validate(df: pd.DataFrame) -> pd.DataFrame: ...
def clean(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame: ...
```

### Validate — what to check

| Check | How |
|---|---|
| Missing required fields | `df[["user_id","parent_asin","rating"]].isnull().any()` |
| Duplicate `(user_id, parent_asin)` pairs | `df.duplicated(["user_id","parent_asin"]).sum()` |
| Invalid ratings | `df["rating"].between(1, 5)` |
| Null / zero timestamp | `df["timestamp"].isnull() or <= 0` |

Log a warning for each issue found; don't raise — the clean step fixes them.

### Clean — what to do

```python
df = df.dropna(subset=["user_id", "parent_asin", "rating"])
df = df[df["rating"].between(1, 5)]
df = df.sort_values("timestamp").drop_duplicates(["user_id","parent_asin"], keep="last")
# cold-start filter
counts = df.groupby("user_id")["parent_asin"].transform("count")
df = df[counts >= min_interactions]
counts = df.groupby("parent_asin")["user_id"].transform("count")
df = df[counts >= min_interactions]
```

---

## Phase 2 — Preprocessing & Feature Engineering

**Files:** `src/data/preprocess.py`, `src/features/build_features.py`

### Encode IDs

```python
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["user_idx"] = user_enc.fit_transform(df["user_id"])
df["item_idx"] = item_enc.fit_transform(df["parent_asin"])
num_users = df["user_idx"].nunique()
num_items = df["item_idx"].nunique()
```

### Implicit feedback

```python
df["label"] = 1   # any interaction = positive signal
```

### Negative sampling

```python
def negative_sample(
    df: pd.DataFrame,
    num_users: int,
    num_items: int,
    num_neg: int = 4,
) -> pd.DataFrame:
    # Build per-user positive set to avoid collisions
    user_pos: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(df["user_idx"], df["item_idx"]):
        user_pos[u].add(i)

    rows = []
    for _, row in df.iterrows():
        rows.append((row["user_idx"], row["item_idx"], 1))
        neg_count = 0
        while neg_count < num_neg:
            j = random.randint(0, num_items - 1)
            if j not in user_pos[row["user_idx"]]:
                rows.append((row["user_idx"], j, 0))
                neg_count += 1

    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])
```

### NCFDataset

```python
class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.labels[idx]
```

---

## Phase 3 — Data Splitting

**File:** `src/data/preprocess.py`

### Leave-one-out

```python
def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_idx", "timestamp"])
    df["rank"] = df.groupby("user_idx").cumcount(ascending=False)
    test  = df[df["rank"] == 0].drop(columns="rank")
    val   = df[df["rank"] == 1].drop(columns="rank")
    train = df[df["rank"] >= 2].drop(columns="rank")
    return train, val, test
```

> Users with fewer than 3 interactions are excluded by Phase 1's `min_interactions=5` filter — no special case needed here.

---

## Phase 4 — NeuMF Model

**File:** `src/models/neumf.py`

### Architecture

```python
class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 64,
        mlp_dim: int = 64,
        mlp_layers: list[int] = [256, 128, 64, 32],
    ):
        super().__init__()
        # GMF embeddings
        self.gmf_user = nn.Embedding(num_users, gmf_dim)
        self.gmf_item = nn.Embedding(num_items, gmf_dim)
        # MLP embeddings
        self.mlp_user = nn.Embedding(num_users, mlp_dim)
        self.mlp_item = nn.Embedding(num_items, mlp_dim)
        # MLP tower
        mlp_input = mlp_dim * 2
        layers = []
        for out in mlp_layers:
            layers += [nn.Linear(mlp_input, out), nn.ReLU()]
            mlp_input = out
        self.mlp = nn.Sequential(*layers)
        # Fusion
        self.output = nn.Linear(gmf_dim + mlp_layers[-1], 1)

    def forward(self, users, items):
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in  = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused   = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(fused).squeeze(-1)   # raw logits
```

### Hyperparameters

| Param | Default |
|---|---|
| `gmf_dim` | 64 |
| `mlp_dim` | 64 |
| `mlp_layers` | `[256, 128, 64, 32]` |

---

## Phase 5 — Training Loop

**File:** `src/models/train.py`

```python
def train(
    model: NeuMF,
    train_loader: DataLoader,
    val_data,            # (val_df, user_pos, num_items)
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[NeuMF, dict]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_hr, best_state, history = 0.0, None, []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for users, items, labels in train_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(users, items), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, *val_data, device=device)
        history.append({"epoch": epoch, "loss": total_loss, **metrics})

        if metrics["HR@10"] > best_hr:
            best_hr = metrics["HR@10"]
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d} | loss={total_loss:.4f} | HR@10={metrics['HR@10']:.4f}")

    model.load_state_dict(best_state)
    return model, history
```

### Hyperparameters

| Param | Default |
|---|---|
| `batch_size` | 256 |
| `epochs` | 20 |
| `lr` | 1e-3 |
| `num_neg_train` | 4 |

---

## Phase 6 — Evaluate

**File:** `src/models/predict.py`

```python
def hit_rate_at_k(scores: list[tuple[float, int]], k: int = 10) -> float:
    top_k = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    return float(any(label == 1 for _, label in top_k))

def ndcg_at_k(scores: list[tuple[float, int]], k: int = 10) -> float:
    top_k = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    for rank, (_, label) in enumerate(top_k, start=1):
        if label == 1:
            return 1.0 / math.log2(rank + 1)
    return 0.0

def evaluate(
    model: NeuMF,
    test_df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    k: int = 10,
    num_neg: int = 99,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    hr_list, ndcg_list = [], []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            u, pos = int(row["user_idx"]), int(row["item_idx"])
            negs = []
            while len(negs) < num_neg:
                j = random.randint(0, num_items - 1)
                if j not in user_pos[u] and j != pos:
                    negs.append(j)
            candidates = [pos] + negs
            users_t = torch.tensor([u] * len(candidates), dtype=torch.long).to(device)
            items_t = torch.tensor(candidates, dtype=torch.long).to(device)
            preds   = model(users_t, items_t).cpu().tolist()
            scored  = [(p, int(c == pos)) for p, c in zip(preds, candidates)]
            hr_list.append(hit_rate_at_k(scored, k))
            ndcg_list.append(ndcg_at_k(scored, k))

    return {f"HR@{k}": sum(hr_list) / len(hr_list),
            f"NDCG@{k}": sum(ndcg_list) / len(ndcg_list)}
```

### Acceptance Threshold

| Metric | Minimum |
|---|---|
| `HR@10` | ≥ 0.60 |
| `NDCG@10` | ≥ 0.35 |

If below threshold → check data quality (Phase 1) or tune `gmf_dim`, `mlp_layers`, `lr`, `num_neg_train` (Phase 5).

---

## Phase 7 — Save / Load

**File:** `src/models/save_load.py`

```python
def save_model(
    model: NeuMF,
    path: str,
    config: dict,
    metadata: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": config,
                "metadata": metadata or {}}, path)

def load_model(path: str) -> tuple[NeuMF, dict]:
    checkpoint = torch.load(path, weights_only=True)
    model = NeuMF(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["metadata"]
```

### Usage

```python
# Save after training
save_model(model, "models/neumf_best.pt",
           config={"num_users": num_users, "num_items": num_items,
                   "gmf_dim": 64, "mlp_dim": 64},
           metadata={"HR@10": 0.67, "NDCG@10": 0.41})

# Reload later
model, meta = load_model("models/neumf_best.pt")
```

---

## Phase 8 — Pipeline Orchestration

**File:** `src/pipeline/pipeline.py`

```python
def run_pipeline(data_dir: str = "data/raw/explicit",
                 model_path: str = "models/neumf_best.pt") -> None:
    # Phase 1
    df = load_all(data_dir)
    df = validate(df)
    df = clean(df, min_interactions=5)
    # Phase 2 + 3
    df, num_users, num_items, user_enc, item_enc = encode(df)
    train_df, val_df, test_df = split(df)
    # Phase 2 (features)
    user_pos = build_user_pos(train_df)
    train_sampled = negative_sample(train_df, num_users, num_items, num_neg=4)
    train_loader  = DataLoader(NCFDataset(train_sampled), batch_size=256, shuffle=True)
    # Phase 4 + 5
    model = NeuMF(num_users, num_items)
    model, history = train(model, train_loader,
                           val_data=(val_df, user_pos, num_items))
    # Phase 6
    metrics = evaluate(model, test_df, user_pos, num_items)
    print(f"Test  HR@10={metrics['HR@10']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}")
    # Phase 7
    save_model(model, model_path,
               config={"num_users": num_users, "num_items": num_items,
                       "gmf_dim": 64, "mlp_dim": 64},
               metadata=metrics)
    # Acceptance gate
    if metrics["HR@10"] < 0.60 or metrics["NDCG@10"] < 0.35:
        print("Below threshold — check data quality or retrain with different config.")
        sys.exit(1)
```

---

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---|---|---|
| `HR@10 < 0.30` | Too many cold users in test set | Check `min_interactions` filter in Phase 1 |
| Loss is NaN | Learning rate too high or bad init | Lower `lr` to `1e-4`; check embedding init |
| OOM during training | Too many users × items × negatives | Reduce `batch_size` or filter more aggressively |
| Val HR@10 not improving | `num_neg_train` too low or `epochs` too few | Increase to `num_neg=8` or `epochs=30` |
| Negative sampling very slow | Large item space | Pre-build negative pools per user |
| `load_model` fails | `weights_only` mismatch or config missing | Ensure config saved alongside state_dict |

---

## Run

```bash
# Full pipeline
python -m src.pipeline.pipeline

# Smoke-test individual phases
python -c "from src.data.load_data import load_all; df = load_all('data/raw/explicit'); print(df.shape)"
```

Expected: `HR@10 ≥ 0.60`, `NDCG@10 ≥ 0.35`
