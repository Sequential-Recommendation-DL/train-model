# NeuMF — Neural Matrix Factorization

## 1. Kiến trúc tổng quan

NeuMF kết hợp hai nhánh song song: **GMF** (Generalized Matrix Factorization) và **MLP** (Multi-Layer Perceptron). Mỗi nhánh có bộ embedding riêng, không chia sẻ trọng số với nhau.

```
user_id ──► gmf_user_emb (64d) ──► element-wise product ──► gmf_out (64d) ──► Dropout(0.5) ──┐
item_id ──► gmf_item_emb (64d) ──┘                                                             │
                                                                                               ├──► concat (72d) ──► Linear(72,1) ──► logit
user_id ──► mlp_user_emb (64d) ──► concat (128d) ──► [Linear→ReLU→Dropout] × 5 ──► (8d) ──────┘
item_id ──► mlp_item_emb (64d) ──┘
```

**4 bảng embedding** (`neumf.py:35–38`):

| Tên | Shape | Vai trò |
|-----|-------|---------|
| `gmf_user` | `(num_users, 64)` | Biểu diễn người dùng cho nhánh tuyến tính |
| `gmf_item` | `(num_items, 64)` | Biểu diễn sản phẩm cho nhánh tuyến tính |
| `mlp_user` | `(num_users, 64)` | Biểu diễn người dùng cho nhánh phi tuyến |
| `mlp_item` | `(num_items, 64)` | Biểu diễn sản phẩm cho nhánh phi tuyến |

Tất cả embedding khởi tạo với `N(0, 0.01)` (`neumf.py:55`).

---

## 2. Chi tiết từng nhánh

### Nhánh GMF (`neumf.py:80–82`)

```python
gmf_out = gmf_user(u) * gmf_item(i)          # element-wise product, shape (B, 64)
gmf_out = F.dropout(gmf_out, p=0.5, training=self.training)
```

Phép nhân element-wise mô hình hóa **tương tác tuyến tính** giữa user và item. Nếu một chiều embedding của user cao và item cũng cao → tích lớn → model học được sự tương đồng trực tiếp. Dropout 0.5 áp dụng sau tích để tránh GMF branch ghi nhớ tập train (fix overfitting).

### Nhánh MLP (`neumf.py:40–45`)

```python
mlp_in  = concat(mlp_user(u), mlp_item(i))   # shape (B, 128)
mlp_out = Sequential(                         # shape (B, 8)
    Linear(128, 64), ReLU(), Dropout(0.5),
    Linear(64,  32), ReLU(), Dropout(0.5),
    Linear(32,  16), ReLU(), Dropout(0.5),
    Linear(16,   8), ReLU(), Dropout(0.5),
    Linear( 8,   8), ReLU(), Dropout(0.5),
)
```

Phép nối (concat) rồi đưa qua MLP cho phép model học **tương tác phi tuyến** phức tạp. Kích thước giảm dần theo tỉ lệ 1/2 mỗi tầng (halving funnel) cho đến khi nhỏ hơn `min_hidden=8`.

### Lớp output (`neumf.py:47`)

```python
logit = Linear(64 + 8, 1)(concat(gmf_out, mlp_out))   # shape (B,)
```

Không có sigmoid trong `forward`. Logit thô được dùng trực tiếp trong BPR loss khi train, và để ranking khi inference.

---

## 3. Số lượng tham số (ví dụ 250k users, 50k items)

| Thành phần | Tham số |
|------------|---------|
| Embeddings (4 bảng) | `(250k + 50k) × 64 × 2 ≈ 38.4M` |
| MLP layers | `128×64 + 64×32 + 32×16 + 16×8 + 8×8 = 11,008` |
| Output layer | `72 × 1 + 1 = 73` |

Phần lớn tham số nằm ở embedding → weight decay và dropout là quan trọng nhất để regularize.

---

## 4. Huấn luyện

### Loss — BPR (Bayesian Personalized Ranking)

```
L = -mean( log σ( score(u, pos) - score(u, neg) ) )
```

Thay vì dự đoán rating tuyệt đối, BPR chỉ yêu cầu **pos item xếp hạng cao hơn neg item**. Phù hợp với implicit feedback (chỉ có tương tác, không có nhãn rõ ràng).

### Hard Negative Mining (`train.py:18–59`)

Mỗi `hard_neg_freq=3` epoch, neg item được chọn bằng cách tính cosine similarity giữa GMF user embedding và toàn bộ item embedding:

```
scores = gmf_user_emb[u] @ gmf_item_emb.T    # (B, num_items)
mask out positive items
chọn top-K items có score cao nhất làm hard negatives
```

Hard negatives là những item *gần* với user trong không gian embedding nhưng chưa được tương tác — khó phân biệt hơn random negatives, buộc model học biểu diễn tốt hơn.

### Validation dùng random negatives (`train.py:62–89`)

Val loss dùng **random negatives cố định** (seed=42, xây dựng một lần). Lý do: hard negatives trên val sẽ collapse về 0 khi model tốt lên vì chính model tạo ra hard negatives — không phản ánh generalization. Random negatives cho tín hiệu val loss ổn định và khách quan.

### Optimizer và Schedule (`train.py:109–110`)

```python
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = CosineAnnealingLR(T_max=20, eta_min=0.00001)
```

**AdamW** tách weight decay ra khỏi gradient adaptivity — đảm bảo L2 penalty `1e-3` tác động đồng đều lên tất cả tham số kể cả embedding. LR giảm dần theo cosine từ `1e-3` xuống `1e-5` trong 20 epoch.

### Early Stopping

Theo dõi `HR@10` trên val set. Không cải thiện sau `patience=5` epoch liên tiếp → dừng và load lại `best_state`. Checkpoint tốt nhất giữ trong bộ nhớ qua `copy.deepcopy(model.state_dict())`.

---

## 5. Đánh giá

### Giao thức Leave-One-Out

`split()` (`preprocess.py:91–106`): với mỗi user, interaction mới nhất → test, mới nhì → val, còn lại → train. Với `min_interactions=5`: mỗi user có ≥ 3 training interactions, 1 val, 1 test.

### Metrics

**HR@10** (Hit Rate): tỉ lệ user có test item trong top-10 predictions (pool = 1 pos + 99 random neg).

```
HR@10 = số user có test item trong top-10 / tổng số user eval
```

**NDCG@10**: tính đến vị trí xếp hạng — rank càng cao điểm càng nhiều.

```
NDCG@10 = 1 / log2(rank + 1)   nếu test item trong top-10, ngược lại = 0
```

Phase 5 cảnh báo nếu `HR@10 < 0.60` hoặc `NDCG@10 < 0.35`.

---

## 6. Luồng dữ liệu đầy đủ

```
main.py
  └─ tải JSONL từ HuggingFace → data/raw/explicit/{electronics,musical_instrument}.csv

python -m data.preprocess.build --rows 250000
  ├─ load_all()           đọc 2 CSV, concat
  ├─ validate()           kiểm tra null / duplicate / invalid rating
  ├─ clean()              xóa null, xóa duplicate, clip rating, lọc user < 5 interactions
  ├─ sample 250k users    (seed=42)
  ├─ encode()             pd.factorize → user_idx, item_idx
  ├─ split()              leave-one-out per user
  └─ saveCleanData()      → data/processed/mi5_u250000_{train,val,test}.pkl + meta.json

python -m src.pipeline.pipeline
  ├─ loadCacheData()      đọc pkl từ data/processed/
  ├─ build_user_pos()     dict[user_idx → set(item_idx)] từ train_df
  ├─ NeuMF(...)           khởi tạo model
  ├─ train()              BPR + hard neg mining + AdamW + CosineAnnealingLR + early stop
  ├─ evaluate_full()      HR@10, NDCG@10, AUC, F1 trên test set
  └─ save_results()       checkpoint .pt + metrics.json + plots → results/{timestamp}/
```

---

## 7. Xác minh model xây dựng đúng

```bash
python -c "
from src.models.neumf import NeuMF
import torch
m = NeuMF(num_users=1000, num_items=500)
m.train()
u = torch.randint(0, 1000, (8,))
i = torch.randint(0, 500,  (8,))
out = m(u, i)
assert out.shape == (8,), f'Expected (8,), got {out.shape}'
m.eval()
assert m(u, i).shape == (8,)
print('OK:', m.param_summary())
"
```

Kết quả đúng: in `OK:` kèm param summary. Output shape là `(batch_size,)` — logit vô hướng cho mỗi cặp (user, item).
