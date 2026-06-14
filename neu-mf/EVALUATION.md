# Đánh Giá Mô Hình NeuMF — Hệ Thống Gợi Ý

> **Ngày:** 14-06-2026 | **Run mới nhất:** `results/14-06-2026_13h26`  
> **Dataset:** Taobao UserBehavior (Kaggle) — 100M dòng raw

---

## 1. Động Lực và Ý Tưởng

Collaborative Filtering (CF) truyền thống — điển hình là Matrix Factorization (MF) — biểu diễn tương tác user–item bằng tích vô hướng (dot product) trong không gian embedding. Cách tiếp cận này đơn giản và hiệu quả, nhưng bị giới hạn bởi tính **tuyến tính**: tích vô hướng chỉ nắm bắt được quan hệ tuyến tính giữa các chiều embedding, bỏ sót các pattern phi tuyến phức tạp trong dữ liệu tương tác thực tế.

**Neural Collaborative Filtering** (He et al., 2017) đề xuất **NeuMF** — kết hợp hai nhánh:
- **GMF (Generalized Matrix Factorization):** giữ lại sức mạnh của dot product truyền thống thông qua phép nhân element-wise giữa hai embedding vector.
- **MLP (Multi-Layer Perceptron):** học các tương tác phi tuyến phức tạp giữa user và item thông qua nhiều lớp fully-connected.

Hai nhánh được kết hợp và tối ưu đồng thời, cho phép mô hình vừa tận dụng sự đơn giản của MF vừa học được các pattern phi tuyến mà MF không thể nắm bắt.

---

## 2. Dữ Liệu

### 2.1. Nguồn Gốc

**Taobao UserBehavior** (Kaggle) — dữ liệu hành vi người dùng trên nền tảng thương mại điện tử Taobao. Raw data gốc có **100,150,807 dòng** ghi lại 4 loại hành vi:

| Behavior | Điểm | Ý nghĩa |
|----------|------|---------|
| `pv` | 1 | Page view — xem sản phẩm |
| `fav` | 2 | Favorite — thêm vào yêu thích |
| `cart` | 3 | Add to cart — thêm vào giỏ hàng |
| `buy` | 4 | Purchase — mua hàng |

Thang điểm phản ánh **mức độ ý định mua hàng** — càng gần hành vi mua thì điểm càng cao.

### 2.2. Pipeline Xử Lý Dữ Liệu (v2)

```
Raw (100M rows)
  → Clean: xóa duplicate + lọc timestamp
  → Score: behavior → điểm (pv=1, fav=2, cart=3, buy=4)
  → Groupby (user, item): Label = sum(scores), Timestamp = max(timestamp)
  → Stratified sample 250K theo khung 4h (~209 giờ = ~9 ngày)
  → K-core filtering (k=5): loại user/item có < 5 interactions
  → Normalize: Label = tanh(Label / 5.0) → range (0, 1)
  → Leave-last-out split: interaction cuối mỗi user → val
```

### 2.3. Thống Kê Dataset Hiện Tại

| Thông số | Giá trị |
|----------|---------|
| Raw rows | 100,150,807 |
| Sample size | 250,000 |
| Tập huấn luyện | 225,000 samples |
| Tập validation | 25,000 samples (gốc), 22,318 cold-start loại bỏ |
| Users (train) | 187,767 |
| Items | 160,910 |
| Min interactions/user | 5 (k-core) |

### 2.4. Thiết Kế Label

**Trong file CSV:** Label có thể là giá trị liên tục — được normalize bằng `tanh(sum_behavior_scores / 5.0)`:

| Behavior tích lũy | Label raw | Label normalized |
|-------------------|-----------|-----------------|
| pv | 1 | **0.197** |
| fav | 2 | **0.380** |
| cart | 3 | **0.537** |
| buy | 4 | **0.665** |
| pv + buy | 5 | **0.762** |
| pv + fav + cart + buy | 10 | **0.964** |

**Trong training code:** Cả `NeuMFTrainDataset` và `NeuMFDataset` đều convert label về **binary** trước khi đưa vào model:

```python
self.labels = (pos_df['label'].values > 0).astype(np.float32)  # → 0.0 hoặc 1.0
```

Nghĩa là mọi interaction đều được coi là `1.0` bất kể mức độ tương tác (pv hay buy đều = 1.0). Negative samples = `0.0`. Model **không học được sự khác biệt** giữa pv và buy — chỉ biết có/không có tương tác.

**Phân bố lệch trái mạnh:** 75% mẫu có label ở mức `pv` duy nhất. Đây là đặc điểm phổ biến của dữ liệu implicit feedback thương mại điện tử.

### 2.5. Chiến Lược Split — Leave-Last-Out

Với mỗi user, interaction **cuối cùng theo timestamp** được giữ cho validation. Toàn bộ interactions trước đó dùng để train.

**Ưu điểm so với random split:**
- User xuất hiện ở cả train lẫn val → embedding đã được train → không cold-start.
- Đánh giá đúng bài toán: dự đoán hành vi **tiếp theo** của user.
- Theo đúng thứ tự thời gian — tránh data leakage.

---

## 3. Embedding Layer

NeuMF duy trì **4 bảng embedding độc lập** thay vì 2:

| Layer | Kích thước | Params |
|-------|-----------|--------|
| `gmf_user_embedding` | 187,767 × 16 | 3.0M |
| `gmf_item_embedding` | 160,910 × 16 | 2.6M |
| `mlp_user_embedding` | 187,767 × 16 | 3.0M |
| `mlp_item_embedding` | 160,910 × 16 | 2.6M |
| MLP layers (Linear + BN) | — | ~4.9K |
| Final layer | 32 × 1 | 33 |
| **Tổng** | | **11.2M** |

Embedding chiếm **~99.96%** tổng số tham số — đặc điểm điển hình của recommendation model khi user/item space lớn.

**Lý do tách 4 bảng embedding riêng biệt:**
- **GMF** cần embedding phản ánh "compatibility" tuyến tính giữa user và item (tối ưu cho dot product).
- **MLP** cần embedding phản ánh "feature representation" phù hợp với mạng phi tuyến (tối ưu cho concatenation → nonlinear transformation).
- Hai nhiệm vụ có gradient flow khác nhau; tách riêng giúp mỗi bảng được tối ưu độc lập cho mục tiêu của nó.

**Khởi tạo trọng số:**
- Embedding: `Normal(mean=0, std=0.01)` — giá trị nhỏ để tránh gradient exploding bước đầu.
- Linear layers: `Kaiming Uniform (a=1, nonlinearity='relu')` — phù hợp với activation ReLU.
- Bias: `zeros`.

---

## 4. Lựa Chọn Embedding Dim

Embedding dim hiện tại: **`EMBEDDING_DIM = 16`**

Mỗi user/item được biểu diễn bằng **2 vector 16 chiều** (một cho GMF, một cho MLP) → tổng 32 chiều latent space.

**Phân tích:**

| Tiêu chí | Nhận xét |
|----------|---------|
| Tốc độ training | Nhanh do embedding nhỏ |
| Risk overfitting | Thấp — regularization mạnh |
| Capacity | Có thể thiếu cho 348K+ entities |
| Paper gốc NeuMF | Dùng 8–64 với MovieLens (~10K entities) |

Với ~348K entities (users + items), 16 chiều là **thận trọng**. Tăng lên 32–64 có thể cải thiện biểu diễn. Nhưng với dataset thưa (75% mẫu chỉ là `pv`), embedding dim nhỏ giảm nguy cơ memorize noise.

---

## 5. Kiến Trúc Mô Hình

```
User ID ──┬──► [gmf_user_emb(16)] ──┬──► element-wise multiply ──► gmf_vector(16)
          │                          │
Item ID ──┼──► [gmf_item_emb(16)] ──┘
          │
          ├──► [mlp_user_emb(16)] ──┬──► cat[user, item] (32)
          │                          │    ──► Linear(32→64) + BN + ReLU + Dropout(0.4)
          └──► [mlp_item_emb(16)] ──┘    ──► Linear(64→32) + BN + ReLU + Dropout(0.4)
                                          ──► Linear(32→16) + BN + ReLU + Dropout(0.4)
                                          ──► mlp_vector(16)

cat[gmf_vector(16), mlp_vector(16)] ──► Linear(32→1) ──► sigmoid ──► score ∈ (0,1)
```

**Chi tiết MLP funnel:** `[32 → 64 → 32 → 16]` — mở rộng rồi thu hẹp. Input được nới rộng lên 64 trước khi thu hẹp về 16 — giúp model có nhiều capacity ở tầng đầu để học interactions phức tạp. Paper gốc NeuMF dùng funnel thu hẹp dần `[256 → 128 → 64]`; thiết kế này khác nhưng không sai.

**Embedding Dropout (rate=0.3):** áp dụng trực tiếp lên embedding trước khi vào từng nhánh — giảm memorization các cặp (user, item) hiếm gặp trong training set.

**Activation cuối:** `sigmoid` — đưa output về (0, 1), phù hợp với BCE loss và label đã normalize về (0, 1).

---

## 6. Loss Function

**Hàm loss:** Binary Cross Entropy (BCE)

```python
# models/neumf.py — training_step
predictions = self(user_indices, item_indices)   # sigmoid output ∈ (0, 1)
loss = F.binary_cross_entropy(predictions, labels)
# labels ∈ {0.0, 1.0}  ← binary (xem section 2.4)
```

> **Lưu ý về naming:** `training_summary.txt` và title của learning_curves.png ghi "MSE Loss" nhưng code thực dùng `F.binary_cross_entropy`. Đây là lỗi trong string literal ở `main.py:183` và `callbacks.py:81` — loss thực sự là **BCE**, không phải MSE.

BCE được áp dụng cho **binary label** (0 hoặc 1) — không phải soft/continuous. Model học phân biệt có/không tương tác, không phân biệt mức độ tương tác.

**Negative Sampling trong training:**
- 16 negatives/positive — lấy ngẫu nhiên theo phân phối uniform (`POP_ALPHA=0.0`).
- **Dynamic sampling:** `NeuMFTrainDataset.__getitem__` sample negative trực tiếp khi DataLoader gọi, không precompute → negatives khác nhau mỗi epoch.
- Tổng mẫu effective: n_train × 17 samples/epoch.

---

## 7. Quá Trình Học Embedding

Embedding học hoàn toàn từ dữ liệu tương tác, không dùng pretrained features.

**Gradient flow:**
- Chỉ embedding của user/item có trong batch mới nhận gradient → embedding của user/item hiếm gặp được cập nhật ít hơn.
- Sparsity thực tế: ~1.5 interactions/user trung bình trong train (225K / 187K) — rất thưa.
- K-core (k=5) đã lọc các user/item quá thưa trước khi training, giúp mọi embedding có đủ signal.

**Embedding dropout (rate=0.3):**
- Zero out ngẫu nhiên 30% chiều embedding trong mỗi forward pass.
- Buộc model học **redundant representations** — không phụ thuộc vào chiều cụ thể nào.
- Đặc biệt quan trọng vì data sparse: nếu model "nhớ" embedding của một cặp (user, item) cụ thể, dropout ngăn cản việc này.

**Weight decay riêng cho embedding (2e-3):**
- Cao hơn các layer khác 20× — regularization mạnh hơn cho embedding vì dễ overfit với sparse data.

---

## 8. Optimizer và Learning Rate Schedule

```
AdamW: lr=2e-4, betas=(0.9, 0.999), eps=1e-8
├── Embedding params → weight_decay=2e-3
└── Other params    → weight_decay=1e-4

LR Schedule: 1-epoch warmup → Cosine decay (50 epochs max)
Gradient clipping: max_norm=1.0
```

**Warmup 1 epoch:** LR tăng tuyến tính 0 → 2e-4, tránh gradient unstable khi embedding vừa được khởi tạo ngẫu nhiên.

**Cosine decay:** LR giảm mượt theo hàm cosine. Trong thực tế chỉ chạy 10 epochs nên LR chỉ giảm nhẹ từ 2e-4 → ~1.84e-4 — schedule chưa phát huy tác dụng đầy đủ do early stopping quá sớm.

---

## 9. Phương Pháp Giảm Overfitting

| Kỹ thuật | Cấu hình | Mục tiêu |
|----------|----------|----------|
| Dropout (MLP) | `p=0.4` | Tránh co-adaptation giữa neurons |
| Embedding Dropout | `p=0.3` | Tránh memorize cặp user-item hiếm |
| BatchNorm1d | Sau mỗi Linear layer | Ổn định gradient, implicit regularization |
| L2 Weight Decay | emb=2e-3, linear=1e-4 | Penalize embedding/weights lớn |
| Gradient Clipping | max_norm=1.0 | Tránh gradient exploding |
| Early Stopping | patience=1, min_delta=1e-3 | Dừng khi không cải thiện |
| Dynamic Negative Sampling | 16 negatives mới/epoch | Tránh memorize negatives cố định |

**Quan sát:** Không có dấu hiệu overfitting — val loss nhỏ hơn train loss, val AUC cao hơn train AUC qua tất cả các epoch. Nguyên nhân: Dropout (0.4 + 0.3) chỉ active trong training nên training loss bị penalize nhiều hơn.

**Nhận xét về Early Stopping:** `patience=1` rất aggressive — dừng ngay nếu HR@10 không cải thiện ≥ 0.001 sau 1 epoch. Learning curve vẫn đang tăng đều → model có thể còn tiếp tục cải thiện nếu train thêm epoch.

---

## 10. Quá Trình Huấn Luyện — Các Epoch

Tổng thời gian: **36.44 phút** | Hardware: GPU (CUDA) | Epochs thực tế: **10 (early stop)**

### Learning Curve

| Epoch | Train Loss | Val Loss | Train AUC | Val AUC | Val HR@10 | Val NDCG@10 |
|-------|-----------|---------|-----------|---------|-----------|-------------|
| 0 | — | 0.2244 | — | 0.4950 | 0.0870 | 0.0396 |
| 1 | 0.2517 | 0.2236 | 0.4995 | 0.5293 | 0.1110 | 0.0489 |
| 2 | 0.2253 | 0.2225 | 0.5026 | 0.5716 | 0.1945 | 0.1008 |
| 3 | 0.2236 | 0.2207 | 0.5224 | 0.5910 | 0.2460 | 0.1501 |
| 4 | 0.2222 | 0.2187 | 0.5675 | 0.6095 | 0.2620 | 0.1730 |
| 5 | 0.2203 | 0.2162 | 0.6031 | 0.6364 | 0.3070 | 0.1953 |
| 6 | 0.2181 | 0.2131 | 0.6233 | 0.6515 | 0.3565 | 0.2172 |
| 7 | 0.2165 | 0.2102 | 0.6253 | 0.6626 | 0.3840 | 0.2308 |
| **8** ✓ | **0.2170** | **0.2083** | **0.6169** | **0.6718** | **0.4085** | **0.2420** |
| 9 | 0.2197 | 0.2076 | 0.6101 | 0.6772 | 0.4045 | 0.2425 |

**Epoch tốt nhất:** Epoch 8 (val_hr10=0.4085). Epoch 9 cho val_loss và val_auc cao hơn nhưng HR@10 thấp hơn một chút → early stopping kích hoạt, checkpoint epoch 8 được lưu.

### Quan Sát

1. **Val Loss < Train Loss:** Dropout chỉ active trong training → training prediction nhiễu hơn → train loss bị penalize. Đây là hiện tượng bình thường khi regularization mạnh.

2. **Val AUC > Train AUC:** Cùng nguyên nhân. Không có dấu hiệu overfitting.

3. **HR@10 tăng đều:** Từ 0.087 (epoch 0) → 0.409 (epoch 8). Learning curve chưa plateau — model chưa đạt trần khả năng, bị dừng sớm.

4. **Bước nhảy lớn epoch 2:** HR@10 từ 0.111 → 0.195 (tăng 75%). Giai đoạn model bắt đầu học được signal ranking quan trọng.

---

## 11. Kết Quả

### Ranking Metrics (validation set, 1 positive vs 99 negatives mỗi user)

| Mô hình | HR@10 | NDCG@10 |
|---------|-------|---------|
| NeuMF (epoch-level, 2000 samples) | 0.4085 | 0.2420 |
| **NeuMF (full eval, checkpoint epoch 8)** | **0.3934** | **0.2351** |
| **ItemPop Baseline** | **0.5414** | **0.3081** |


### Classification Metrics (val set, tỷ lệ 1:16 pos:neg)

| Metric | Giá trị |
|--------|---------|
| ROC AUC | 0.6718 |
| PR AUC (Average Precision) | 0.2013 |
| F1 Score (tại threshold tối ưu) | 0.2613 |
| Recall | 0.2949 |
| Precision | 0.2345 |
| Optimal threshold | 0.0858 |

**Threshold thấp (0.0858):** Sigmoid output của model thường rất nhỏ — phần lớn predictions nằm dưới 0.15. Nguyên nhân: training với 1:16 pos:neg khiến model bias về phía predict thấp; nhiều label positive cũng có giá trị nhỏ (0.197 cho `pv`).

---

## 12. Đánh Giá Dựa Trên Biểu Đồ (run `14-06-2026_13h26`)

### 12.1. Learning Curves — Model Chưa Hội Tụ

- **Loss:** Train loss giảm đều 0.25 → 0.21; val loss phẳng hơn (~0.22). Khoảng cách nhỏ — không overfitting.
- **ROC AUC:** Train và val tăng song song từ ~0.5 → ~0.67.
- **HR@10 và NDCG@10:** Cả hai đường **vẫn đang tăng ở epoch cuối (epoch 9)**, chưa plateau. Early stopping kích hoạt tại epoch 9 (best epoch 8) trước khi hội tụ — cần train thêm epoch.

### 12.2. Score Distribution — Separation Yếu

- Positive mean = **0.094**, Negative mean = **0.050** — chênh lệch chỉ 0.044.
- Hai phân phối **overlap lớn** trong vùng [0, 0.15]. Model học được tín hiệu nhưng chưa tách biệt rõ positive khỏi negative.

### 12.3. Confusion Matrix (threshold=0.086) — Recall Kém

| | Pred Negative | Pred Positive |
|---|---|---|
| **True Negative** | 40,000 ✓ | 2,582 ✗ |
| **True Positive** | 1,891 ✗ | 791 ✓ |

- Recall = 791 / (791 + 1891) = **29.5%** — model bỏ sót 70.5% positive.
- Precision = 791 / (791 + 2582) = **23.4%**.
- Dataset mất cân bằng nặng (~16:1 neg:pos) khiến model nghiêng về predict negative.

### 12.4. PR Curve (AP = 0.2013) — Hơn Random 3.4×

- AP = 0.2013 so với random baseline 0.059 → tốt hơn random **3.4 lần**.
- Đường cong dốc xuống nhanh: precision ~0.5 chỉ giữ được ở recall ~0.1. Điển hình với dữ liệu imbalanced nặng.

### 12.5. ROC Curve (AUC = 0.6718) — Mức Trung Bình

- AUC 0.67 trên đường random (0.5) nhưng xa mức tốt (≥ 0.75).
- Đường cong cong đều — model phân biệt nhất quán nhưng không mạnh.

### 12.6. Tổng Hợp

| Vấn đề | Bằng chứng từ biểu đồ |
|---|---|
| Train chưa đủ epoch | HR@10 và NDCG@10 vẫn tăng ở epoch 9 |
| Score separation yếu | Pos mean 0.094 vs Neg mean 0.050, overlap lớn |
| Recall thấp | 1891 FN vs chỉ 791 TP |
| Chưa đạt chất lượng | HR@10=0.39, NDCG@10=0.24 |

## 13. Tóm Tắt Kỹ Thuật

```
NeuMF (Neural Matrix Factorization)
├── Data: Taobao UserBehavior — 100M raw → 250K sampled
│   ├── Users: 187,767 | Items: 160,910 (sau k-core k=5)
│   ├── Label: tanh(sum_behavior_scores / 5) ∈ (0, 1)
│   └── Split: leave-last-out temporal
├── Architecture: GMF branch + MLP branch (4 embedding tables)
│   ├── Embedding dim: 16 (mỗi branch)
│   ├── MLP funnel: [32 → 64 → 32 → 16] + BN + ReLU + Dropout(0.4)
│   └── Final: Linear(32, 1) → Sigmoid
├── Loss: Binary Cross Entropy (BCE), 1:16 pos:neg
├── Optimizer: AdamW (lr=2e-4) + Warmup(1) + Cosine decay
├── Regularization: Dropout(0.4), EmbDropout(0.3), BatchNorm, L2, GradClip(1.0)
├── Training: 10 epochs / 36.4 min (GPU) — early stop epoch 9
└── Results (epoch 8 checkpoint):
    ├── HR@10=0.3934 | NDCG@10=0.2351
    ├── ROC AUC=0.6718 | PR AUC=0.2013
    └── ItemPop baseline: HR@10=0.5414 (vượt NeuMF +37.6%)
```
