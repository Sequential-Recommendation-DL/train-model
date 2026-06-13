# DOCS — Train Model Data Pipeline

## 1. Tổng quan

Project xây dựng pipeline xử lý dữ liệu hành vi người dùng (UserBehavior từ Taobao) thành bộ dữ liệu train/val cho các mô hình recommendation (DeepFM, NeuMF).

**Luồng dữ liệu tổng quát:**

```
Kaggle (raw CSV)
  → data/raw/UserBehavior.csv  (100M dòng)
  → data/preprocess/build.py    (xử lý)
  → data/processs/train.csv     (225K dòng)
  → data/processs/val.csv       (25K dòng)
```

---

## 2. Cấu trúc thư mục

```
data/
├── DOCS.md                  # Tài liệu này
├── raw/
│   └── UserBehavior.csv     # Raw data gốc 100M dòng (3.5GB)
├── preprocess/               # Code xử lý
│   ├── __init__.py
│   ├── config.py            # Cấu hình (paths, params)
│   ├── build.py             # Pipeline chính
│   ├── eda.py               # EDA (đọc từ output của build)
│   ├── charts.py            # Vẽ biểu đồ so sánh (sigmoid vs tanh)
│   ├── load_data.py          # Download từ Kaggle
│   ├── utils.py             # Helper (timer, ensure_dir)
│   └── README.md
└── processs/                # Output của build
    ├── train.csv            # Training set
    ├── val.csv              # Validation set
    ├── metadata.json        # Thông số của lần build gần nhất
    └── eda/                 # Biểu đồ EDA
        ├── label_distribution.png
        ├── normalize_comparison.png
        └── user_item_stats.png
```

---

## 3. Pipeline chi tiết (build.py)

### 3.1. Các bước

```
Step 1:  Load        — Đọc 100M dòng từ data/raw/UserBehavior.csv
Step 2:  Clean       — Xoá duplicate + lọc timestamp
Step 3:  Score       — Gán điểm behavior (pv=1, fav=2, cart=3, buy=4)
Step 4:  Groupby     — Gom (user, item) → sum Label, max Timestamp
Step 4b: Sample      — Stratified theo khung 4h (rải đều thời gian)
Step 4c: K-core      — Lọc user/item có ≥ k interactions (k=5)
Step 4d: Normalize   — tanh(Label_raw / 5) → (0, 1)
Step 5:  Split       — Leave-last-out (interaction cuối → val)
Step 6:  Save        — Ghi train.csv + val.csv
Step 7:  Metadata    — Ghi metadata.json
```

### 3.2. Giải thích từng bước

#### Step 1 — Load
```python
df = pd.read_csv(RAW_DATA, names=COLUMNS, dtype=DTYPES)
```
Đọc file CSV với schema:
| Cột | Kiểu | Mô tả |
|---|---|---|
| user_id | int32 | ID user |
| item_id | int32 | ID item |
| category_id | int32 | ID danh mục |
| behavior | category | pv / fav / cart / buy |
| timestamp | int32 | Unix timestamp |

#### Step 2 — Clean
- `drop_duplicates()`: Xoá dòng trùng lặp (thường rất ít)
- `timestamp.between(TIMESTAMP_RANGE)`: Giữ các dòng trong khoảng thời gian cấu hình

#### Step 3 — Score behavior
```python
score_map = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}
df["label"] = df["behavior"].map(score_map)
```
Chuyển behavior thành điểm số theo thang đo mức độ tương tác:
- **pv** (xem) = 1 — tương tác nhẹ nhất
- **fav** (thích) = 2 — quan tâm
- **cart** (giỏ hàng) = 3 — có ý định mua
- **buy** (mua) = 4 — chốt đơn

Ý tưởng: càng gần mua thì điểm càng cao, tạo ra Label có tính thứ bậc.

#### Step 4 — Group by (user, item)
```python
g = df.groupby(["user_id", "item_id"], as_index=False)
df = g.agg(Timestamp=("timestamp", "max"), Label=("label", "sum"))
```
Gom các behavior của cùng 1 user với **cùng 1 item**:
- `Label` = tổng điểm các behavior (vd: pv(1)+fav(2)+cart(3)+buy(4) = **10**)
- `Timestamp` = thời điểm xảy ra behavior **cuối cùng** (max)

Giảm từ 99M dòng raw → 75M cặp (user, item).

**Ví dụ:**
```
Raw (5 dòng):
UserA, ItemX, pv,  t=100
UserA, ItemX, fav, t=200
UserA, ItemX, cart, t=300
UserA, ItemX, buy, t=400
UserA, ItemY, pv,  t=500

Sau groupby (2 dòng):
UserA, ItemX, Timestamp=400, Label=1+2+3+4=10
UserA, ItemY, Timestamp=500, Label=1
```

#### Step 4b — Stratified sample by hour
```python
df["_hour_bin"] = df["Timestamp"] // (HOUR_BIN_SIZE * 3600)  # 4h buckets
freq = df["_hour_bin"].value_counts(normalize=True)
samples_per_bin = (freq * n_rows).round().astype(int)
```

Thay vì lấy top-N dòng gần nhất (chỉ được 8 tiếng cuối), ta **rải đều mẫu** theo khung 4h:
- Chia timestamp thành các bucket 4h
- Mỗi bucket lấy số dòng tỉ lệ với dung lượng của nó
- Kết quả: data trải dài **~209 tiếng (9 ngày)** thay vì chỉ 8 tiếng

Chỉ chạy khi `--rows` được truyền vào. Nếu không, giữ toàn bộ dữ liệu.

#### Step 4c — K-core filtering (mới)
```python
def _k_core_filter(df, k):
    while len(df) != prev_len:
        # Lọc user có < k interactions
        # Lọc item có < k interactions
        # Lặp cho đến khi converge
```

Giải quyết vấn đề **dữ liệu quá thưa** sau sampling:
- Đảm bảo mỗi user có ≥ k interactions → embedding có đủ signal
- Đảm bảo mỗi item có ≥ k users → item embedding có ý nghĩa
- Lặp vì xoá user/item thưa có thể làm user/item khác trở nên thưa

**Tại sao cần k-core?**

Từ EDA cũ (trước k-core):

![user_item_stats](processs/eda/user_item_stats.png)

- Đa số user chỉ tương tác với **1 item** → embedding không học được gì
- Đa số item chỉ được **1 user** tương tác → item embedding vô nghĩa
- Sparsity: **99.997%** — ma trận cực thưa

K-core filtering loại bỏ các user/item quá thưa, giúp model học embedding hiệu quả hơn.

#### Step 4d — Normalize Label to (0, 1)
```python
df["Label"] = np.tanh(df["Label"] / 5.0)
```

**Thay đổi so với v1:** `tanh(x/5)` thay vì `2*tanh(x/5)`

| | v1: 2·tanh(x/5) | v2: tanh(x/5) |
|---|---|---|
| **Range** | (0, 2) | (0, 1) |
| **BCE Loss** | ❌ Sai — BCE yêu cầu [0, 1] | ✅ Đúng |
| **Giãn label** | Giãn đều label 1-4 | Giãn đều label 1-4 |

Giữ nguyên ưu điểm giãn đều khoảng cách giữa các mức tương tác (so với sigmoid bão hoà), đồng thời đưa label về range [0, 1] hợp lệ cho BCE Loss.

| Label raw | tanh(x/5) |
|---|---|
| 1 (pv) | **0.197** |
| 2 (fav) | **0.380** |
| 3 (cart) | **0.537** |
| 4 (buy) | **0.665** |
| 10 (pv+fav+cart+buy) | **0.964** |

#### Step 5 — Leave-last-out split (thay user-holdout)
```python
df = df.sort_values(["UserId", "Timestamp"])
val_idx = df.groupby("UserId")["Timestamp"].idxmax()
val = df.loc[val_idx]
train = df.drop(val_idx)
```

**Tại sao thay user-holdout?**

| | User-holdout 90/10 (cũ) | Leave-last-out (mới) |
|---|---|---|
| **Val users** | 100% cold-start (user chưa thấy) | User đã có trong train |
| **Embedding** | Chưa học → predict vô nghĩa | Đã học → predict có ý nghĩa |
| **Temporal** | Không theo thời gian | Đúng thứ tự thời gian |

Với mỗi user, lấy **interaction cuối cùng** (theo timestamp) làm val:
- User xuất hiện ở **cả train và val** → embedding đã được train
- Đánh giá khả năng dự đoán **hành vi tiếp theo** — đúng bài toán recommendation
- User chỉ có 1 interaction sẽ chỉ nằm trong train (không vào val)

---

## 4. EDA — Đọc và hiểu biểu đồ

### 4.1. Label Distribution

![label_distribution](processs/eda/label_distribution.png)

**Cách đọc:**
- Trục X: Label sau normalize (0..1)
- Trục Y: Số lượng mẫu (log scale)
- Đường đứt: Giá trị trung bình

**Ý nghĩa:**
- Phân bố lệch trái mạnh (đa số là pv)
- Train vs Val: phân bố gần như giống nhau → split tốt

### 4.2. User & Item Statistics

![user_item_stats](processs/eda/user_item_stats.png)

**Cách đọc:**
- Biểu đồ trái: Phân phối số lượng item mỗi user tương tác
- Biểu đồ phải: Phân phối số lượng user mỗi item có

**Ý nghĩa (sau k-core):**
- Mỗi user/item đều có ≥ 5 interactions
- Sparsity giảm đáng kể so với raw data

### 4.3. Normalize Comparison

![normalize_comparison](processs/eda/normalize_comparison.png)

**Cách đọc:**
- Đường đỏ: `2 * sigmoid(x)` — bão hoà nhanh, label ≥4 gần như bằng nhau
- Đường xanh: `2 * tanh(x/5)` — dãn đều hơn
- Pipeline v2 dùng `tanh(x/5)` (không nhân 2) để range (0, 1) thay vì (0, 2)

---

## 5. So sánh pipeline v1 → v2

| Bước | v1 (cũ) | v2 (mới) | Lý do |
|---|---|---|---|
| **Sampling** | Stratified 4h | Stratified 4h (giữ) | Rải đều dữ liệu ~209h |
| **K-core** | Không có | k=5 (thêm mới) | Giảm sparsity, embedding có ý nghĩa |
| **Label** | 2·tanh(x/5) → (0, 2) | tanh(x/5) → (0, 1) | BCE cần label ∈ [0, 1] |
| **Split** | User-holdout 90/10 | Leave-last-out | Tránh cold-start trên val |

---

## 6. Cách chạy

```bash
# 1. Download dữ liệu (nếu chưa có)
python -m data.preprocess.load_data

# 2. Build pipeline
python -m data.preprocess.build --rows 250000   # 250K → stratified sample → k-core
python -m data.preprocess.build --rows 500000   # 500K → stratified sample → k-core
python -m data.preprocess.build                 # full → k-core (không sample)

# 3. EDA
python -m data.preprocess.eda

# 4. Charts
python -m data.preprocess.charts
```

### Tham số `--rows`

| Giá trị | Hành vi | Mục đích |
|---|---|---|
| 250000 | Stratified sample 250K → k-core | Phát triển, test nhanh |
| 500000 | Stratified sample 500K → k-core | Train vừa |
| (bỏ qua) | Full 75M → k-core | Toàn bộ dữ liệu |

**Lưu ý:** Số dòng output thực tế có thể ít hơn `--rows` vì k-core filtering sẽ loại bỏ user/item thưa sau sampling.

---

## 7. Kiến trúc output (train.csv / val.csv)

| Cột | Kiểu | Ví dụ | Mô tả |
|---|---|---|---|
| UserId | int32 | 1 | ID user gốc (không encode) |
| ItemId | int32 | 2268318 | ID item gốc (không encode) |
| Timestamp | int32 | 1512345600 | Thời điểm tương tác cuối (max) |
| Label | float32 | 0.1974 | Score đã normalize: tanh(sum_scores/5) |

**train.csv:** Tất cả interactions trừ interaction cuối của mỗi user.

**val.csv:** Interaction cuối cùng (theo timestamp) của mỗi user. User đều có mặt trong train → embedding đã được học.

---

## 8. File metadata.json

```json
{
  "n_rows_take": 250000,
  "n_rows_raw": 100150807,
  "n_rows_clean": 250000,
  "n_train": 224000,
  "n_val": 26000,
  "n_users": 26000,
  "n_items": 50000,
  "n_train_users": 26000,
  "n_val_users": 26000,
  "hour_bin_size": 4,
  "k_core": 5,
  "sampling": "stratified_by_hour + k_core",
  "split": "leave_last_out",
  "label_raw_range": [1, 46],
  "label_norm": "tanh(x/5) -> (0, 1)",
  "label_percentiles": {
    "1%": 0.1974, "50%": 0.1974,
    "75%": 0.3800, "90%": 0.6640, "99%": 0.9217
  },
  "timestamp_span_hours": 209,
  "random_seed": 42,
  "columns": ["UserId", "ItemId", "Timestamp", "Label"]
}
```

| Field | Ý nghĩa |
|---|---|
| n_rows_take | Số dòng yêu cầu (--rows) |
| n_rows_raw | Số dòng raw gốc |
| n_rows_clean | Số dòng output thực tế (train + val) |
| n_train / n_val | Số dòng train/val |
| n_users / n_items | User/item unique (sau k-core) |
| hour_bin_size | Khung giờ cho stratified sampling |
| k_core | Ngưỡng k-core filtering |
| sampling | Phương pháp sampling |
| split | Phương pháp split |
| label_percentiles | Phân phối Label |
| timestamp_span_hours | Khoảng thời gian (giờ) |

---

*Generated by train-model preprocess pipeline v2.*
