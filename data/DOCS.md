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
Step 4c: Normalize   — 2 * tanh(Label_raw / 5) → (0, 2)
Step 5:  Split       — Chia user: 90% train, 10% val (user-holdout)
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

#### Step 4c — Normalize Label (quan trọng)
```python
df["Label"] = 2.0 * np.tanh(df["Label"] / 5.0)
```

**Vấn đề:** Label gốc là tổng score (1..208). Phân bố rất lệch, cần đưa về khoảng chuẩn cho model học.

**Tại sao tanh thay vì sigmoid?**

![normalize_comparison](processs/eda/normalize_comparison.png)

| | Sigmoid | Tanh(x/5) |
|---|---|---|
| Label=1 → | 1.46 | **0.39** |
| Label=2 → | 1.76 | **0.76** |
| Label=3 → | 1.91 | **1.07** |
| Label=4 → | 1.96 | **1.33** |
| Label≥4 | sát nhau → khó phân biệt | giãn đều → dễ phân biệt |

Tanh dãn khoảng cách giữa các Label 1-4, giúp model phân biệt rõ ràng các mức độ tương tác.

#### Step 5 — Split by user (user-holdout)
```python
unique_users = pd.Series(df["UserId"].unique())
train_users, val_users = train_test_split(unique_users, test_size=0.1)
train = df[df["UserId"].isin(train_users)]
val = df[df["UserId"].isin(val_users)]
```

**Tại sao split user thay vì split row?**
- Split row: cùng 1 user có thể xuất hiện ở cả train và val → model thấy user đó rồi → đánh giá lạc quan (leakage)
- Split user: val chỉ gồm user **chưa từng thấy** → đánh giá thực tế hơn (cold-start scenario)

---

## 4. EDA — Đọc và hiểu biểu đồ

### 4.1. Label Distribution

![label_distribution](processs/eda/label_distribution.png)

**Cách đọc:**
- Trục X: Label sau normalize (0..2)
- Trục Y: Số lượng mẫu (log scale)
- Đường đứt: Giá trị trung bình

**Ý nghĩa:**
- **68%** dữ liệu ở bin 0.00-0.50 (chỉ pv)
- Phân bố lệch trái mạnh → model có xu hướng predict thấp
- Train vs Val: phân bố gần như giống nhau → split tốt

**Thông số quan trọng từ metadata.json:**
| Percentile | Label |
|---|---|
| 50% (median) | ~0.39 |
| 75% | ~0.76 |
| 90% | ~1.33 |
| 95% | ~1.52 |
| 99% | ~1.93 |

### 4.2. User & Item Statistics

![user_item_stats](processs/eda/user_item_stats.png)

**Cách đọc:**
- Biểu đồ trái: Phân phối số lượng item mỗi user tương tác
- Biểu đồ phải: Phân phối số lượng user mỗi item có

**Ý nghĩa:**
- Đa số user chỉ tương tác với **1 item** → cold-start là vấn đề chính
- Đa số item chỉ được **1 user** tương tác → item embedding khó học
- Sparsity: **99.997%** (ma trận user-item cực thưa)

### 4.3. Normalize Comparison

![normalize_comparison](processs/eda/normalize_comparison.png)

**Cách đọc:**
- Đường đỏ: `2 * sigmoid(x)` — bão hoà nhanh
- Đường xanh: `2 * tanh(x/5)` — dãn đều hơn
- Điểm đánh dấu: behavior cụ thể (pv=1, fav=2, cart=3, buy=4)
- Mũi tên giải thích: điểm yếu của sigmoid và ưu điểm của tanh

---

## 5. So sánh trước và sau cải tiến

| Vấn đề | Trước | Sau |
|---|---|---|
| **Item thưa** | 4.1M items, 70% chỉ 1 lần | Giữ nguyên (không filter — 57% mất dữ liệu nếu lọc) |
| **Khung giờ** | 8 tiếng cuối | 209 tiếng (9 ngày) |
| **Train/val overlap** | User xuất hiện cả 2 | Zero overlap |
| **Sampling** | Top-N timestamp | Stratified 4h-buckets |
| **Normalize** | sigmoid → bão hoà | tanh → dãn đều |

---

## 6. Cách chạy

```bash
# 1. Download dữ liệu (nếu chưa có)
python -m data.preprocess.load_data

# 2. Build pipeline
python -m data.preprocess.build --rows 250000

# 3. EDA
python -m data.preprocess.eda

# 4. Charts
python -m data.preprocess.charts
```

### Tham số `--rows`

| Giá trị | Số dòng output | Mục đích |
|---|---|---|
| 250000 | 250K | Phát triển, test nhanh |
| 500000 | 500K | Train thật (cần GPU mạnh) |
| (bỏ qua) | 75M (full) | Toàn bộ dữ liệu |

---

## 7. Kiến trúc output (train.csv / val.csv)

| Cột | Kiểu | Ví dụ | Mô tả |
|---|---|---|---|
| UserId | int32 | 1 | ID user gốc (không encode) |
| ItemId | int32 | 2268318 | ID item gốc (không encode) |
| Timestamp | int32 | 1512345600 | Thời điểm tương tác cuối (max) |
| Label | float32 | 0.3948 | Score đã normalize: 2*tanh(sum_scores/5) |

Mỗi dòng = 1 cặp `(user, item)` duy nhất. Label là điểm tương tác tổng hợp từ tất cả behavior của cặp đó.

---

## 8. File metadata.json

```json
{
  "n_rows_take": 250000,
  "n_rows_raw": 100150807,
  "n_rows_clean": 250000,
  "n_train": 225089,
  "n_val": 24911,
  "n_users": 205881,
  "n_items": 174241,
  "n_train_users": 185292,
  "n_val_users": 20589,
  "hour_bin_size": 4,
  "sampling": "stratified_by_hour",
  "split": "user_holdout",
  "label_raw_range": [1, 82],
  "label_norm": "2*tanh(x/5) -> (0, 2)",
  "label_percentiles": {
    "1%": 0.3948, "50%": 0.3948,
    "75%": 0.7599, "90%": 1.3281, "99%": 1.9281
  },
  "timestamp_span_hours": 209,
  "columns": ["UserId", "ItemId", "Timestamp", "Label"]
}
```

| Field | Ý nghĩa |
|---|---|
| n_rows_take | Số dòng yêu cầu (--rows) |
| n_rows_raw | Số dòng raw gốc |
| n_rows_clean | Số dòng output thực tế |
| n_train / n_val | Số dòng train/val |
| n_users / n_items | User/item unique |
| sampling | Phương pháp sampling |
| split | Phương pháp split |
| label_percentiles | Phân phối Label |
| timestamp_span_hours | Khoảng thời gian (giờ) |

---

*Generated by train-model preprocess pipeline.*
