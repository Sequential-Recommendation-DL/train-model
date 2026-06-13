# Preprocess

Pipeline chuyển raw `UserBehavior.csv` → `train.csv` + `val.csv` cho các mô hình recommendation.

## Output

Mỗi file chỉ có **4 cột**:

| Cột | Gốc | Mô tả |
|---|---|---|
| `UserId` | `user_id` | ID người dùng |
| `ItemId` | `item_id` | ID sản phẩm |
| `Timestamp` | `timestamp` | Thời điểm tương tác cuối (max) |
| `Label` | `behavior` | Score normalize: tanh(sum_scores/5) ∈ (0, 1) |

Mỗi dòng là 1 cặp `(user, item)` — gộp các behavior của user đó với item đó lại, **Label = tanh(sum(score)/5)**. Timestamp lấy giá trị lớn nhất (gần đây nhất).

## Pipeline

```
Load → Dedup + filter timestamp → Score → Groupby → Stratified sample → Normalize → Temporal split → Save
```

### Thay đổi v2 so với v1

| Bước | v1 | v2 | Lý do |
|---|---|---|---|
| Label | 2·tanh(x/5) → (0,2) | tanh(x/5) → (0,1) | BCE cần [0,1] |
| Split | User-holdout 90/10 | Temporal 90/10 | Dự đoán tương lai, tránh cold-start |

## Chạy

```bash
python -m data.preprocess.build                 # full data
python -m data.preprocess.build --rows 250000   # 250K stratified sample
python -m data.preprocess.build --rows 500000   # 500K stratified sample
```

Output lưu ở `data/processs/`.

## EDA

Phân tích từ output của build (train.csv + val.csv). Chạy sau khi build:

```bash
python -m data.preprocess.eda
```

Biểu đồ lưu ở `data/processs/eda/`:

| Biểu đồ | Mô tả |
|---|---|
| ![label_distribution](../processs/eda/label_distribution.png) | Label distribution train vs val |
| ![user_item_stats](../processs/eda/user_item_stats.png) | Interactions/user + Interactions/item |
