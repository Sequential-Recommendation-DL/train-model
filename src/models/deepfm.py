
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset


# class ExplicitDataset(Dataset):
#     def __init__(self, source, user_map=None, item_map=None, brand_map=None, category_map=None):
#         df = pd.read_csv(source) if isinstance(source, str) else source.copy()

#         self.y = (df["rating"] >= 4).astype(int).values

#         # chỉ train mới tự build map, val/test nhận map từ train
#         if user_map is None:
#             user_map = {u: i for i, u in enumerate(sorted(df["user_idx"].unique()))}
#         if item_map is None:
#             item_map = {p: i for i, p in enumerate(sorted(df["item_idx"].unique()))}
#         if brand_map is None:
#             brand_map = {b: i for i, b in enumerate(sorted(df["brand_idx"].unique()))}
#         if category_map is None:
#             category_map = {c: i for i, c in enumerate(sorted(df["category_idx"].unique()))}

#         self.user_map     = user_map
#         self.item_map     = item_map
#         self.brand_map    = brand_map
#         self.category_map = category_map

#         # map qua dict, unseen value → 0
#         self.X = np.stack([
#             df["user_idx"].map(user_map).fillna(0).astype(int).values,
#             df["item_idx"].map(item_map).fillna(0).astype(int).values,
#             df["brand_idx"].map(brand_map).fillna(0).astype(int).values,
#             df["category_idx"].map(category_map).fillna(0).astype(int).values,
#             df["price_idx"].fillna(0).astype(int).values,
#         ], axis=1)

#         # field_dims = vocab size từng field
#         self.field_dims = np.array([
#             len(user_map),
#             len(item_map),
#             len(brand_map),
#             len(category_map),
#             10,  # price buckets cố định 10 khoảng
#         ])

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.X[idx], dtype=torch.long),
#             torch.tensor(self.y[idx], dtype=torch.float32),
#         )


# class DeepFM(nn.Module):
#     def __init__(self, field_dims, embed_dim=10, mlp_dims=[64, 32, 16], dropout=0.3):
#         super().__init__()
#         self.num_fields = len(field_dims)
#         num_inputs = int(sum(field_dims))

#         self.embedding = nn.Embedding(num_inputs, embed_dim)
#         self.fc        = nn.Embedding(num_inputs, 1)
#         self.bias      = nn.Parameter(torch.zeros(1))

#         self.register_buffer(
#             "offsets",
#             torch.tensor([0] + list(np.cumsum(field_dims)[:-1]), dtype=torch.long)
#         )

#         input_dim = self.num_fields * embed_dim
#         layers = []
#         for dim in mlp_dims:
#             layers += [nn.Linear(input_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
#             input_dim = dim
#         layers.append(nn.Linear(input_dim, 1))
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x + self.offsets

#         embed_x = self.embedding(x)
#         square_of_sum = torch.sum(embed_x, dim=1) ** 2
#         sum_of_square = torch.sum(embed_x ** 2, dim=1)
#         fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

#         linear = self.fc(x).sum(dim=1)
#         deep   = self.mlp(embed_x.view(embed_x.size(0), -1))

#         return linear + fm + deep + self.bias

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ExplicitDataset(Dataset):
    """
    Dataset cho implicit feedback.
    Mọi row trong df đều là positive (label=1).
    Negative sampling được xử lý riêng trong pipeline.
    """
    def __init__(self, source, user_map=None, item_map=None, brand_map=None,
                 category_map=None):
        df = pd.read_csv(source) if isinstance(source, str) else source.copy()

        # implicit feedback: mọi row đều là positive
        self.y = np.ones(len(df), dtype=np.float32)

        # chỉ train mới tự build map, val/test nhận map từ train
        if user_map is None:
            user_map = {u: i for i, u in enumerate(sorted(df["user_idx"].unique()))}
        if item_map is None:
            item_map = {p: i for i, p in enumerate(sorted(df["item_idx"].unique()))}
        if brand_map is None:
            brand_map = {b: i for i, b in enumerate(sorted(df["brand_idx"].unique()))}
        if category_map is None:
            category_map = {c: i for i, c in enumerate(sorted(df["category_idx"].unique()))}

        self.user_map     = user_map
        self.item_map     = item_map
        self.brand_map    = brand_map
        self.category_map = category_map

        # [SỬA] 7 fields: user, item, brand, category, price, hour, dayofweek
        self.X = np.stack([
            df["user_idx"].map(user_map).fillna(0).astype(int).values,
            df["item_idx"].map(item_map).fillna(0).astype(int).values,
            df["brand_idx"].map(brand_map).fillna(0).astype(int).values,
            df["category_idx"].map(category_map).fillna(0).astype(int).values,
            df["price_idx"].fillna(0).astype(int).values,
            df["hour_idx"].fillna(0).astype(int).values,       # 0-23
            df["dayofweek_idx"].fillna(0).astype(int).values,  # 0-6
        ], axis=1)

        self.field_dims = np.array([
            len(user_map),
            len(item_map),
            len(brand_map),
            len(category_map),
            10,   # price buckets
            24,   # hour: 0-23
            7,    # dayofweek: 0-6
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=10, mlp_dims=[64, 32, 16], dropout=0.5):# ban đầu dropout=0.3
        super().__init__()
        self.num_fields = len(field_dims)
        num_inputs = int(sum(field_dims))

        self.embedding = nn.Embedding(num_inputs, embed_dim)
        self.fc        = nn.Embedding(num_inputs, 1)
        self.bias      = nn.Parameter(torch.zeros(1))

        self.register_buffer(
            "offsets",
            torch.tensor([0] + list(np.cumsum(field_dims)[:-1]), dtype=torch.long)
        )

        input_dim = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [nn.Linear(input_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.offsets

        embed_x = self.embedding(x)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        linear = self.fc(x).sum(dim=1)
        deep   = self.mlp(embed_x.view(embed_x.size(0), -1))

        # Không sigmoid ở đây — dùng BCEWithLogitsLoss
        return linear + fm + deep + self.bias