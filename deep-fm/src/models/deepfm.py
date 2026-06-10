
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset


# class ExplicitDataset(Dataset):
#     """
#     Dataset cho Taobao — output của build.py.
#     Chỉ 2 fields: user, item.
#     Cột đầu vào: UserId, ItemId, Timestamp, Label
#     """
#     def __init__(self, source, user_map=None, item_map=None):
#         df = pd.read_csv(source) if isinstance(source, str) else source.copy()

#         # Implicit feedback: mọi row đều là positive
#         self.y = np.ones(len(df), dtype=np.float32)

#         # Train tự build map; val/test nhận map từ train
#         if user_map is None:
#             user_map = {u: i for i, u in enumerate(sorted(df["UserId"].unique()))}
#         if item_map is None:
#             item_map = {p: i for i, p in enumerate(sorted(df["ItemId"].unique()))}

#         self.user_map = user_map
#         self.item_map = item_map

#         # 2 fields: user, item
#         self.X = np.stack([
#             df["UserId"].map(user_map).fillna(0).astype(int).values,
#             df["ItemId"].map(item_map).fillna(0).astype(int).values,
#         ], axis=1)

#         self.field_dims = np.array([
#             len(user_map),  # users
#             len(item_map),  # items
#         ])

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.X[idx], dtype=torch.long),
#             torch.tensor(self.y[idx], dtype=torch.float32),
#         )


# class DeepFM(nn.Module):
#     def __init__(self, field_dims, embed_dim=10, mlp_dims=[64, 32, 16], dropout=0.5):
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

#         # FM part
#         square_of_sum = torch.sum(embed_x, dim=1) ** 2
#         sum_of_square = torch.sum(embed_x ** 2, dim=1)
#         fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

#         # Linear + Deep
#         linear = self.fc(x).sum(dim=1)
#         deep   = self.mlp(embed_x.view(embed_x.size(0), -1))

#         # Không sigmoid — dùng BCEWithLogitsLoss ở ngoài
#         return linear + fm + deep + self.bias
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ExplicitDataset(Dataset):
    """
    Dataset cho Taobao — output của build.py.
    3 fields: user, item, category.
    Cột đầu vào: UserId, ItemId, Timestamp, Label, CategoryId
    """
    def __init__(self, source, user_map=None, item_map=None, category_map=None):
        df = pd.read_csv(source) if isinstance(source, str) else source.copy()

        # Implicit feedback: mọi row đều là positive
        self.y = np.ones(len(df), dtype=np.float32)

        # Train tự build map; val/test nhận map từ train
        if user_map is None:
            user_map = {u: i for i, u in enumerate(sorted(df["UserId"].unique()))}
        if item_map is None:
            item_map = {p: i for i, p in enumerate(sorted(df["ItemId"].unique()))}
        if category_map is None:
            category_map = {c: i for i, c in enumerate(sorted(df["CategoryId"].unique()))}

        self.user_map     = user_map
        self.item_map     = item_map
        self.category_map = category_map

        # 3 fields: user, item, category
        self.X = np.stack([
            df["UserId"].map(user_map).fillna(0).astype(int).values,
            df["ItemId"].map(item_map).fillna(0).astype(int).values,
            df["CategoryId"].map(category_map).fillna(0).astype(int).values,
        ], axis=1)

        self.field_dims = np.array([
            len(user_map),      # users
            len(item_map),      # items
            len(category_map),  # categories
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=10, mlp_dims=[64, 32, 16], dropout=0.5):
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

        # FM part
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        # Linear + Deep
        linear = self.fc(x).sum(dim=1)
        deep   = self.mlp(embed_x.view(embed_x.size(0), -1))

        # Không sigmoid — dùng BCEWithLogitsLoss ở ngoài
        return linear + fm + deep + self.bias