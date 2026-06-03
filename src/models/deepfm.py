import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ExplicitDataset(Dataset):
    def __init__(self, source, user_map=None, item_map=None, time_map=None):
        # nhận cả path (str) lẫn DataFrame trực tiếp
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source.copy()

        # binarize rating
        self.y = (df["rating"] >= 4).astype(int).values

        # tạo mapping nếu chưa có (chỉ train mới tự build)
        if user_map is None:
            user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
        if item_map is None:
            item_map = {p: i for i, p in enumerate(df["parent_asin"].unique())}
        if time_map is None:
            time_map = {t: i for i, t in enumerate(df["timestamp"].unique())}

        self.user_map = user_map
        self.item_map = item_map
        self.time_map = time_map

        # ánh xạ sang số nguyên, unseen value → 0
        self.X = np.stack([
            df["user_id"].map(self.user_map).fillna(0).astype(int).values,
            df["parent_asin"].map(self.item_map).fillna(0).astype(int).values,
            df["timestamp"].map(self.time_map).fillna(0).astype(int).values,
        ], axis=1)

        # field_dims để DeepFM biết vocab size từng field
        self.field_dims = np.array([
            len(self.user_map),
            len(self.item_map),
            len(self.time_map),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=10, mlp_dims=[30, 20, 10], dropout=0.1):
        super().__init__()
        self.num_fields = len(field_dims)
        num_inputs = int(sum(field_dims))

        self.embedding = nn.Embedding(num_inputs, embed_dim)
        self.fc = nn.Embedding(num_inputs, 1)

        # offset để map index về đúng embedding space
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
        # cộng offset để map đúng vào embedding matrix
        x = x + self.offsets

        # FM part
        embed_x = self.embedding(x)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        # Linear part
        linear = self.fc(x).sum(dim=1)

        # Deep part
        deep = self.mlp(embed_x.view(embed_x.size(0), -1))
        self.bias = nn.Parameter(torch.zeros(1))

        return torch.sigmoid(linear + fm + deep + self.bias)