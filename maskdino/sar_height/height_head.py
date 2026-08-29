"""Direct building-height prediction from multimodal geometry queries."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class HeightHead(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1, activation="softplus"):
        super().__init__()
        self.activation = activation.lower()
        if self.activation not in {"softplus", "relu", "identity"}:
            raise ValueError(f"unsupported height activation: {activation}")
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # A small non-zero projection lets the first height step propagate all
        # the way into deformable attention and the SAR encoder.
        nn.init.normal_(self.linear2.weight, std=0.01)
        # Softplus(approximately 2.95) ~= 3 m provides positive, non-saturated
        # initial predictions while the multimodal branch starts from scratch.
        nn.init.constant_(self.linear2.bias, math.log(math.expm1(3.0)))

    def forward(self, query):
        height = self.linear2(
            self.dropout(F.relu(self.linear1(self.norm(query)), inplace=True))
        ).squeeze(-1)
        if self.activation == "softplus":
            return F.softplus(height)
        if self.activation == "relu":
            return F.relu(height)
        return height
