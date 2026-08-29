"""Low-dimensional optical/SAR projections used only for correlation."""

import torch
from torch import nn
from torch.nn import functional as F


class MatchingProjection(nn.Module):
    def __init__(self, optical_channels=256, sar_channels=256, match_dim=64):
        super().__init__()
        self.optical_projection = nn.Conv2d(optical_channels, match_dim, kernel_size=1)
        self.sar_projection = nn.Conv2d(sar_channels, match_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.optical_projection.weight)
        nn.init.zeros_(self.optical_projection.bias)
        nn.init.xavier_uniform_(self.sar_projection.weight)
        nn.init.zeros_(self.sar_projection.bias)

    def forward(self, optical_feature, sar_feature):
        if optical_feature.shape[-2:] != sar_feature.shape[-2:]:
            optical_feature = F.interpolate(
                optical_feature,
                size=sar_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        optical = F.normalize(self.optical_projection(optical_feature), p=2, dim=1)
        sar = F.normalize(self.sar_projection(sar_feature), p=2, dim=1)
        return optical, sar
