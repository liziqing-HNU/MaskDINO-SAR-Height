"""Tile-level masked correlation and differentiable soft-argmax offset."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GlobalCorrelation(nn.Module):
    def __init__(self, search_radius=8, temperature=0.1, matching_stride=8):
        super().__init__()
        if search_radius < 0:
            raise ValueError("search_radius must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.search_radius = int(search_radius)
        self.temperature = float(temperature)
        self.matching_stride = float(matching_stride)

    @staticmethod
    def _prepare_mask(mask, feature):
        if mask is None:
            return torch.ones(
                feature.shape[0], 1, *feature.shape[-2:],
                dtype=feature.dtype,
                device=feature.device,
            )
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.shape[-2:] != feature.shape[-2:]:
            mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        return (mask > 0.5).to(dtype=feature.dtype, device=feature.device)

    def forward(
        self,
        optical_feature: torch.Tensor,
        sar_feature: torch.Tensor,
        image_size: Tuple[int, int],
        optical_valid_mask: Optional[torch.Tensor] = None,
        sar_valid_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        if optical_feature.shape != sar_feature.shape:
            raise ValueError(
                "global matching features must have the same shape, got "
                f"{tuple(optical_feature.shape)} and {tuple(sar_feature.shape)}"
            )
        batch, channels, height, width = optical_feature.shape
        optical_valid = self._prepare_mask(optical_valid_mask, optical_feature)
        sar_valid = self._prepare_mask(sar_valid_mask, sar_feature)
        optical_feature = optical_feature * optical_valid
        sar_feature = sar_feature * sar_valid

        radius = self.search_radius
        # Grouped convolution evaluates every complete-image displacement in a
        # single operation. Output (R+dy, R+dx) compares optical(y,x) with
        # SAR(y+dy,x+dx), matching the sign convention in the specification.
        sar_grouped = sar_feature.reshape(1, batch * channels, height, width)
        optical_kernels = optical_feature.reshape(batch, channels, height, width)
        correlation = F.conv2d(
            sar_grouped,
            optical_kernels,
            padding=radius,
            groups=batch,
        ).reshape(batch, 2 * radius + 1, 2 * radius + 1)

        overlap = F.conv2d(
            sar_valid.reshape(1, batch, height, width),
            optical_valid.reshape(batch, 1, height, width),
            padding=radius,
            groups=batch,
        ).reshape(batch, 2 * radius + 1, 2 * radius + 1)
        valid_candidates = overlap > 0
        correlation = correlation / overlap.clamp_min(1.0)
        correlation = correlation.masked_fill(~valid_candidates, -torch.inf)
        probability = F.softmax(correlation.flatten(1) / self.temperature, dim=-1)

        coordinates = torch.arange(
            -radius,
            radius + 1,
            dtype=optical_feature.dtype,
            device=optical_feature.device,
        )
        dy, dx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        candidates = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
        offset_feature_pixels = probability @ candidates
        image_height, image_width = image_size
        scale = optical_feature.new_tensor(
            [self.matching_stride / image_width, self.matching_stride / image_height]
        )
        normalized_offset = offset_feature_pixels * scale

        if return_details:
            return normalized_offset, {
                "correlation": correlation,
                "probability": probability.reshape_as(correlation),
                "offset_feature_pixels": offset_feature_pixels,
                "overlap": overlap,
            }
        return normalized_offset
