"""Query-conditioned local optical/SAR correlation."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def shift_reference_boxes(
    optical_reference_boxes: torch.Tensor,
    global_offset: torch.Tensor,
    local_offsets: torch.Tensor,
    clamp_center: bool = True,
) -> torch.Tensor:
    """Shift only ``(cx, cy)`` while retaining optical box width and height."""
    if optical_reference_boxes.ndim != 3 or optical_reference_boxes.shape[-1] != 4:
        raise ValueError("optical_reference_boxes must have shape [B,N,4]")
    if global_offset.shape != (optical_reference_boxes.shape[0], 2):
        raise ValueError("global_offset must have shape [B,2]")
    if local_offsets.shape != (*optical_reference_boxes.shape[:2], 2):
        raise ValueError("local_offsets must have shape [B,N,2]")
    shifted = optical_reference_boxes.clone()
    center = shifted[..., :2] + global_offset[:, None, :] + local_offsets
    if clamp_center:
        center = center.clamp(0.0, 1.0)
    shifted[..., :2] = center
    return shifted


class LocalCorrelation(nn.Module):
    def __init__(
        self,
        query_dim=256,
        match_dim=64,
        search_radius=4,
        temperature=0.1,
        matching_stride=8,
    ):
        super().__init__()
        if search_radius < 0:
            raise ValueError("search_radius must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.query_projection = nn.Linear(query_dim, match_dim)
        self.search_radius = int(search_radius)
        self.temperature = float(temperature)
        self.matching_stride = float(matching_stride)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)

    @staticmethod
    def _sample(feature: torch.Tensor, coordinates: torch.Tensor):
        # coordinates are normalized image coordinates in [0,1]. grid_sample
        # consumes [-1,1] and treats out-of-range locations as zero.
        grid = coordinates.mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled

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
        query: torch.Tensor,
        optical_reference_boxes: torch.Tensor,
        optical_feature: torch.Tensor,
        sar_feature: torch.Tensor,
        global_offset: torch.Tensor,
        image_size: Tuple[int, int],
        optical_valid_mask: Optional[torch.Tensor] = None,
        sar_valid_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        batch, num_queries, _ = query.shape
        if optical_reference_boxes.shape != (batch, num_queries, 4):
            raise ValueError("reference/query shapes are inconsistent")
        if optical_feature.shape != sar_feature.shape:
            raise ValueError("local matching features must have identical shapes")

        optical_valid = self._prepare_mask(optical_valid_mask, optical_feature)
        sar_valid = self._prepare_mask(sar_valid_mask, sar_feature)
        optical_feature = optical_feature * optical_valid
        sar_feature = sar_feature * sar_valid

        optical_center = optical_reference_boxes[..., :2]
        optical_sample_grid = optical_center[:, :, None, :]
        optical_sample = self._sample(optical_feature, optical_sample_grid)
        optical_sample = optical_sample.squeeze(-1).transpose(1, 2)
        template = F.normalize(
            self.query_projection(query) + optical_sample,
            p=2,
            dim=-1,
        )

        radius = self.search_radius
        offsets = torch.arange(
            -radius,
            radius + 1,
            dtype=query.dtype,
            device=query.device,
        )
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_pixels = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
        image_height, image_width = image_size
        normalization = query.new_tensor(
            [self.matching_stride / image_width, self.matching_stride / image_height]
        )
        candidate_offsets = offset_pixels * normalization

        search_center = optical_center + global_offset[:, None, :]
        candidate_coordinates = (
            search_center[:, :, None, :] + candidate_offsets[None, None, :, :]
        )
        sampled_sar = self._sample(sar_feature, candidate_coordinates)
        # [B,C,N,K] -> [B,N,K,C]
        sampled_sar = sampled_sar.permute(0, 2, 3, 1)
        logits = torch.einsum("bnc,bnkc->bnk", template, sampled_sar)

        sampled_valid = self._sample(sar_valid, candidate_coordinates).squeeze(1)
        inside = ((candidate_coordinates >= 0.0) & (candidate_coordinates <= 1.0)).all(-1)
        candidate_valid = inside & (sampled_valid > 0.5)

        # A manual masked softmax avoids NaNs for a query whose complete search
        # window lies outside valid SAR data; that query receives zero offset.
        masked_logits = (logits / self.temperature).masked_fill(
            ~candidate_valid, -torch.inf
        )
        max_logits = masked_logits.amax(dim=-1, keepdim=True)
        max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
        probability = torch.exp(masked_logits - max_logits) * candidate_valid.to(logits.dtype)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        local_offset = torch.einsum("bnk,kc->bnc", probability, candidate_offsets)

        if return_details:
            return local_offset, {
                "probability": probability,
                "candidate_valid": candidate_valid,
                "candidate_coordinates": candidate_coordinates,
                "offset_feature_pixels": torch.einsum(
                    "bnk,kc->bnc", probability, offset_pixels
                ),
            }
        return local_offset
