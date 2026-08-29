"""Parameter-free SAR phase descriptors and fixed intensity normalization."""

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class PhaseDescriptorGenerator(nn.Module):
    """Build the 39-channel wrapped-phase descriptor from three SAR bands.

    Neighbours outside the image or outside ``valid_mask`` are explicitly
    zeroed. No circular wrapping (``torch.roll``) is used.
    """

    OFFSETS = tuple(
        (dx, dy)
        for radius in (1, 2, 4)
        for dx, dy in (
            (radius, 0),
            (0, radius),
            (radius, radius),
            (radius, -radius),
        )
    )

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.padding = max(max(abs(dx), abs(dy)) for dx, dy in self.OFFSETS)

    @staticmethod
    def _as_mask(valid_mask: Optional[torch.Tensor], reference: torch.Tensor):
        if valid_mask is None:
            return torch.ones_like(reference, dtype=torch.bool)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[:, None]
        if valid_mask.shape != reference.shape:
            raise ValueError(
                "valid_mask must have shape [B,H,W] or [B,1,H,W], got "
                f"{tuple(valid_mask.shape)} for {tuple(reference.shape)}"
            )
        return valid_mask.to(device=reference.device, dtype=torch.bool)

    def _neighbour(self, tensor: torch.Tensor, dx: int, dy: int):
        pad = self.padding
        padded = F.pad(tensor, (pad, pad, pad, pad), mode="constant", value=0.0)
        height, width = tensor.shape[-2:]
        y0 = pad + dy
        x0 = pad + dx
        return padded[..., y0 : y0 + height, x0 : x0 + width]

    def forward(
        self,
        sin_phi: torch.Tensor,
        cos_phi: torch.Tensor,
        coherence: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not (sin_phi.shape == cos_phi.shape == coherence.shape):
            raise ValueError("sin_phi, cos_phi and coherence must have identical shapes")
        if sin_phi.ndim != 4 or sin_phi.shape[1] != 1:
            raise ValueError("phase inputs must have shape [B,1,H,W]")

        valid = self._as_mask(valid_mask, sin_phi)
        finite = torch.isfinite(sin_phi) & torch.isfinite(cos_phi) & torch.isfinite(coherence)
        valid = valid & finite

        sin_phi = torch.nan_to_num(sin_phi, nan=0.0, posinf=0.0, neginf=0.0)
        cos_phi = torch.nan_to_num(cos_phi, nan=0.0, posinf=0.0, neginf=0.0)
        coherence = torch.nan_to_num(coherence, nan=0.0, posinf=0.0, neginf=0.0)

        # Bilinear geometric augmentation perturbs the unit phase vector. Restore
        # it before constructing pairwise descriptors.
        phase_norm = torch.sqrt(sin_phi.square() + cos_phi.square() + self.eps)
        sin_phi = sin_phi / phase_norm
        cos_phi = cos_phi / phase_norm
        coherence = coherence.clamp(0.0, 1.0)

        zeros = torch.zeros_like(sin_phi)
        sin_phi = torch.where(valid, sin_phi, zeros)
        cos_phi = torch.where(valid, cos_phi, zeros)
        coherence = torch.where(valid, coherence, zeros)

        descriptors = [sin_phi, cos_phi, coherence]
        valid_float = valid.to(dtype=sin_phi.dtype)
        for dx, dy in self.OFFSETS:
            sin_neighbour = self._neighbour(sin_phi, dx, dy)
            cos_neighbour = self._neighbour(cos_phi, dx, dy)
            coherence_neighbour = self._neighbour(coherence, dx, dy)
            neighbour_valid = self._neighbour(valid_float, dx, dy) > 0.5
            pair_valid = valid & neighbour_valid

            ds = sin_phi * cos_neighbour - cos_phi * sin_neighbour
            dc = cos_phi * cos_neighbour + sin_phi * sin_neighbour
            pair_coherence = torch.sqrt(
                (coherence * coherence_neighbour).clamp_min(0.0)
            )
            descriptors.extend(
                [
                    torch.where(pair_valid, ds, zeros),
                    torch.where(pair_valid, dc, zeros),
                    torch.where(pair_valid, pair_coherence, zeros),
                ]
            )

        output = torch.cat(descriptors, dim=1)
        if output.shape[1] != 39:
            raise RuntimeError(f"expected 39 phase channels, got {output.shape[1]}")
        return output


class IntensityNormalizer(nn.Module):
    """Apply fixed train-set clipping and z-score normalization to intensity dB."""

    def __init__(
        self,
        clip: Sequence[float],
        mean: float,
        std: float,
        eps: float = 1e-6,
    ):
        super().__init__()
        if len(clip) != 2 or clip[0] >= clip[1]:
            raise ValueError(f"clip must be [minimum, maximum], got {clip}")
        if std <= 0:
            raise ValueError("intensity standard deviation must be positive")
        self.register_buffer("clip_min", torch.tensor(float(clip[0])), persistent=True)
        self.register_buffer("clip_max", torch.tensor(float(clip[1])), persistent=True)
        self.register_buffer("mean", torch.tensor(float(mean)), persistent=True)
        self.register_buffer("std", torch.tensor(float(std)), persistent=True)
        self.eps = float(eps)

    def forward(
        self,
        intensity_db: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        intensity_db = torch.nan_to_num(
            intensity_db,
            nan=float(self.mean),
            posinf=float(self.clip_max),
            neginf=float(self.clip_min),
        )
        normalized = (
            intensity_db.clamp(float(self.clip_min), float(self.clip_max)) - self.mean
        ) / (self.std + self.eps)
        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[:, None]
            normalized = normalized * valid_mask.to(normalized.dtype)
        return normalized
