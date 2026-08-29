"""End-to-end SAR geometry branch attached to MaskDINO queries."""

from typing import Optional

import torch
from torch import nn

from .geometry_decoder import GeometryDecoder
from .global_correlation import GlobalCorrelation
from .height_head import HeightHead
from .local_correlation import LocalCorrelation, shift_reference_boxes
from .matching_projection import MatchingProjection
from .phase_descriptor import IntensityNormalizer, PhaseDescriptorGenerator
from .sar_resnet_fpn import SARResNet18FPN
from .sar_stem import SARLightweightStem


class MaskDINOSARHeightBranch(nn.Module):
    """Fuse four-channel SAR into final MaskDINO instance queries."""

    def __init__(self, cfg, optical_channels=256):
        super().__init__()
        self.phase_descriptor_enabled = bool(cfg.MODEL.SAR.PHASE_DESCRIPTOR)
        self.global_enabled = bool(cfg.MODEL.ALIGN.GLOBAL_ENABLED)
        self.local_enabled = bool(cfg.MODEL.ALIGN.LOCAL_CORR_ENABLED)
        self.freeze_maskdino = bool(cfg.MODEL.MASKDINO.FREEZE)
        self.detach_reference = bool(cfg.MODEL.MASKDINO.DETACH_REFERENCE)
        hidden_dim = int(cfg.MODEL.GEOMETRY_DECODER.D_MODEL)

        self.phase_descriptor = PhaseDescriptorGenerator()
        self.intensity_normalizer = IntensityNormalizer(
            cfg.MODEL.SAR.INTENSITY_CLIP,
            cfg.MODEL.SAR.INTENSITY_MEAN,
            cfg.MODEL.SAR.INTENSITY_STD,
        )
        self.stem = SARLightweightStem(
            phase_channels=39 if self.phase_descriptor_enabled else 3,
            amp_enabled=cfg.MODEL.SAR.AMP_ENABLED,
            amp_gate_enabled=cfg.MODEL.SAR.AMP_GATE_ENABLED,
        )
        self.encoder = SARResNet18FPN(
            out_channels=hidden_dim,
            pretrained_weights=cfg.MODEL.SAR.RESNET18_WEIGHTS,
            freeze_batch_norm=cfg.MODEL.SAR.FREEZE_BATCH_NORM,
        )
        self.matching_projection = MatchingProjection(
            optical_channels=optical_channels,
            sar_channels=hidden_dim,
            match_dim=cfg.MODEL.SAR.MATCH_DIM,
        )
        self.global_correlation = GlobalCorrelation(
            search_radius=cfg.MODEL.ALIGN.GLOBAL_SEARCH_RADIUS,
            temperature=cfg.MODEL.ALIGN.GLOBAL_TEMPERATURE,
            matching_stride=cfg.MODEL.ALIGN.MATCHING_STRIDE,
        )
        self.local_correlation = LocalCorrelation(
            query_dim=hidden_dim,
            match_dim=cfg.MODEL.SAR.MATCH_DIM,
            search_radius=cfg.MODEL.ALIGN.LOCAL_SEARCH_RADIUS,
            temperature=cfg.MODEL.ALIGN.LOCAL_TEMPERATURE,
            matching_stride=cfg.MODEL.ALIGN.MATCHING_STRIDE,
        )
        self.geometry_query_projection = nn.Sequential(
            nn.Linear(cfg.MODEL.MaskDINO.HIDDEN_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.geometry_decoder = GeometryDecoder(
            num_layers=cfg.MODEL.GEOMETRY_DECODER.NUM_LAYERS,
            d_model=hidden_dim,
            nheads=cfg.MODEL.GEOMETRY_DECODER.NHEADS,
            n_levels=cfg.MODEL.GEOMETRY_DECODER.N_LEVELS,
            n_points=cfg.MODEL.GEOMETRY_DECODER.N_POINTS,
            dim_feedforward=cfg.MODEL.GEOMETRY_DECODER.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.GEOMETRY_DECODER.DROPOUT,
            use_self_attention=cfg.MODEL.GEOMETRY_DECODER.USE_SELF_ATTN,
        )
        self.height_head = HeightHead(
            hidden_dim=hidden_dim,
            dropout=cfg.MODEL.HEIGHT.DROPOUT,
            activation=cfg.MODEL.HEIGHT.ACTIVATION,
        )
        nn.init.xavier_uniform_(self.geometry_query_projection[0].weight)
        nn.init.zeros_(self.geometry_query_projection[0].bias)

    @staticmethod
    def _raw_phase(sin_phi, cos_phi, coherence, valid_mask):
        norm = torch.sqrt(sin_phi.square() + cos_phi.square() + 1e-6)
        phase = torch.cat(
            [sin_phi / norm, cos_phi / norm, coherence.clamp(0.0, 1.0)], dim=1
        )
        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[:, None]
            phase = phase * valid_mask.to(phase.dtype)
        return torch.nan_to_num(phase)

    def forward(
        self,
        decoder_query: torch.Tensor,
        reference_boxes: torch.Tensor,
        optical_feature_s8: torch.Tensor,
        sar_4ch: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        optical_valid_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ):
        if sar_4ch.ndim != 4 or sar_4ch.shape[1] != 4:
            raise ValueError(
                "SAR input must be [B,4,H,W] in "
                "[sin(phi), cos(phi), coherence, intensity_db] order"
            )
        if decoder_query.shape[:2] != reference_boxes.shape[:2]:
            raise ValueError("decoder query and reference box counts differ")

        if self.freeze_maskdino:
            decoder_query = decoder_query.detach()
            optical_feature_s8 = optical_feature_s8.detach()
        if self.freeze_maskdino or self.detach_reference:
            reference_boxes = reference_boxes.detach()
        reference_boxes = reference_boxes.clamp(0.0, 1.0)
        query = self.geometry_query_projection(decoder_query)

        sin_phi = sar_4ch[:, 0:1]
        cos_phi = sar_4ch[:, 1:2]
        coherence = sar_4ch[:, 2:3]
        intensity_db = sar_4ch[:, 3:4]
        if self.phase_descriptor_enabled:
            phase = self.phase_descriptor(
                sin_phi, cos_phi, coherence, sar_valid_mask
            )
        else:
            phase = self._raw_phase(
                sin_phi, cos_phi, coherence, sar_valid_mask
            )
        intensity = self.intensity_normalizer(intensity_db, sar_valid_mask)
        coherence_clean = torch.nan_to_num(coherence).clamp(0.0, 1.0)
        if sar_valid_mask is not None:
            mask = sar_valid_mask[:, None] if sar_valid_mask.ndim == 3 else sar_valid_mask
            coherence_clean = coherence_clean * mask.to(coherence_clean.dtype)
        amplitude = torch.cat([intensity, coherence_clean], dim=1)

        stem_feature, stem_details = self.stem(phase, amplitude)
        sar_features, stage_features = self.encoder(stem_feature)
        p2, p3, p4, p5 = sar_features
        optical_match, sar_match = self.matching_projection(optical_feature_s8, p3)
        image_size = tuple(sar_4ch.shape[-2:])

        if self.global_enabled:
            global_offset = self.global_correlation(
                optical_match,
                sar_match,
                image_size=image_size,
                optical_valid_mask=optical_valid_mask,
                sar_valid_mask=sar_valid_mask,
            )
        else:
            global_offset = query.new_zeros(query.shape[0], 2)

        if self.local_enabled:
            local_offsets = self.local_correlation(
                query,
                reference_boxes,
                optical_match,
                sar_match,
                global_offset,
                image_size=image_size,
                optical_valid_mask=optical_valid_mask,
                sar_valid_mask=sar_valid_mask,
            )
        else:
            local_offsets = query.new_zeros(*query.shape[:2], 2)

        sar_reference_boxes = shift_reference_boxes(
            reference_boxes, global_offset, local_offsets
        )
        geometry_query = self.geometry_decoder(
            query,
            sar_reference_boxes,
            sar_features,
            valid_mask=sar_valid_mask,
        )
        pred_heights = self.height_head(geometry_query)

        outputs = {
            "pred_heights": pred_heights,
            "global_sar_offset": global_offset,
            "local_sar_offsets": local_offsets,
            "sar_reference_boxes": sar_reference_boxes,
            "geometry_query": geometry_query,
        }
        if return_intermediates:
            outputs.update(
                {
                    "sar_fpn_features": (p2, p3, p4, p5),
                    "sar_stage_features": stage_features,
                    "sar_stem_details": stem_details,
                    "optical_matching_feature": optical_match,
                    "sar_matching_feature": sar_match,
                    "phase_descriptor": phase,
                }
            )
        return outputs
