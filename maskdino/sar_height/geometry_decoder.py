"""Three-layer multi-scale deformable SAR cross-attention decoder."""

from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn
from maskdino.modeling.pixel_decoder.position_encoding import PositionEmbeddingSine


class GeometryDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nheads=8,
        n_levels=4,
        n_points=4,
        dim_feedforward=1024,
        dropout=0.1,
        use_self_attention=False,
    ):
        super().__init__()
        self.use_self_attention = bool(use_self_attention)
        if self.use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                d_model, nheads, dropout=dropout, batch_first=True
            )
            self.self_dropout = nn.Dropout(dropout)
            self.self_norm = nn.LayerNorm(d_model)
        else:
            self.self_attention = None
            self.self_dropout = None
            self.self_norm = None

        self.cross_attention = MSDeformAttn(d_model, n_levels, nheads, n_points)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        query,
        reference_boxes,
        memory,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        if self.self_attention is not None:
            attended = self.self_attention(query, query, query, need_weights=False)[0]
            query = self.self_norm(query + self.self_dropout(attended))
        attended = self.cross_attention(
            query,
            reference_boxes,
            memory,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        query = self.cross_norm(query + self.cross_dropout(attended))
        update = self.linear2(self.ffn_dropout(self.activation(self.linear1(query))))
        return self.ffn_norm(query + self.output_dropout(update))


class GeometryDecoder(nn.Module):
    def __init__(
        self,
        num_layers=3,
        d_model=256,
        nheads=8,
        n_levels=4,
        n_points=4,
        dim_feedforward=1024,
        dropout=0.1,
        use_self_attention=False,
    ):
        super().__init__()
        if n_levels != 4:
            raise ValueError("the baseline requires four SAR FPN levels")
        self.n_levels = int(n_levels)
        self.layers = nn.ModuleList(
            [
                GeometryDecoderLayer(
                    d_model=d_model,
                    nheads=nheads,
                    n_levels=n_levels,
                    n_points=n_points,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_self_attention=use_self_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.level_embedding = nn.Parameter(torch.empty(n_levels, d_model))
        nn.init.normal_(self.level_embedding)

    @staticmethod
    def _level_mask(valid_mask, feature):
        if valid_mask is None:
            return torch.zeros(
                feature.shape[0], *feature.shape[-2:],
                dtype=torch.bool,
                device=feature.device,
            )
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[:, None]
        resized = F.interpolate(valid_mask.float(), size=feature.shape[-2:], mode="nearest")
        return ~(resized[:, 0] > 0.5)

    @staticmethod
    def _valid_ratio(mask):
        # Standard Deformable-DETR rectangular padding ratio. Irregular invalid
        # pixels are still handled by the flattened padding mask.
        height, width = mask.shape[-2:]
        valid_height = (~mask[:, :, 0]).sum(1).float() / max(height, 1)
        valid_width = (~mask[:, 0, :]).sum(1).float() / max(width, 1)
        return torch.stack([valid_width, valid_height], dim=-1)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        if len(multi_scale_features) != self.n_levels:
            raise ValueError(
                f"expected {self.n_levels} SAR levels, got {len(multi_scale_features)}"
            )
        query = query.float()
        reference_boxes = reference_boxes.float()
        memory_parts = []
        mask_parts = []
        spatial_shapes = []
        level_masks = []
        for level, feature in enumerate(multi_scale_features):
            feature = feature.float()
            mask = self._level_mask(valid_mask, feature)
            position = self.position_embedding(feature, mask)
            value = feature + position + self.level_embedding[level][None, :, None, None]
            memory_parts.append(value.flatten(2).transpose(1, 2))
            mask_parts.append(mask.flatten(1))
            spatial_shapes.append(feature.shape[-2:])
            level_masks.append(mask)

        memory = torch.cat(memory_parts, dim=1)
        padding_mask = torch.cat(mask_parts, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=query.device
        )
        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros(1),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            ]
        )
        valid_ratios = torch.stack(
            [self._valid_ratio(mask) for mask in level_masks], dim=1
        )
        # Our boxes are normalized to the padded image, exactly the coordinate
        # convention expected by MSDeformAttn, so they are repeated unchanged.
        references = reference_boxes[:, :, None, :].repeat(1, 1, self.n_levels, 1)

        intermediates = []
        for layer in self.layers:
            query = layer(
                query,
                references,
                memory,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
            intermediates.append(query)
        if return_details:
            return query, {
                "intermediate_queries": intermediates,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
                "padding_mask": padding_mask,
            }
        return query
