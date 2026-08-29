"""Lightweight phase/amplitude stems and gated early fusion."""

import torch
from torch import nn


def _conv_gn_relu(in_channels, out_channels, kernel_size=3, stride=1):
    padding = kernel_size // 2
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(inplace=True),
    )


class SARLightweightStem(nn.Module):
    """Two shallow stems followed by optional amplitude gating."""

    def __init__(
        self,
        phase_channels: int = 39,
        amp_enabled: bool = True,
        amp_gate_enabled: bool = True,
    ):
        super().__init__()
        self.amp_enabled = bool(amp_enabled)
        self.amp_gate_enabled = bool(amp_gate_enabled)

        self.phase_stem = nn.Sequential(
            _conv_gn_relu(phase_channels, 32, kernel_size=3, stride=2),
            _conv_gn_relu(32, 64, kernel_size=3, stride=2),
        )
        if self.amp_enabled:
            self.amp_stem = nn.Sequential(
                _conv_gn_relu(2, 16, kernel_size=3, stride=2),
                _conv_gn_relu(16, 32, kernel_size=3, stride=2),
                nn.Conv2d(32, 64, kernel_size=1),
            )
            self.amp_gate = nn.Conv2d(128, 64, kernel_size=1)
        else:
            self.amp_stem = None
            self.amp_gate = None

        self.fusion_conv = _conv_gn_relu(64, 64, kernel_size=3, stride=1)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, phase: torch.Tensor, amplitude: torch.Tensor = None):
        phase_feature = self.phase_stem(phase)
        amp_feature = None
        amp_gate = None
        if self.amp_enabled:
            if amplitude is None:
                raise ValueError("amplitude input is required when AMP_ENABLED=True")
            amp_feature = self.amp_stem(amplitude)
            if self.amp_gate_enabled:
                amp_gate = torch.sigmoid(
                    self.amp_gate(torch.cat([phase_feature, amp_feature], dim=1))
                )
                fused = phase_feature + amp_gate * amp_feature
            else:
                fused = phase_feature + amp_feature
        else:
            fused = phase_feature
        return self.fusion_conv(fused), {
            "phase_feature": phase_feature,
            "amplitude_feature": amp_feature,
            "amplitude_gate": amp_gate,
        }
