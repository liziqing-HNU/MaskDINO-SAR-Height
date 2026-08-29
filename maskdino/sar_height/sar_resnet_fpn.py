"""One shared ResNet18 body and a four-level SAR feature pyramid."""

import logging
import os
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from detectron2.layers import FrozenBatchNorm2d


LOGGER = logging.getLogger(__name__)


class SARResNet18FPN(nn.Module):
    """Run a stride-4 stem feature through ResNet18 layer1-layer4 and FPN."""

    def __init__(
        self,
        out_channels: int = 256,
        pretrained_weights: str = "",
        freeze_batch_norm: bool = True,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if pretrained_weights:
            self._load_imagenet_weights(pretrained_weights)
        if freeze_batch_norm:
            FrozenBatchNorm2d.convert_frozen_batchnorm(self)

        in_channels = (64, 128, 256, 512)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(channels, out_channels, kernel_size=1) for channels in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
                for _ in in_channels
            ]
        )
        self._reset_fpn_parameters()

    def _load_imagenet_weights(self, source: str):
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            state_dict = torch.hub.load_state_dict_from_url(
                source, map_location="cpu", progress=True, check_hash=True
            )
        else:
            source = os.path.expanduser(source)
            if not os.path.isfile(source):
                raise FileNotFoundError(f"SAR ResNet18 weights do not exist: {source}")
            state_dict = torch.load(source, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

        compatible = {}
        for key, value in state_dict.items():
            key = key.removeprefix("module.")
            if key.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
                compatible[key] = value
        incompatible = self.load_state_dict(compatible, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key in compatible]
        if unexpected:
            raise RuntimeError(f"unexpected ResNet18 keys: {unexpected}")
        LOGGER.info(
            "Loaded %d ImageNet ResNet18 layer1-layer4 tensors from %s",
            len(compatible),
            source,
        )

    def _reset_fpn_parameters(self):
        for module in list(self.lateral_convs) + list(self.output_convs):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, stem_feature: torch.Tensor):
        c2 = self.layer1(stem_feature)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        stages = (c2, c3, c4, c5)

        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, stages)]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        pyramid = [
            conv(feature) for conv, feature in zip(self.output_convs, laterals)
        ]
        return pyramid, {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }
