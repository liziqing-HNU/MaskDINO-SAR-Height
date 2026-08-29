# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging
import math

import cv2
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if not is_train:
        return [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                "choice",
            )
        ]
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        height_on,
        recompute_boxes_from_masks,
        sar_enabled,
        sar_channel_order,
        synthetic_shift_supervision,
        synthetic_shift_probability,
        synthetic_shift_max_pixels,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.height_on = height_on
        self.recompute_boxes_from_masks = recompute_boxes_from_masks
        self.sar_enabled = sar_enabled
        self.sar_channel_order = tuple(int(index) for index in sar_channel_order)
        self.synthetic_shift_supervision = synthetic_shift_supervision
        self.synthetic_shift_probability = float(synthetic_shift_probability)
        self.synthetic_shift_max_pixels = int(synthetic_shift_max_pixels)
        if sorted(self.sar_channel_order) != [0, 1, 2, 3]:
            raise ValueError(
                "INPUT.SAR_CHANNEL_ORDER must be a permutation of [0,1,2,3]"
            )
        logging.getLogger(__name__).info(
            "Recompute instance boxes from masks: {}".format(
                self.recompute_boxes_from_masks
            )
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "height_on": cfg.MODEL.MaskDINO.HEIGHT_ON or cfg.MODEL.SAR.ENABLED,
            "recompute_boxes_from_masks": (
                cfg.INPUT.RECOMPUTE_BOXES_FROM_MASKS
            ),
            "sar_enabled": cfg.MODEL.SAR.ENABLED,
            "sar_channel_order": cfg.INPUT.SAR_CHANNEL_ORDER,
            "synthetic_shift_supervision": (
                cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_SUPERVISION
            ),
            "synthetic_shift_probability": (
                cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_PROBABILITY
            ),
            "synthetic_shift_max_pixels": (
                cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_MAX_PIXELS
            ),
        }
        return ret

    @staticmethod
    def _translate(array, dx, dy, fill_value=0):
        """Translate an HWC/HW array without circular wrapping."""
        output = np.full_like(array, fill_value)
        height, width = array.shape[:2]
        source_y0 = max(-dy, 0)
        source_y1 = height - max(dy, 0)
        source_x0 = max(-dx, 0)
        source_x1 = width - max(dx, 0)
        target_y0 = max(dy, 0)
        target_y1 = height - max(-dy, 0)
        target_x0 = max(dx, 0)
        target_x1 = width - max(-dx, 0)
        if source_y1 > source_y0 and source_x1 > source_x0:
            output[target_y0:target_y1, target_x0:target_x1] = array[
                source_y0:source_y1, source_x0:source_x1
            ]
        return output

    def _read_sar(self, file_name):
        sar = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        if sar is None:
            raise OSError(f"failed to read SAR GeoTIFF: {file_name}")
        if sar.ndim != 3 or sar.shape[2] != 4:
            raise ValueError(
                f"expected four SAR channels in {file_name}, got {sar.shape}"
            )
        # OpenCV reverses the first three GeoTIFF bands for this dataset.
        sar = sar[..., self.sar_channel_order].astype(np.float32, copy=False)
        valid = np.isfinite(sar).all(axis=2)
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)
        return sar, valid

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if self.sar_enabled:
            if "sar_file_name" not in dataset_dict:
                raise KeyError(
                    "MODEL.SAR.ENABLED=True but dataset record has no sar_file_name"
                )
            sar_image, sar_valid_mask = self._read_sar(
                dataset_dict["sar_file_name"]
            )
            if sar_image.shape[:2] != image.shape[:2]:
                raise ValueError(
                    "paired optical/SAR dimensions differ for "
                    f"{dataset_dict['file_name']} and {dataset_dict['sar_file_name']}"
                )

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        if self.sar_enabled:
            # The exact same geometric transform is applied to raw four-channel
            # SAR before any phase descriptor is generated.
            sar_image = transforms.apply_image(sar_image)
            sar_valid_mask = transforms.apply_segmentation(
                sar_valid_mask.astype(np.uint8)
            ).astype(bool)

            shift_x = 0
            shift_y = 0
            if (
                self.is_train
                and self.synthetic_shift_supervision
                and self.synthetic_shift_max_pixels > 0
                and np.random.random() < self.synthetic_shift_probability
            ):
                limit = self.synthetic_shift_max_pixels
                shift_x = int(np.random.randint(-limit, limit + 1))
                shift_y = int(np.random.randint(-limit, limit + 1))
                sar_image = self._translate(sar_image, shift_x, shift_y, 0.0)
                sar_valid_mask = self._translate(
                    sar_valid_mask, shift_x, shift_y, False
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        if self.sar_enabled:
            dataset_dict["sar_image"] = torch.as_tensor(
                np.ascontiguousarray(sar_image.transpose(2, 0, 1)),
                dtype=torch.float32,
            )
            dataset_dict["sar_valid_mask"] = torch.as_tensor(
                np.ascontiguousarray(sar_valid_mask), dtype=torch.bool
            )
            # Pixel displacement is retained so normalization can use the final
            # batch padding dimensions inside the model.
            dataset_dict["synthetic_sar_shift_pixels"] = torch.tensor(
                [shift_x, shift_y], dtype=torch.float32
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # BitMasks support both polygons and COCO RLEs (used by footprints
            # with holes in SP6).
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format="bitmask"
            )
            if self.height_on:
                heights = []
                height_valid = []
                for anno in annos:
                    height = anno.get("height")
                    valid = (
                        bool(anno.get("height_valid", height is not None))
                        and height is not None
                        and math.isfinite(float(height))
                        and float(height) >= 0
                    )
                    heights.append(float(height) if valid else 0.0)
                    height_valid.append(valid)
                instances.gt_heights = torch.as_tensor(heights, dtype=torch.float32)
                instances.gt_height_valid = torch.as_tensor(
                    height_valid, dtype=torch.bool
                )
            if not instances.has('gt_masks'):  # this is to avoid empty annotation
                instances.gt_masks = BitMasks(
                    torch.zeros(
                        (0, image_shape[0], image_shape[1]), dtype=torch.bool
                    )
                )
            # Standard instance segmentation uses the tight mask bounds as its
            # box target after crop/resize. BONAI deliberately pairs a broader
            # building bbox with a roof mask, so its transformed annotation box
            # must be preserved instead.
            if self.recompute_boxes_from_masks:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # MaskDINO consumes an N x H x W tensor rather than a BitMasks wrapper.
            if instances.has("gt_masks"):
                instances.gt_masks = instances.gt_masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
