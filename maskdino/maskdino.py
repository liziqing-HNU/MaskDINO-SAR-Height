# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .sar_height import MaskDINOSARHeightBranch
from .utils import box_ops


@META_ARCH_REGISTRY.register()
class MaskDINO(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        height_on: bool = False,
        height_scale: float = 10.0,
        sar_enabled: bool = False,
        sar_height_branch: nn.Module = None,
        freeze_maskdino: bool = False,
        synthetic_shift_supervision: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss
        self.height_on = height_on
        self.height_scale = height_scale
        self.sar_enabled = sar_enabled
        self.sar_height_branch = sar_height_branch
        self.freeze_maskdino = freeze_maskdino
        self.synthetic_shift_supervision = synthetic_shift_supervision
        if self.sar_enabled and self.sar_height_branch is None:
            raise ValueError("SAR is enabled but no SAR height branch was constructed")
        if self.freeze_maskdino:
            for module in (self.backbone, self.sem_seg_head):
                module.requires_grad_(False)
                module.eval()

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        sar_enabled = cfg.MODEL.SAR.ENABLED
        height_on = cfg.MODEL.MaskDINO.HEIGHT_ON or sar_enabled
        height_weight = (
            cfg.MODEL.HEIGHT.LOSS_WEIGHT
            if sar_enabled
            else cfg.MODEL.MaskDINO.HEIGHT_LOSS_WEIGHT
        )
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        if height_on:
            weight_dict.update({"loss_height": height_weight})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({
                k + f'_interm': v
                for k, v in weight_dict.items()
                if not (sar_enabled and k == "loss_height")
            })
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","boxes"]
        else:
            dn_losses=[]
        if height_on and dn != "no" and not sar_enabled:
            dn_losses.append("heights")
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({
                    k + f"_{i}": v
                    for k, v in weight_dict.items()
                    if not (sar_enabled and k.startswith("loss_height"))
                })
            weight_dict.update(aux_weight_dict)
        if sar_enabled and cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_SUPERVISION:
            weight_dict["loss_global_offset"] = (
                cfg.MODEL.ALIGN.GLOBAL_LOSS_WEIGHT
            )
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]
        if height_on:
            losses.append("heights")
        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            height_scale=(
                cfg.MODEL.HEIGHT.SCALE
                if sar_enabled
                else cfg.MODEL.MaskDINO.HEIGHT_SCALE
            ),
            height_loss_beta=(
                cfg.MODEL.HEIGHT.LOSS_BETA
                if sar_enabled
                else cfg.MODEL.MaskDINO.HEIGHT_LOSS_BETA
            ),
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
            and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS
            and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "height_on": height_on,
            "height_scale": (
                cfg.MODEL.HEIGHT.SCALE
                if sar_enabled
                else cfg.MODEL.MaskDINO.HEIGHT_SCALE
            ),
            "sar_enabled": sar_enabled,
            "sar_height_branch": (
                MaskDINOSARHeightBranch(
                    cfg,
                    optical_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                )
                if sar_enabled
                else None
            ),
            "freeze_maskdino": (
                sar_enabled and cfg.MODEL.MASKDINO.FREEZE
            ),
            "synthetic_shift_supervision": (
                sar_enabled
                and cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_SUPERVISION
            ),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_maskdino:
            self.backbone.eval()
            self.sem_seg_head.eval()
        return self

    def _prepare_multimodal_inputs(self, batched_inputs, images):
        if not self.sar_enabled:
            return None, None, None
        if any("sar_image" not in item for item in batched_inputs):
            raise KeyError(
                "SAR baseline requires every mapped sample to contain sar_image"
            )
        sar_tensors = [item["sar_image"].to(self.device) for item in batched_inputs]
        sar_images = ImageList.from_tensors(
            sar_tensors, self.size_divisibility, pad_value=0.0
        )
        valid_tensors = []
        for item, sar_tensor in zip(batched_inputs, sar_tensors):
            valid = item.get("sar_valid_mask")
            if valid is None:
                valid = torch.ones(
                    sar_tensor.shape[-2:], dtype=torch.bool, device=self.device
                )
            else:
                valid = valid.to(self.device).bool()
            valid_tensors.append(valid[None].float())
        sar_valid = ImageList.from_tensors(
            valid_tensors, self.size_divisibility, pad_value=0.0
        ).tensor[:, 0].bool()
        if sar_images.tensor.shape[-2:] != images.tensor.shape[-2:]:
            raise ValueError(
                "padded optical and SAR shapes differ: "
                f"{images.tensor.shape[-2:]} vs {sar_images.tensor.shape[-2:]}"
            )

        optical_valid = torch.zeros(
            len(batched_inputs),
            *images.tensor.shape[-2:],
            dtype=torch.bool,
            device=self.device,
        )
        for index, (height, width) in enumerate(images.image_sizes):
            optical_valid[index, :height, :width] = True
        return sar_images.tensor, sar_valid, optical_valid

    def _apply_sar_height_branch(
        self,
        outputs,
        sar_images,
        sar_valid_mask,
        optical_valid_mask,
    ):
        interface = outputs.get("maskdino_features")
        if interface is None:
            raise KeyError("MaskDINO head did not expose maskdino_features")
        sar_outputs = self.sar_height_branch(
            decoder_query=interface["decoder_query"],
            reference_boxes=interface["reference_boxes"],
            optical_feature_s8=interface["optical_feature_s8"],
            sar_4ch=sar_images,
            sar_valid_mask=sar_valid_mask,
            optical_valid_mask=optical_valid_mask,
        )
        outputs.update(sar_outputs)
        return outputs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        sar_images, sar_valid_mask, optical_valid_mask = (
            self._prepare_multimodal_inputs(batched_inputs, images)
        )

        if self.freeze_maskdino:
            with torch.no_grad():
                features = self.backbone(images.tensor)
        else:
            features = self.backbone(images.tensor)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if 'detr' in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            if self.freeze_maskdino:
                with torch.no_grad():
                    outputs, mask_dict = self.sem_seg_head(
                        features, targets=targets
                    )
            else:
                outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            if self.sar_enabled:
                outputs = self._apply_sar_height_branch(
                    outputs,
                    sar_images,
                    sar_valid_mask,
                    optical_valid_mask,
                )
            # bipartite matching-based loss
            losses = self.criterion(
                outputs,
                targets,
                mask_dict,
                compute_maskdino_losses=not self.freeze_maskdino,
            )
            if (
                self.synthetic_shift_supervision
                and self.sar_height_branch.global_enabled
            ):
                shift_pixels = torch.stack(
                    [
                        item["synthetic_sar_shift_pixels"].to(self.device)
                        for item in batched_inputs
                    ]
                )
                height, width = sar_images.shape[-2:]
                shift_scale = shift_pixels.new_tensor([width, height])
                target_global_offset = shift_pixels / shift_scale
                losses["loss_global_offset"] = F.smooth_l1_loss(
                    outputs["global_sar_offset"],
                    target_global_offset,
                    reduction="mean",
                )

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs, _ = self.sem_seg_head(features)
            if self.sar_enabled:
                outputs = self._apply_sar_height_branch(
                    outputs,
                    sar_images,
                    sar_valid_mask,
                    optical_valid_mask,
                )
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            height_results = outputs.get("pred_heights")
            global_offset_results = outputs.get("global_sar_offset")
            local_offset_results = outputs.get("local_sar_offsets")
            if self.height_on:
                assert height_results is not None
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            height_results_per_image = (
                height_results
                if height_results is not None
                else [None] * len(batched_inputs)
            )
            global_results_per_image = (
                global_offset_results
                if global_offset_results is not None
                else [None] * len(batched_inputs)
            )
            local_results_per_image = (
                local_offset_results
                if local_offset_results is not None
                else [None] * len(batched_inputs)
            )
            inference_inputs = zip(
                mask_cls_results,
                mask_pred_results,
                mask_box_results,
                height_results_per_image,
                global_results_per_image,
                local_results_per_image,
                batched_inputs,
                images.image_sizes,
            )
            for (
                mask_cls_result,
                mask_pred_result,
                mask_box_result,
                height_result,
                global_offset_result,
                local_offset_result,
                input_per_image,
                image_size,
            ) in inference_inputs:  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                if global_offset_result is not None:
                    processed_results[-1]["global_sar_offset"] = (
                        global_offset_result
                    )
                if local_offset_result is not None:
                    processed_results[-1]["local_sar_offsets"] = (
                        local_offset_result
                    )
                new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # mask_box_result = mask_box_result.to(mask_pred_result)
                    # mask_box_result = self.box_postprocess(mask_box_result, height, width)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)

                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result,
                        mask_pred_result,
                        mask_box_result,
                        height_result,
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            target = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
                "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            }
            if self.height_on:
                target["heights"] = targets_per_image.gt_heights
                target["height_valid"] = targets_per_image.gt_height_valid
            new_targets.append(target)
        return new_targets

    def prepare_targets_detr(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            target = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
                "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy
            }
            if self.height_on:
                target["heights"] = targets_per_image.gt_heights
                target["height_valid"] = targets_per_image.gt_height_valid
            new_targets.append(target)
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, height_result=None):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        selected_query_indices = topk_indices
        mask_pred = mask_pred[topk_indices]
        if height_result is not None:
            height_result = height_result[topk_indices].squeeze(-1)
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
            selected_query_indices = selected_query_indices[keep]
            if height_result is not None:
                height_result = height_result[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        # Retain the exact query selection used by class, box, mask and height
        # so downstream evaluation/debugging can verify one-to-one alignment.
        result.pred_query_indices = selected_query_indices
        if height_result is not None:
            result.pred_heights = height_result.clamp(min=0) * self.height_scale
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes
