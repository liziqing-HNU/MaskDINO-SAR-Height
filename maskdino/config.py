# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from detectron2.config import CfgNode as CN


def add_maskdino_config(cfg):
    """
    Add config for MaskDINO.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "MaskDINO_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    # Standard instance datasets define bbox as the tight mask bounds. Some
    # datasets (for example BONAI building_bbox + roof_mask) intentionally use
    # a box target that differs from the mask bounds.
    cfg.INPUT.RECOMPUTE_BOXES_FROM_MASKS = True
    # Multimodal SAR input. OpenCV returns the first three GeoTIFF bands in
    # reverse order, hence [2, 1, 0, 3] restores the physical on-disk order
    # [sin(phi), cos(phi), coherence, intensity_db].
    cfg.INPUT.SAR_CHANNEL_ORDER = [2, 1, 0, 3]

    # Optional paths and deterministic split settings used by the SP6 dataset
    # registration. The baseline YAML keeps every machine-dependent path in a
    # single PATHS block and aliases these fields to it.
    cfg.DATASETS.SP6_ROOT = ""
    cfg.DATASETS.SP6_ANNOTATION_FILE = ""
    cfg.DATASETS.SP6_SPLIT_RATIOS = [0.8, 0.1, 0.1]
    cfg.DATASETS.SP6_SPLIT_SEED = 20260829
    cfg.DATASETS.SP6_MAX_TRAIN_IMAGES = 0

    cfg.PATHS = CN()
    cfg.PATHS.DATASET_ROOT = ""
    cfg.PATHS.ANNOTATION_FILE = ""
    cfg.PATHS.MASKDINO_WEIGHTS = ""
    cfg.PATHS.SAR_RESNET18_WEIGHTS = ""
    cfg.PATHS.OUTPUT_DIR = ""

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # MaskDINO model config
    cfg.MODEL.MaskDINO = CN()
    cfg.MODEL.MaskDINO.LEARN_TGT = False

    # loss
    cfg.MODEL.MaskDINO.PANO_BOX_LOSS = False
    cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS = False
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.
    # Per-instance building height regression. Heights are divided by
    # HEIGHT_SCALE for optimization and converted back to metres at inference.
    cfg.MODEL.MaskDINO.HEIGHT_ON = False
    cfg.MODEL.MaskDINO.HEIGHT_LOSS_WEIGHT = 1.0
    cfg.MODEL.MaskDINO.HEIGHT_SCALE = 10.0
    cfg.MODEL.MaskDINO.HEIGHT_LOSS_BETA = 0.1

    # MaskDINO -> SAR geometry interface. Upper-case MASKDINO is deliberately
    # separate from the legacy MaskDINO decoder configuration above.
    cfg.MODEL.MASKDINO = CN()
    cfg.MODEL.MASKDINO.FREEZE = True
    cfg.MODEL.MASKDINO.DETACH_REFERENCE = True

    # Four-channel SAR encoder and matching projections.
    cfg.MODEL.SAR = CN()
    cfg.MODEL.SAR.ENABLED = False
    cfg.MODEL.SAR.PHASE_DESCRIPTOR = True
    cfg.MODEL.SAR.AMP_ENABLED = True
    cfg.MODEL.SAR.AMP_GATE_ENABLED = True
    cfg.MODEL.SAR.RESNET18_WEIGHTS = ""
    cfg.MODEL.SAR.FREEZE_BATCH_NORM = True
    cfg.MODEL.SAR.INTENSITY_CLIP = [-18.224987697601318, 5.063441348075877]
    cfg.MODEL.SAR.INTENSITY_MEAN = -12.176343665977567
    cfg.MODEL.SAR.INTENSITY_STD = 4.21532613169376
    cfg.MODEL.SAR.COHERENCE_CLIP = [0.0, 1.0]
    cfg.MODEL.SAR.MATCH_DIM = 64

    # Weak optical/SAR alignment.
    cfg.MODEL.ALIGN = CN()
    cfg.MODEL.ALIGN.GLOBAL_ENABLED = True
    cfg.MODEL.ALIGN.LOCAL_CORR_ENABLED = True
    cfg.MODEL.ALIGN.MATCHING_STRIDE = 8
    cfg.MODEL.ALIGN.GLOBAL_SEARCH_RADIUS = 8
    cfg.MODEL.ALIGN.LOCAL_SEARCH_RADIUS = 4
    cfg.MODEL.ALIGN.GLOBAL_TEMPERATURE = 0.1
    cfg.MODEL.ALIGN.LOCAL_TEMPERATURE = 0.1
    cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_SUPERVISION = True
    cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_PROBABILITY = 1.0
    cfg.MODEL.ALIGN.SYNTHETIC_SHIFT_MAX_PIXELS = 32
    cfg.MODEL.ALIGN.GLOBAL_LOSS_WEIGHT = 1.0

    # Three-layer SAR deformable cross-attention decoder.
    cfg.MODEL.GEOMETRY_DECODER = CN()
    cfg.MODEL.GEOMETRY_DECODER.NUM_LAYERS = 3
    cfg.MODEL.GEOMETRY_DECODER.USE_SELF_ATTN = False
    cfg.MODEL.GEOMETRY_DECODER.D_MODEL = 256
    cfg.MODEL.GEOMETRY_DECODER.NHEADS = 8
    cfg.MODEL.GEOMETRY_DECODER.N_LEVELS = 4
    cfg.MODEL.GEOMETRY_DECODER.N_POINTS = 4
    cfg.MODEL.GEOMETRY_DECODER.DIM_FEEDFORWARD = 1024
    cfg.MODEL.GEOMETRY_DECODER.DROPOUT = 0.1

    # Final height is predicted directly from the multimodal query. SCALE is
    # kept explicit for compatibility, but is 1.0 for metre-valued regression.
    cfg.MODEL.HEIGHT = CN()
    cfg.MODEL.HEIGHT.ACTIVATION = "softplus"
    cfg.MODEL.HEIGHT.LOSS_WEIGHT = 1.0
    cfg.MODEL.HEIGHT.LOSS_BETA = 1.0
    cfg.MODEL.HEIGHT.SCALE = 1.0
    cfg.MODEL.HEIGHT.DROPOUT = 0.1

    # cost weight
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.MaskDINO.NHEADS = 8
    cfg.MODEL.MaskDINO.DROPOUT = 0.1
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MaskDINO.ENC_LAYERS = 0
    cfg.MODEL.MaskDINO.DEC_LAYERS = 6
    cfg.MODEL.MaskDINO.INITIAL_PRED = True
    cfg.MODEL.MaskDINO.PRE_NORM = False
    cfg.MODEL.MaskDINO.BOX_LOSS = True
    cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.MaskDINO.DN="seg"
    cfg.MODEL.MaskDINO.DN_NOISE_SCALE=0.4
    cfg.MODEL.MaskDINO.DN_NUM=100
    cfg.MODEL.MaskDINO.PRED_CONV=False

    cfg.MODEL.MaskDINO.EVAL_FLAG = 1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # MaskDINO inference config
    cfg.MODEL.MaskDINO.TEST = CN()
    cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = True
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE = 0.06
    cfg.MODEL.MaskDINO.TEST.HEIGHT_SCORE_THRESHOLD = 0.5
    cfg.MODEL.MaskDINO.TEST.HEIGHT_IOU_THRESHOLD = 0.5
    # cfg.MODEL.MaskDINO.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"

    # transformer module
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.Default_loading=True  # a bug in my d2. resume use this; if first time ResNet load, set it false
