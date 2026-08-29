#!/usr/bin/env python3
"""Validate paired SP6 paths, SAR channel semantics, and height annotations."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config


def build_config(config_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def resolve(root, value):
    value = str(value).replace("\\", os.sep)
    return Path(value if os.path.isabs(value) else os.path.join(root, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/sp6/instance-segmentation/maskdino_R50_scratch_bs2_acc8_100ep_height.yaml",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--check-all-files",
        action="store_true",
        help="Check every RGB/SAR pair instead of only sampled pairs.",
    )
    args = parser.parse_args()
    cfg = build_config(args.config_file)
    root = Path(os.getenv("SP6_DATASET_ROOT", cfg.DATASETS.SP6_ROOT))
    annotation_file = Path(
        os.getenv("SP6_ANNOTATION_FILE", cfg.DATASETS.SP6_ANNOTATION_FILE)
    )
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if not annotation_file.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {annotation_file}")

    with annotation_file.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not images:
        raise ValueError("annotation JSON contains no images")
    if not annotations:
        raise ValueError("annotation JSON contains no annotations")

    missing_height = [
        ann["id"]
        for ann in annotations
        if ann.get("height") is None or not np.isfinite(float(ann["height"]))
    ]
    check_images = images
    if not args.check_all_files:
        indices = np.linspace(
            0, len(images) - 1, num=min(args.samples, len(images)), dtype=int
        )
        check_images = [images[index] for index in indices]

    phase_errors = []
    valid_fractions = []
    channel_ranges = [[], [], [], []]
    order = list(cfg.INPUT.SAR_CHANNEL_ORDER)
    for image in check_images:
        rgb_path = resolve(root, image["file_name"])
        sar_path = resolve(root, image["sar_file_name"])
        if not rgb_path.is_file():
            raise FileNotFoundError(f"missing optical image: {rgb_path}")
        if not sar_path.is_file():
            raise FileNotFoundError(f"missing SAR image: {sar_path}")
        if args.check_all_files:
            continue
        sar = cv2.imread(str(sar_path), cv2.IMREAD_UNCHANGED)
        if sar is None or sar.ndim != 3 or sar.shape[2] != 4:
            raise ValueError(f"invalid four-channel SAR GeoTIFF: {sar_path}")
        sar = sar[..., order]
        valid = np.isfinite(sar).all(axis=2)
        if not valid.any():
            raise ValueError(f"SAR tile contains no valid pixels: {sar_path}")
        valid_fractions.append(float(valid.mean()))
        phase_norm = sar[..., 0][valid] ** 2 + sar[..., 1][valid] ** 2
        phase_errors.append(float(np.max(np.abs(phase_norm - 1.0))))
        for channel in range(4):
            values = sar[..., channel][valid]
            channel_ranges[channel].append((float(values.min()), float(values.max())))

    if not args.check_all_files:
        if max(phase_errors) > 1e-3:
            raise ValueError(
                "configured SAR channel order does not produce unit sin/cos vectors; "
                f"maximum error={max(phase_errors):.6g}"
            )
        coherence_min = min(value[0] for value in channel_ranges[2])
        coherence_max = max(value[1] for value in channel_ranges[2])
        if coherence_min < -1e-6 or coherence_max > 1.0 + 1e-6:
            raise ValueError(
                f"coherence channel lies outside [0,1]: {coherence_min}, {coherence_max}"
            )

    summary = {
        "status": "PASS",
        "dataset_root": str(root),
        "annotation_file": str(annotation_file),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "invalid_height_count": len(missing_height),
        "invalid_height_policy": (
            "kept for instance supervision and excluded from height loss"
        ),
        "checked_pair_count": len(check_images),
    }
    if not args.check_all_files:
        summary.update(
            {
                "minimum_sampled_valid_fraction": min(valid_fractions),
                "maximum_sampled_phase_unit_error": max(phase_errors),
                "sampled_channel_ranges": [
                    [
                        min(value[0] for value in ranges),
                        max(value[1] for value in ranges),
                    ]
                    for ranges in channel_ranges
                ],
            }
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
