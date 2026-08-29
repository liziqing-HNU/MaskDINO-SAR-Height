#!/usr/bin/env python3
"""Small CPU-only MaskDINO mask-threshold diagnostic for BONAI.

The script runs the network once per image, captures the raw instance-mask
logits, and applies several sigmoid thresholds without repeating the backbone
forward pass.  It is intended for quick diagnostics rather than full COCO
evaluation.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from pycocotools import mask as mask_util

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances

from maskdino import add_maskdino_config


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/bonai/instance-segmentation/"
    "maskdino_R50_bs2_50ep_roof_512.yaml"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output/bonai_building_bbox_roof_mask_preserve_bbox_R50_bs2_50ep_512"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "model_0007499.pth",
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=Path(
            "/home/lzq/dataset/BONAI/"
            "coco_building_bbox_roof_mask/instances_val.json"
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/home/lzq/dataset/BONAI/test"),
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=Path("/tmp/bonai_facade_leak_candidates.json"),
    )
    parser.add_argument(
        "--image-ids", type=int, nargs="+", default=[5, 13, 218, 242]
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7]
    )
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument(
        "--output-image",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cpu_mask_threshold_test_epoch5.png",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cpu_mask_threshold_test_epoch5.json",
    )
    return parser.parse_args()


def segmentation_to_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_util.frPyObjects(segmentation, height, width)
        rle = mask_util.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(
        segmentation.get("counts"), list
    ):
        rle = mask_util.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def resize_shortest_edge(image, short_edge, max_size):
    height, width = image.shape[:2]
    scale = float(short_edge) / min(height, width)
    if scale * max(height, width) > max_size:
        scale = float(max_size) / max(height, width)
    new_height = int(height * scale + 0.5)
    new_width = int(width * scale + 0.5)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def box_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0) * np.maximum(
        boxes1[:, 3] - boxes1[:, 1], 0
    )
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0) * np.maximum(
        boxes2[:, 3] - boxes2[:, 1], 0
    )
    return intersection / np.maximum(
        area1[:, None] + area2[None, :] - intersection, 1e-6
    )


def xywh_to_xyxy(box):
    x, y, width, height = box
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def mask_metrics(pred_mask, gt_mask, building_box, original_size):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
    pred_area = pred_mask.sum(dtype=np.float64)
    gt_area = gt_mask.sum(dtype=np.float64)
    union = pred_area + gt_area - intersection

    mask_height, mask_width = pred_mask.shape
    original_height, original_width = original_size
    scale_x = mask_width / original_width
    scale_y = mask_height / original_height
    x1, y1, x2, y2 = building_box
    x1 = int(np.floor(x1 * scale_x))
    x2 = int(np.ceil(x2 * scale_x))
    y1 = int(np.floor(y1 * scale_y))
    y2 = int(np.ceil(y2 * scale_y))
    x1, x2 = np.clip([x1, x2], 0, mask_width)
    y1, y2 = np.clip([y1, y2], 0, mask_height)
    bbox_region = np.zeros_like(pred_mask)
    bbox_region[y1:y2, x1:x2] = True
    facade_like = np.logical_and.reduce((pred_mask, bbox_region, ~gt_mask)).sum(
        dtype=np.float64
    )

    return {
        "iou": float(intersection / max(union, 1.0)),
        "precision": float(intersection / max(pred_area, 1.0)),
        "recall": float(intersection / max(gt_area, 1.0)),
        "facade_leak": float(facade_like / max(pred_area, 1.0)),
    }


def build_cpu_model(config_path, checkpoint_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = str(checkpoint_path)
    cfg.SOLVER.AMP.ENABLED = False
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(str(checkpoint_path))

    # Keep logits on the configured 512-pixel inference grid.  This avoids
    # allocating 300 full-resolution 1024x1024 float masks on CPU.
    model.sem_seg_postprocess_before_inference = False
    return cfg, model


def run_one_forward(model, image_tensor, original_height, original_width, max_queries):
    capture = {}

    def capture_instance(mask_cls, mask_pred, mask_boxes, height_result=None):
        class_prob = mask_cls.sigmoid()
        flat_prob = class_prob.flatten()
        query_count = min(max_queries, flat_prob.numel())
        class_scores, flat_indices = flat_prob.topk(query_count, sorted=True)
        class_count = class_prob.shape[1]
        query_indices = flat_indices // class_count
        labels = flat_indices % class_count
        capture["class_scores"] = class_scores.detach().cpu()
        capture["labels"] = labels.detach().cpu()
        capture["mask_logits"] = mask_pred[query_indices].detach().cpu()
        capture["boxes"] = mask_boxes[query_indices].detach().cpu()
        return Instances(mask_pred.shape[-2:])

    model.instance_inference = capture_instance
    inputs = [
        {
            "image": image_tensor,
            "height": original_height,
            "width": original_width,
        }
    ]
    start = time.perf_counter()
    with torch.inference_mode():
        model(inputs)
    elapsed = time.perf_counter() - start
    if not capture:
        raise RuntimeError("Instance logits were not captured.")
    return capture, elapsed


def greedy_matches(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    ious = box_iou_matrix(pred_boxes, gt_boxes)
    matches = []
    used_gt = set()
    for pred_index in np.argsort(-np.asarray(pred_scores)):
        if ious.shape[1] == 0:
            break
        available = [
            gt_index for gt_index in range(ious.shape[1]) if gt_index not in used_gt
        ]
        if not available:
            break
        best_gt = max(available, key=lambda gt_index: ious[pred_index, gt_index])
        if ious[pred_index, best_gt] >= iou_threshold:
            used_gt.add(best_gt)
            matches.append((int(pred_index), int(best_gt), float(ious[pred_index, best_gt])))
    return matches


def evaluate_threshold(
    capture,
    threshold,
    score_threshold,
    annotations,
    gt_masks,
    original_size,
    target_gt_id,
):
    probabilities = capture["mask_logits"].sigmoid()
    binary_masks = probabilities > threshold
    mask_area = binary_masks.flatten(1).sum(1)
    mask_quality = (
        probabilities.mul(binary_masks).flatten(1).sum(1)
        / (mask_area + 1e-6)
    )
    scores = capture["class_scores"] * mask_quality
    keep = scores >= score_threshold

    kept_scores = scores[keep].numpy()
    kept_boxes = capture["boxes"][keep].numpy()
    kept_masks = binary_masks[keep].numpy()
    gt_boxes = np.stack([xywh_to_xyxy(ann["bbox"]) for ann in annotations])
    matches = greedy_matches(kept_boxes, kept_scores, gt_boxes)

    matched_metrics = []
    target_result = None
    for pred_index, gt_index, bbox_iou in matches:
        metrics = mask_metrics(
            kept_masks[pred_index],
            gt_masks[gt_index],
            gt_boxes[gt_index],
            original_size,
        )
        metrics.update(
            {
                "score": float(kept_scores[pred_index]),
                "bbox_iou": bbox_iou,
                "gt_id": int(annotations[gt_index]["id"]),
            }
        )
        matched_metrics.append(metrics)
        if int(annotations[gt_index]["id"]) == target_gt_id:
            target_result = {
                **metrics,
                "mask": kept_masks[pred_index],
                "box": kept_boxes[pred_index],
            }

    aggregate = {
        "detections": int(keep.sum()),
        "bbox_matches": len(matches),
    }
    for key in ("iou", "precision", "recall", "facade_leak"):
        aggregate[key] = (
            float(np.mean([item[key] for item in matched_metrics]))
            if matched_metrics
            else None
        )
    return aggregate, target_result


def overlay_mask(image, mask, color, alpha=0.5):
    result = image.astype(np.float32).copy()
    color = np.asarray(color, dtype=np.float32)
    result[mask] = result[mask] * (1.0 - alpha) + color * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_bounds(box, height, width, expansion=0.35):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    side = max(box_width, box_height) * (1.0 + 2.0 * expansion)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    x1 = max(0, int(np.floor(center_x - side / 2)))
    x2 = min(width, int(np.ceil(center_x + side / 2)))
    y1 = max(0, int(np.floor(center_y - side / 2)))
    y2 = min(height, int(np.ceil(center_y + side / 2)))
    return x1, y1, x2, y2


def visualize(rows, thresholds, output_path, checkpoint_label):
    figure, axes = plt.subplots(
        len(rows),
        len(thresholds) + 1,
        figsize=(4.0 * (len(thresholds) + 1), 3.8 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        image = row["image"]
        height, width = image.shape[:2]
        gt_mask = row["target_gt_mask_full"]
        gt_box = row["target_gt_box"]
        x1, y1, x2, y2 = crop_bounds(gt_box, height, width)

        gt_panel = overlay_mask(image, gt_mask, (0, 220, 255), alpha=0.48)
        cv2.rectangle(
            gt_panel,
            (int(gt_box[0]), int(gt_box[1])),
            (int(gt_box[2]), int(gt_box[3])),
            (255, 230, 0),
            3,
        )
        axes[row_index, 0].imshow(gt_panel[y1:y2, x1:x2])
        axes[row_index, 0].set_title(
            f"image {row['image_id']} / roof GT\nbuilding bbox: yellow",
            fontsize=10,
        )

        for column_index, threshold in enumerate(thresholds, start=1):
            target = row["targets"].get(str(threshold))
            panel = image.copy()
            if target is not None:
                pred_small = target["mask"]
                pred_mask = cv2.resize(
                    pred_small.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                true_positive = np.logical_and(pred_mask, gt_mask)
                false_positive = np.logical_and(pred_mask, ~gt_mask)
                false_negative = np.logical_and(~pred_mask, gt_mask)
                panel = overlay_mask(panel, false_negative, (20, 160, 255), 0.55)
                panel = overlay_mask(panel, false_positive, (255, 30, 30), 0.58)
                panel = overlay_mask(panel, true_positive, (30, 220, 80), 0.55)
                title = (
                    f"threshold {threshold:.1f}\n"
                    f"IoU {target['iou']:.3f}  P {target['precision']:.3f}  "
                    f"R {target['recall']:.3f}\n"
                    f"facade-like leak {target['facade_leak']:.3f}"
                )
            else:
                panel = overlay_mask(panel, gt_mask, (20, 160, 255), 0.55)
                title = f"threshold {threshold:.1f}\nno bbox-matched detection"
            cv2.rectangle(
                panel,
                (int(gt_box[0]), int(gt_box[1])),
                (int(gt_box[2]), int(gt_box[3])),
                (255, 230, 0),
                3,
            )
            axes[row_index, column_index].imshow(panel[y1:y2, x1:x2])
            axes[row_index, column_index].set_title(title, fontsize=9)

        for axis in axes[row_index]:
            axis.axis("off")

    figure.legend(
        handles=[
            Patch(color=(30 / 255, 220 / 255, 80 / 255), label="TP: predicted roof"),
            Patch(color=(1.0, 30 / 255, 30 / 255), label="FP: outside roof GT"),
            Patch(color=(20 / 255, 160 / 255, 1.0), label="FN: missed roof GT"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"BONAI {checkpoint_label}: CPU mask-threshold diagnostic",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def json_ready_target(target):
    if target is None:
        return None
    return {
        key: value
        for key, value in target.items()
        if key not in {"mask", "box"}
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    dataset = json.loads(args.annotation_json.read_text())
    images = {int(item["id"]): item for item in dataset["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    candidates = {}
    if args.candidate_json.is_file():
        for item in json.loads(args.candidate_json.read_text()):
            candidates[int(item["image_id"])] = int(item["gt"]["id"])

    missing = [image_id for image_id in args.image_ids if image_id not in images]
    if missing:
        raise KeyError(f"Unknown image IDs: {missing}")

    print(f"Building CPU model from {args.checkpoint}", flush=True)
    cfg, model = build_cpu_model(args.config, args.checkpoint)
    print(
        f"CPU model ready; images={args.image_ids}, "
        f"thresholds={args.thresholds}, threads={args.threads}",
        flush=True,
    )

    all_results = []
    visualization_rows = []
    total_start = time.perf_counter()

    for position, image_id in enumerate(args.image_ids, start=1):
        image_info = images[image_id]
        image_path = args.image_root / image_info["file_name"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        resized = resize_shortest_edge(
            image, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
        )
        image_tensor = torch.as_tensor(
            np.ascontiguousarray(resized.transpose(2, 0, 1))
        )

        print(
            f"[{position}/{len(args.image_ids)}] forward image {image_id} "
            f"({original_width}x{original_height} -> "
            f"{resized.shape[1]}x{resized.shape[0]})",
            flush=True,
        )
        capture, elapsed = run_one_forward(
            model,
            image_tensor,
            original_height,
            original_width,
            args.max_queries,
        )
        print(f"  forward finished in {elapsed:.1f}s", flush=True)

        annotations = annotations_by_image[image_id]
        mask_height, mask_width = capture["mask_logits"].shape[-2:]
        gt_masks = []
        for annotation in annotations:
            full_mask = segmentation_to_mask(
                annotation["segmentation"], original_height, original_width
            )
            gt_masks.append(
                cv2.resize(
                    full_mask.astype(np.uint8),
                    (mask_width, mask_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
        gt_masks = np.stack(gt_masks)

        target_gt_id = candidates.get(image_id)
        if target_gt_id is None:
            target_gt_id = int(max(annotations, key=lambda item: item["area"])["id"])
        target_index = next(
            index
            for index, annotation in enumerate(annotations)
            if int(annotation["id"]) == target_gt_id
        )
        target_annotation = annotations[target_index]
        target_gt_mask_full = segmentation_to_mask(
            target_annotation["segmentation"], original_height, original_width
        )

        image_result = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "forward_seconds": elapsed,
            "target_gt_id": target_gt_id,
            "thresholds": {},
        }
        row = {
            "image_id": image_id,
            "image": image,
            "target_gt_mask_full": target_gt_mask_full,
            "target_gt_box": xywh_to_xyxy(target_annotation["bbox"]),
            "targets": {},
        }
        for threshold in args.thresholds:
            aggregate, target = evaluate_threshold(
                capture,
                threshold,
                args.score_threshold,
                annotations,
                gt_masks,
                (original_height, original_width),
                target_gt_id,
            )
            key = str(threshold)
            image_result["thresholds"][key] = {
                "aggregate": aggregate,
                "target": json_ready_target(target),
            }
            row["targets"][key] = target
            target_text = (
                "not matched"
                if target is None
                else (
                    f"IoU={target['iou']:.3f}, P={target['precision']:.3f}, "
                    f"R={target['recall']:.3f}, leak={target['facade_leak']:.3f}"
                )
            )
            print(
                f"  threshold {threshold:.1f}: detections={aggregate['detections']}, "
                f"matches={aggregate['bbox_matches']}, target {target_text}",
                flush=True,
            )
        all_results.append(image_result)
        visualization_rows.append(row)

    threshold_summary = {}
    for threshold in args.thresholds:
        key = str(threshold)
        aggregates = [
            result["thresholds"][key]["aggregate"] for result in all_results
        ]
        targets = [
            result["thresholds"][key]["target"]
            for result in all_results
            if result["thresholds"][key]["target"] is not None
        ]
        matched_values = {
            metric: [
                aggregate[metric]
                for aggregate in aggregates
                if aggregate[metric] is not None
            ]
            for metric in ("iou", "precision", "recall", "facade_leak")
        }
        threshold_summary[key] = {
            "mean_detections_per_image": float(
                np.mean([item["detections"] for item in aggregates])
            ),
            "total_bbox_matches": int(
                sum(item["bbox_matches"] for item in aggregates)
            ),
            "matched_instance_mean": {
                metric: (
                    float(np.mean(values)) if values else None
                )
                for metric, values in matched_values.items()
            },
            "selected_facade_case_mean": {
                metric: (
                    float(np.mean([target[metric] for target in targets]))
                    if targets
                    else None
                )
                for metric in ("iou", "precision", "recall", "facade_leak")
            },
        }

    output = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": "cpu",
        "inference_grid": [
            int(capture["mask_logits"].shape[-2]),
            int(capture["mask_logits"].shape[-1]),
        ],
        "score_threshold": args.score_threshold,
        "max_queries": args.max_queries,
        "image_ids": args.image_ids,
        "total_seconds": time.perf_counter() - total_start,
        "summary": threshold_summary,
        "images": all_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    visualize(
        visualization_rows,
        args.thresholds,
        args.output_image,
        args.checkpoint.stem,
    )

    print("\nSummary", flush=True)
    for threshold in args.thresholds:
        summary = threshold_summary[str(threshold)]
        matched = summary["matched_instance_mean"]
        selected = summary["selected_facade_case_mean"]
        print(
            f"threshold {threshold:.1f}: all matched "
            f"IoU={matched['iou']:.3f}, P={matched['precision']:.3f}, "
            f"R={matched['recall']:.3f}, leak={matched['facade_leak']:.3f}; "
            f"selected leak={selected['facade_leak']:.3f}",
            flush=True,
        )
    print(f"Saved metrics: {args.output_json}", flush=True)
    print(f"Saved visualization: {args.output_image}", flush=True)


if __name__ == "__main__":
    main()
