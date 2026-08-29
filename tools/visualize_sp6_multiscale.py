#!/usr/bin/env python3
"""Visualize SP6 predictions from several test resolutions side by side."""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util


RESOLUTIONS = (512, 768, 1024)
METRICS = {
    512: {"bbox_ap": 19.8078, "mask_ap": 17.1689, "height_mae": 2.3424, "coverage": 34.1356},
    768: {"bbox_ap": 21.2184, "mask_ap": 19.4636, "height_mae": 1.8503, "coverage": 36.1835},
    1024: {"bbox_ap": 18.5415, "mask_ap": 18.2435, "height_mae": 1.8546, "coverage": 30.5181},
}

GT_COLOR = (40, 230, 40)
PRED_COLOR = (30, 120, 255)
TEXT_COLOR = (245, 245, 245)
HEADER_COLOR = (28, 31, 36)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lzq/dataset/sp6_coco_512_balanced"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lzq/Project/MaskDINO/output"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(
            "/home/lzq/Project/MaskDINO/output/sp6_eval_ep10_multiscale"
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def annotation_area(annotation):
    if "area" in annotation:
        return float(annotation["area"])
    return float(annotation["bbox"][2] * annotation["bbox"][3])


def build_image_stats(images, annotations):
    annotations_by_image = defaultdict(list)
    for annotation in annotations:
        if not annotation.get("iscrowd", 0):
            annotations_by_image[annotation["image_id"]].append(annotation)

    stats = []
    for image in images:
        image_annotations = annotations_by_image[image["id"]]
        if not image_annotations:
            continue
        areas = np.asarray(
            [annotation_area(annotation) for annotation in image_annotations],
            dtype=np.float64,
        )
        stats.append(
            {
                "image_id": image["id"],
                "count": len(image_annotations),
                "small": int(np.sum(areas < 32**2)),
                "medium": int(np.sum((areas >= 32**2) & (areas < 96**2))),
                "large": int(np.sum(areas >= 96**2)),
                "max_area": float(areas.max()),
                "mean_area": float(areas.mean()),
            }
        )
    return stats, annotations_by_image


def choose_representative_samples(stats):
    counts = np.asarray([item["count"] for item in stats], dtype=np.float64)
    q20, q40, q70, q85, q95 = np.percentile(counts, [20, 40, 70, 85, 95])

    dense_pool = [
        item
        for item in stats
        if q85 <= item["count"] <= q95
        and item["small"] / item["count"] >= 0.75
    ]
    if not dense_pool:
        dense_pool = sorted(stats, key=lambda item: item["count"], reverse=True)[:50]
    dense = max(
        dense_pool,
        key=lambda item: (
            item["small"] / item["count"],
            item["count"],
        ),
    )

    mixed_pool = [
        item
        for item in stats
        if q40 <= item["count"] <= q70
        and item["small"] >= 2
        and item["medium"] >= 2
        and item["large"] >= 1
    ]
    if not mixed_pool:
        mixed_pool = [
            item
            for item in stats
            if item["small"] and item["medium"] and item["large"]
        ]

    def mixed_score(item):
        proportions = np.asarray(
            [item["small"], item["medium"], item["large"]], dtype=np.float64
        )
        proportions /= proportions.sum()
        entropy = -float(np.sum(proportions * np.log(proportions + 1e-12)))
        return entropy, -abs(item["count"] - np.median(counts))

    mixed = max(mixed_pool, key=mixed_score)

    large_pool = [
        item
        for item in stats
        if 1 <= item["count"] <= max(4, q20) and item["large"] >= 1
    ]
    if not large_pool:
        large_pool = [item for item in stats if item["large"] >= 1]
    sparse_large = max(
        large_pool,
        key=lambda item: (item["max_area"], item["mean_area"]),
    )

    return {
        "dense_small": dense,
        "mixed_scale": mixed,
        "sparse_large": sparse_large,
    }


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


def blend_mask(image, mask, color, alpha):
    result = image.copy()
    color_array = np.asarray(color, dtype=np.float32)
    result[mask] = (
        result[mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha
    ).astype(np.uint8)
    return result


def draw_contours(image, masks, color, thickness):
    for mask in masks:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def draw_text_label(image, text, x, y, scale=0.34):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(np.clip(x, 0, max(0, image.shape[1] - width - 4)))
    y = int(np.clip(y, height + 3, image.shape[0] - baseline - 2))
    cv2.rectangle(
        image,
        (x, y - height - 3),
        (x + width + 3, y + baseline + 2),
        (10, 10, 10),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 1, y),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def annotation_anchor(annotation):
    x, y, width, height = annotation["bbox"]
    return int(x + width / 2), int(y + height / 2)


def render_ground_truth(image, annotations):
    result = image.copy()
    masks = []
    for annotation in sorted(annotations, key=annotation_area, reverse=True):
        mask = segmentation_to_mask(
            annotation["segmentation"], image.shape[0], image.shape[1]
        )
        masks.append(mask)
        result = blend_mask(result, mask, GT_COLOR, 0.24)
    draw_contours(result, masks, GT_COLOR, 1)

    valid_heights = [
        annotation
        for annotation in annotations
        if annotation.get("height_valid", True) and annotation.get("height") is not None
    ]
    for annotation in sorted(
        valid_heights, key=annotation_area, reverse=True
    )[:8]:
        x, y = annotation_anchor(annotation)
        draw_text_label(result, "h={:.1f}m".format(annotation["height"]), x, y)
    return result


def render_predictions(image, predictions, gt_annotations):
    result = image.copy()
    pred_masks = []
    prediction_masks = []
    for prediction in sorted(predictions, key=lambda item: item["score"]):
        mask = segmentation_to_mask(
            prediction["segmentation"], image.shape[0], image.shape[1]
        )
        prediction_masks.append((prediction, mask))
        pred_masks.append(mask)
        result = blend_mask(result, mask, PRED_COLOR, 0.32)

    draw_contours(result, pred_masks, PRED_COLOR, 1)

    gt_masks = [
        segmentation_to_mask(
            annotation["segmentation"], image.shape[0], image.shape[1]
        )
        for annotation in gt_annotations
    ]
    draw_contours(result, gt_masks, GT_COLOR, 1)

    for prediction, _ in sorted(
        prediction_masks, key=lambda item: item[0]["score"], reverse=True
    )[:8]:
        x, y = annotation_anchor(prediction)
        height = prediction.get("height")
        label = "s={:.2f}".format(prediction["score"])
        if height is not None:
            label += " h={:.1f}m".format(height)
        draw_text_label(result, label, x, y)
    return result


def add_panel_header(image, title, subtitle):
    header_height = 66
    panel = np.full(
        (image.shape[0] + header_height, image.shape[1], 3),
        HEADER_COLOR,
        dtype=np.uint8,
    )
    panel[header_height:] = image
    cv2.putText(
        panel,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 210, 215),
        1,
        cv2.LINE_AA,
    )
    return panel


def load_selected_predictions(output_root, selected_ids, threshold):
    selected = {}
    for resolution in RESOLUTIONS:
        prediction_path = (
            output_root
            / "sp6_eval_ep10_{}".format(resolution)
            / "inference"
            / "coco_instances_results.json"
        )
        with prediction_path.open("r", encoding="utf-8") as handle:
            all_predictions = json.load(handle)
        predictions_by_image = defaultdict(list)
        for prediction in all_predictions:
            image_id = prediction["image_id"]
            if image_id in selected_ids and prediction["score"] >= threshold:
                predictions_by_image[image_id].append(prediction)
        selected[resolution] = predictions_by_image
        del all_predictions
        gc.collect()
    return selected


def build_overview_header(width, threshold):
    height = 118
    header = np.full((height, width, 3), (20, 23, 27), dtype=np.uint8)
    cv2.putText(
        header,
        "SP6 epoch-10 checkpoint: test-resolution comparison",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    legend = (
        "Green outline/fill = ground truth; orange mask = prediction "
        "(score >= {:.2f})".format(threshold)
    )
    cv2.putText(
        header,
        legend,
        (18, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (210, 215, 220),
        1,
        cv2.LINE_AA,
    )
    for index, resolution in enumerate(RESOLUTIONS):
        metric = METRICS[resolution]
        text = (
            "{}: bbox AP {:.2f} | mask AP {:.2f} | height MAE {:.2f}m "
            "| coverage {:.2f}%"
        ).format(
            resolution,
            metric["bbox_ap"],
            metric["mask_ap"],
            metric["height_mae"],
            metric["coverage"],
        )
        x = 18 + index * (width // 3)
        cv2.putText(
            header,
            text,
            (x, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return header


def main():
    args = parse_args()
    annotation_path = args.dataset_root / "annotations" / "instances_val.json"
    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    stats, annotations_by_image = build_image_stats(
        coco["images"], coco["annotations"]
    )
    selected_stats = choose_representative_samples(stats)
    selected_ids = {
        item["image_id"] for item in selected_stats.values()
    }
    images_by_id = {image["id"]: image for image in coco["images"]}
    predictions = load_selected_predictions(
        args.output_root, selected_ids, args.score_threshold
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    overview_rows = []
    manifest = {}
    display_names = {
        "dense_small": "Dense small buildings",
        "mixed_scale": "Mixed-scale buildings",
        "sparse_large": "Sparse large building",
    }

    for sample_key, stat in selected_stats.items():
        image_id = stat["image_id"]
        image_info = images_by_id[image_id]
        relative_path = Path(
            image_info.get("rgb_file_name", image_info["file_name"])
        )
        image_path = args.dataset_root / relative_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Could not read {}".format(image_path))

        gt_annotations = annotations_by_image[image_id]
        gt_render = render_ground_truth(image, gt_annotations)
        gt_subtitle = "N={} | small/medium/large={}/{}/{}".format(
            stat["count"], stat["small"], stat["medium"], stat["large"]
        )
        panels = [
            add_panel_header(
                gt_render,
                "{} | Ground truth".format(display_names[sample_key]),
                gt_subtitle,
            )
        ]

        manifest[sample_key] = {
            **stat,
            "file_name": str(relative_path),
            "prediction_counts_at_threshold": {},
        }
        for resolution in RESOLUTIONS:
            image_predictions = predictions[resolution].get(image_id, [])
            scores = [prediction["score"] for prediction in image_predictions]
            pred_render = render_predictions(
                image, image_predictions, gt_annotations
            )
            subtitle = "Pred N={} | mean score={:.3f}".format(
                len(image_predictions),
                float(np.mean(scores)) if scores else 0.0,
            )
            panels.append(
                add_panel_header(
                    pred_render,
                    "{} input".format(resolution),
                    subtitle,
                )
            )
            manifest[sample_key]["prediction_counts_at_threshold"][
                str(resolution)
            ] = len(image_predictions)

        row = np.concatenate(panels, axis=1)
        cv2.imwrite(str(args.save_dir / "{}.png".format(sample_key)), row)
        overview_rows.append(row)

    overview = np.concatenate(
        [build_overview_header(overview_rows[0].shape[1], args.score_threshold)]
        + overview_rows,
        axis=0,
    )
    cv2.imwrite(str(args.save_dir / "multiscale_overview.png"), overview)

    with (args.save_dir / "selected_samples.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Saved visualizations to {}".format(args.save_dir))


if __name__ == "__main__":
    main()
