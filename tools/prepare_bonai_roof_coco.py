#!/usr/bin/env python3
"""Prepare BONAI building-box/roof-mask annotations in COCO format.

BONAI uses two annotation layouts:

* The five trainval files store roof polygons in ``segmentation``.
* The official test file stores footprints in ``segmentation`` and exposes the
  roof polygon separately as ``roof_mask``.

This script handles both layouts explicitly, always uses ``building_bbox`` for
box supervision, always uses the roof polygon for mask supervision, and maps
BONAI's ``ignore_flag`` to COCO's ``iscrowd``.
"""

import argparse
import json
import os
from pathlib import Path

from pycocotools import mask as coco_mask


TRAIN_FILE_NAMES = (
    "bonai_beijing_trainval.json",
    "bonai_chengdu_trainval.json",
    "bonai_haerbin_trainval.json",
    "bonai_jinan_trainval.json",
    "bonai_shanghai_trainval.json",
)
VAL_FILE_NAME = "bonai_shanghai_xian_test.json"
CATEGORIES = [{"id": 1, "name": "building", "supercategory": "none"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/lzq/dataset/BONAI"),
        help="Root of the extracted BONAI dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory "
            "(default: ROOT/coco_building_bbox_roof_mask)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated annotation files.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_polygon_segmentation(segmentation, source_annotation_id):
    """Return a COCO list-of-polygons representation."""
    if not segmentation:
        raise ValueError(
            "Empty roof mask in source annotation {}".format(
                source_annotation_id
            )
        )

    if isinstance(segmentation, dict):
        raise ValueError(
            "Expected polygon roof mask in source annotation {}, got RLE".format(
                source_annotation_id
            )
        )

    if isinstance(segmentation[0], (int, float)):
        polygons = [segmentation]
    else:
        polygons = segmentation

    normalized = []
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError(
                "Invalid roof polygon in source annotation {}".format(
                    source_annotation_id
                )
            )
        normalized.append(list(polygon))
    return normalized


def polygon_area(segmentation, image_height, image_width):
    """Calculate the rasterized roof-mask area used by COCO evaluation."""
    rles = coco_mask.frPyObjects(
        segmentation,
        image_height,
        image_width,
    )
    return float(coco_mask.area(coco_mask.merge(rles)))


def get_roof_segmentation(annotation):
    """Read the roof polygon from either BONAI annotation layout."""
    roof_mask = annotation.get("roof_mask")
    if roof_mask is None:
        roof_mask = annotation["segmentation"]
    return normalize_polygon_segmentation(roof_mask, annotation.get("id"))


def clean_annotation(
    annotation,
    image_id,
    annotation_id,
    image_height,
    image_width,
):
    """Retain building bbox and roof mask, or skip a fully invisible roof."""
    segmentation = get_roof_segmentation(annotation)
    area = polygon_area(segmentation, image_height, image_width)
    if area <= 0:
        return None

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": list(annotation["building_bbox"]),
        "area": area,
        "segmentation": segmentation,
        "iscrowd": int(
            bool(annotation.get("iscrowd", 0))
            or bool(annotation.get("ignore_flag", 0))
        ),
    }


def clean_image(image, image_id):
    result = dict(image)
    result["id"] = image_id
    return result


def merge_sources(source_paths, description):
    images = []
    annotations = []
    next_image_id = 1
    next_annotation_id = 1
    licenses = []
    skipped_empty_masks = 0

    for source_path in source_paths:
        source = load_json(source_path)
        if source.get("categories") != CATEGORIES:
            raise ValueError(
                "Unexpected categories in {}: {}".format(
                    source_path, source.get("categories")
                )
            )
        if not licenses:
            licenses = source.get("licenses", [])

        image_id_map = {}
        image_size_map = {}
        for image in source["images"]:
            image_id_map[image["id"]] = next_image_id
            image_size_map[image["id"]] = (
                image["height"],
                image["width"],
            )
            images.append(clean_image(image, next_image_id))
            next_image_id += 1

        source_skipped_empty_masks = 0
        for annotation in source["annotations"]:
            old_image_id = annotation["image_id"]
            if old_image_id not in image_id_map:
                raise ValueError(
                    "Annotation {} in {} references missing image {}".format(
                        annotation.get("id"), source_path, old_image_id
                    )
                )
            cleaned = clean_annotation(
                annotation,
                image_id_map[old_image_id],
                next_annotation_id,
                *image_size_map[old_image_id],
            )
            if cleaned is None:
                source_skipped_empty_masks += 1
                skipped_empty_masks += 1
                continue
            annotations.append(cleaned)
            next_annotation_id += 1

        print(
            "Loaded {}: {} images, {} annotations, {} raster-empty roof "
            "annotations skipped".format(
                source_path.name,
                len(source["images"]),
                len(source["annotations"]),
                source_skipped_empty_masks,
            )
        )

    return {
        "info": {
            "description": description,
            "source": "BONAI",
            "bbox_target": "building_bbox",
            "mask_target": "roof_mask",
            "ignore_policy": "BONAI ignore_flag mapped to COCO iscrowd",
            "empty_mask_policy": (
                "roof masks with zero visible pixels are excluded"
            ),
            "empty_masks_excluded": skipped_empty_masks,
        },
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
        "type": "instances",
    }


def validate_dataset(dataset, image_root):
    image_ids = [image["id"] for image in dataset["images"]]
    annotation_ids = [
        annotation["id"] for annotation in dataset["annotations"]
    ]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Generated image IDs are not unique")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Generated annotation IDs are not unique")

    image_id_set = set(image_ids)
    missing_references = [
        annotation["id"]
        for annotation in dataset["annotations"]
        if annotation["image_id"] not in image_id_set
    ]
    if missing_references:
        raise ValueError(
            "{} annotations reference missing images".format(
                len(missing_references)
            )
        )

    missing_images = [
        image["file_name"]
        for image in dataset["images"]
        if not (image_root / image["file_name"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            "{} referenced image files are missing; first: {}".format(
                len(missing_images), missing_images[0]
            )
        )

    for annotation in dataset["annotations"]:
        bbox = annotation["bbox"]
        segmentation = annotation["segmentation"]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(
                "Invalid bbox in annotation {}".format(annotation["id"])
            )
        if not segmentation:
            raise ValueError(
                "Empty roof segmentation in annotation {}".format(
                    annotation["id"]
                )
            )
        if annotation["area"] <= 0:
            raise ValueError(
                "Non-positive roof area in annotation {}".format(
                    annotation["id"]
                )
            )

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        for polygon in segmentation:
            xs = polygon[0::2]
            ys = polygon[1::2]
            if (
                min(xs) < x_min - 1e-6
                or max(xs) > x_max + 1e-6
                or min(ys) < y_min - 1e-6
                or max(ys) > y_max + 1e-6
            ):
                raise ValueError(
                    "Roof polygon lies outside building bbox in annotation "
                    "{}".format(annotation["id"])
                )


def write_json(dataset, output_path, force):
    if output_path.exists() and not force:
        raise FileExistsError(
            "{} already exists; pass --force to replace it".format(output_path)
        )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            dataset,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    os.replace(temporary_path, output_path)


def print_summary(name, dataset, output_path):
    crowd = sum(
        annotation["iscrowd"] for annotation in dataset["annotations"]
    )
    usable = len(dataset["annotations"]) - crowd
    print(
        "{}: {} images, {} annotations, {} train/eval instances, "
        "{} ignored as crowd".format(
            name,
            len(dataset["images"]),
            len(dataset["annotations"]),
            usable,
            crowd,
        )
    )
    print("Wrote {}".format(output_path))


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "coco_building_bbox_roof_mask"
    )
    source_dir = root / "coco"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [source_dir / name for name in TRAIN_FILE_NAMES]
    val_path = source_dir / VAL_FILE_NAME
    for path in train_paths + [val_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train = merge_sources(
        train_paths,
        "BONAI five-city trainval split, building bbox and roof mask",
    )
    val = merge_sources(
        [val_path],
        "BONAI official Shanghai/Xian test split, building bbox and roof mask",
    )

    validate_dataset(train, root / "trainval" / "images")
    validate_dataset(val, root / "test")

    train_output = output_dir / "instances_train.json"
    val_output = output_dir / "instances_val.json"
    write_json(train, train_output, args.force)
    write_json(val, val_output, args.force)
    print_summary("train", train, train_output)
    print_summary("val", val, val_output)


if __name__ == "__main__":
    main()
