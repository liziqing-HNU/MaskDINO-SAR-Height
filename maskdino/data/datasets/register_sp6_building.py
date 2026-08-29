"""Register paired optical/SAR SpaceNet6 building-height data."""

import hashlib
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


DEFAULT_SP6_ROOT = "datasets/sp6_sar_height"
DEFAULT_ANNOTATION = "building_height_coco_repaired.json"
_SPLITS = ("train", "val", "test")


def _validate_ratios(split_ratios):
    if len(split_ratios) != 3:
        raise ValueError("SP6_SPLIT_RATIOS must contain train/val/test ratios")
    split_ratios = tuple(float(value) for value in split_ratios)
    if any(value < 0 for value in split_ratios):
        raise ValueError("SP6 split ratios must be non-negative")
    total = sum(split_ratios)
    if total <= 0:
        raise ValueError("at least one SP6 split ratio must be positive")
    return tuple(value / total for value in split_ratios)


def _split_for_image(image, split_ratios, split_seed):
    key = image.get("chip_id", image["id"])
    digest = hashlib.sha256(f"{split_seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    train_end = split_ratios[0]
    val_end = train_end + split_ratios[1]
    if value < train_end:
        return "train"
    if value < val_end:
        return "val"
    return "test"


def _resolve_path(root, path):
    path = str(path).replace("\\", os.sep)
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(root, path))


def load_sp6_building_json(
    json_file,
    image_root,
    dataset_name,
    split,
    split_ratios=(0.8, 0.1, 0.1),
    split_seed=20260829,
    max_train_images=0,
):
    """Load COCO records while retaining paired SAR paths and height labels."""
    split_ratios = _validate_ratios(split_ratios)
    with open(json_file, "r", encoding="utf-8") as handle:
        coco_data = json.load(handle)
    image_metadata = {image["id"]: image for image in coco_data["images"]}

    records = load_coco_json(
        json_file,
        image_root,
        dataset_name,
        extra_annotation_keys=["height", "height_valid", "Building_ID"],
    )
    selected = []
    for record in records:
        image = image_metadata[record["image_id"]]
        if _split_for_image(image, split_ratios, split_seed) != split:
            continue
        sar_file_name = image.get("sar_file_name")
        if not sar_file_name:
            raise KeyError(
                f"image {record['image_id']} has no sar_file_name in {json_file}"
            )
        record["sar_file_name"] = _resolve_path(image_root, sar_file_name)
        if image.get("source_pair_file_name"):
            record["source_pair_file_name"] = _resolve_path(
                image_root, image["source_pair_file_name"]
            )
        record["chip_id"] = image.get("chip_id", str(record["image_id"]))
        selected.append(record)

    selected.sort(key=lambda record: record["image_id"])
    if split == "train" and int(max_train_images) > 0:
        selected = selected[: int(max_train_images)]
    return selected


def register_sp6_building(
    root=None,
    annotation_file=None,
    split_ratios=(0.8, 0.1, 0.1),
    split_seed=20260829,
    max_train_images=0,
):
    """Register deterministic train/val/test views of one paired COCO file."""
    root = os.path.abspath(
        os.path.expanduser(root or os.getenv("SP6_DATASET_ROOT", DEFAULT_SP6_ROOT))
    )
    annotation_file = annotation_file or os.getenv(
        "SP6_ANNOTATION_FILE", os.path.join(root, DEFAULT_ANNOTATION)
    )
    annotation_file = _resolve_path(root, os.path.expanduser(annotation_file))
    split_ratios = _validate_ratios(split_ratios)

    for split in _SPLITS:
        name = f"sp6_building_{split}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

        DatasetCatalog.register(
            name,
            lambda split=split, name=name: load_sp6_building_json(
                annotation_file,
                root,
                name,
                split,
                split_ratios,
                split_seed,
                max_train_images,
            ),
        )
        MetadataCatalog.get(name).set(
            # COCOEvaluator initially loads the source file, then the custom
            # height evaluator filters its API to this registered split.
            json_file=annotation_file,
            image_root=root,
            evaluator_type="coco_height",
            thing_classes=["building"],
            thing_dataset_id_to_contiguous_id={1: 0},
            height_unit="m",
            input_modality="rgb_sar",
            sp6_root=root,
            sp6_annotation_file=annotation_file,
            sp6_split=split,
            sp6_split_ratios=split_ratios,
            sp6_split_seed=int(split_seed),
            sp6_max_train_images=int(max_train_images),
        )


# Names are available to tools that import maskdino before loading a project
# config. train_net.py refreshes them with the configured local/server paths.
register_sp6_building()
