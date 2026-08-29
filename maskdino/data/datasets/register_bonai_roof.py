import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


DEFAULT_BONAI_ROOT = "/home/lzq/dataset/BONAI"
_SPLITS = {
    "bonai_roof_train": (
        "coco_building_bbox_roof_mask/instances_train.json",
        "trainval/images",
    ),
    "bonai_roof_val": (
        "coco_building_bbox_roof_mask/instances_val.json",
        "test",
    ),
}


def register_bonai_roof(root=None):
    """Register BONAI building-bbox/roof-mask instance datasets."""
    root = os.path.abspath(
        os.path.expanduser(
            root or os.getenv("BONAI_DATASET_ROOT", DEFAULT_BONAI_ROOT)
        )
    )

    for name, (json_relative, image_relative) in _SPLITS.items():
        json_file = os.path.join(root, json_relative)
        image_root = os.path.join(root, image_relative)

        if name in DatasetCatalog.list():
            registered_root = MetadataCatalog.get(name).get("bonai_root")
            if registered_root == root:
                continue
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

        register_coco_instances(
            name,
            {
                "bonai_root": root,
                "thing_classes": ["building"],
                "input_modality": "rgb",
                "annotation_target": "building_bbox+roof_mask",
                "bbox_target": "building_bbox",
                "mask_target": "roof_mask",
            },
            json_file,
            image_root,
        )


register_bonai_roof()
