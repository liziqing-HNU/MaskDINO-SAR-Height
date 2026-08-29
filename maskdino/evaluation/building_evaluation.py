import itertools
import math
import contextlib
import io
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import DatasetCatalog
from pycocotools.coco import COCO


class BuildingHeightCOCOEvaluator(COCOEvaluator):
    """COCO bbox/segm evaluation plus matched building-height metrics."""

    def __init__(
        self,
        dataset_name,
        *,
        height_score_threshold=0.5,
        height_iou_threshold=0.5,
        **kwargs,
    ):
        super().__init__(dataset_name, **kwargs)
        self._height_score_threshold = height_score_threshold
        self._height_iou_threshold = height_iou_threshold
        # SP6 uses one source COCO file with deterministic registered views.
        # Restrict the evaluator API to the requested view so train images are
        # never counted as missing validation/test predictions.
        if self._metadata.get("sp6_split") is not None:
            selected_ids = {
                record["image_id"] for record in DatasetCatalog.get(dataset_name)
            }
            if selected_ids != set(self._coco_api.getImgIds()):
                source = self._coco_api.dataset
                filtered = {
                    key: value
                    for key, value in source.items()
                    if key not in {"images", "annotations"}
                }
                filtered["images"] = [
                    image for image in source.get("images", [])
                    if image["id"] in selected_ids
                ]
                filtered["annotations"] = [
                    annotation for annotation in source.get("annotations", [])
                    if annotation["image_id"] in selected_ids
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    filtered_api = COCO()
                    filtered_api.dataset = filtered
                    filtered_api.createIndex()
                self._coco_api = filtered_api
                self._do_evaluation = "annotations" in filtered

    def process(self, inputs, outputs):
        for input_per_image, output_per_image in zip(inputs, outputs):
            prediction = {"image_id": input_per_image["image_id"]}

            if "instances" in output_per_image:
                instances = output_per_image["instances"].to(self._cpu_device)
                coco_instances = instances_to_coco_json(
                    instances, input_per_image["image_id"]
                )
                if instances.has("pred_heights"):
                    heights = instances.pred_heights.flatten().tolist()
                    for result, height in zip(coco_instances, heights):
                        result["height"] = float(height)
                prediction["instances"] = coco_instances
            if "proposals" in output_per_image:
                prediction["proposals"] = output_per_image["proposals"].to(
                    self._cpu_device
                )
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        # The parent computes standard bbox/segm AP and writes a COCO JSON. The
        # extra "height" key is retained in that file for downstream use.
        super()._eval_predictions(predictions, img_ids=img_ids)

        if not self._do_evaluation:
            return
        coco_results = list(
            itertools.chain.from_iterable(
                prediction["instances"] for prediction in predictions
            )
        )
        if not any("height" in result for result in coco_results):
            self._logger.warning(
                "Height evaluation skipped because predictions have no 'height' field."
            )
            return
        self._results["height"] = self._evaluate_height(coco_results, img_ids)

    @staticmethod
    def _json_rle_to_mask_rle(rle):
        rle = dict(rle)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return rle

    def _evaluate_height(self, coco_results, img_ids=None):
        selected_img_ids = set(
            self._coco_api.getImgIds() if img_ids is None else img_ids
        )
        predictions_by_image = defaultdict(list)
        for result in coco_results:
            height = result.get("height")
            if (
                result["image_id"] in selected_img_ids
                and result.get("score", 0.0) >= self._height_score_threshold
                and height is not None
                and math.isfinite(float(height))
                and "segmentation" in result
            ):
                predictions_by_image[result["image_id"]].append(result)

        errors = []
        valid_gt_count = 0
        for image_id in selected_img_ids:
            annotations = self._coco_api.loadAnns(
                self._coco_api.getAnnIds(imgIds=[image_id], iscrowd=None)
            )
            annotations = [
                annotation
                for annotation in annotations
                if bool(
                    annotation.get(
                        "height_valid", annotation.get("height") is not None
                    )
                )
                and annotation.get("height") is not None
                and math.isfinite(float(annotation["height"]))
                and float(annotation["height"]) >= 0
            ]
            valid_gt_count += len(annotations)

            image_predictions = sorted(
                predictions_by_image.get(image_id, []),
                key=lambda prediction: prediction["score"],
                reverse=True,
            )
            if not annotations or not image_predictions:
                continue

            prediction_rles = [
                self._json_rle_to_mask_rle(prediction["segmentation"])
                for prediction in image_predictions
            ]
            target_rles = [
                self._coco_api.annToRLE(annotation)
                for annotation in annotations
            ]
            ious = np.asarray(
                mask_util.iou(
                    prediction_rles,
                    target_rles,
                    [int(annotation.get("iscrowd", 0)) for annotation in annotations],
                )
            )

            matched_targets = set()
            for prediction_index, prediction in enumerate(image_predictions):
                available_targets = [
                    index
                    for index in range(len(annotations))
                    if index not in matched_targets
                ]
                if not available_targets:
                    break
                best_target = max(
                    available_targets,
                    key=lambda index: ious[prediction_index, index],
                )
                if (
                    ious[prediction_index, best_target]
                    < self._height_iou_threshold
                ):
                    continue
                matched_targets.add(best_target)
                errors.append(
                    float(prediction["height"])
                    - float(annotations[best_target]["height"])
                )

        if errors:
            errors = np.asarray(errors, dtype=np.float64)
            metrics = {
                "MAE": float(np.abs(errors).mean()),
                "RMSE": float(np.sqrt(np.square(errors).mean())),
                "Bias": float(errors.mean()),
                "Matched": int(errors.size),
                "Coverage": float(100.0 * errors.size / max(valid_gt_count, 1)),
            }
        else:
            metrics = {
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "Bias": float("nan"),
                "Matched": 0,
                "Coverage": 0.0,
            }

        self._logger.info(
            "Height metrics at score >= %.2f and mask IoU >= %.2f: %s",
            self._height_score_threshold,
            self._height_iou_threshold,
            metrics,
        )
        return metrics
