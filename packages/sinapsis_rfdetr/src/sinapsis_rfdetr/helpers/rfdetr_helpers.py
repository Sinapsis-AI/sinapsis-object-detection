# -*- coding: utf-8 -*-

from os import makedirs
from pathlib import Path

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from rfdetr.util.coco_classes import COCO_CLASSES
from sinapsis_core.data_containers.annotations import (
    BoundingBox,
    ImageAnnotations,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from sinapsis_data_visualization.helpers.detection_utils import bbox_xyxy_to_xywh
from supervision import Detections
from supervision.dataset.formats.coco import coco_categories_to_classes
from supervision.utils.file import read_json_file


@dataclass(frozen=True)
class RFDETRKeys:
    """
    A data class to store constant keys used for RF-DETR model configuration.
    """

    rfdetr: str = "rfdetr"
    output_dir: str = "output_dir"
    pretrain_weights: str = "pretrain_weights"
    dataset_dir: str = "dataset_dir"
    categories: str = "categories"
    output: str = "output"


def get_annotations(prediction_results: Detections, annotations_path: str | None = None) -> list[ImageAnnotations]:
    """
    Generate annotations from bounding box detections for object detection
    using the RF-DETR model.

    This function converts the bounding box detections from the RF-DETR model
    output into a list of `ImageAnnotations` objects.

    Args:
        prediction_results (Detections): A results object from the RF-DETR
            model containing the detection outputs. Each detection contains
            several components in a fixed order:
            - `detection[0]`: Bounding box coordinates in `xyxy` format.
            - `detection[1]`: Segmentation mask (not used in this function).
            - `detection[2]`: Confidence score of the detection (float).
            - `detection[3]`: Class ID of the detected object.
            - `detection[4]`: Tracker ID (not used in this function).
            - `detection[5]`: Additional data related to the detection (not used in this function).
            - `detection[6]`: Metadata associated with the detection (not used in this function).

    Returns:
        list[ImageAnnotations]: A list of `ImageAnnotations` objects, each
            containing the label, bounding box coordinates (in `xywh` format), and
            the confidence score for a detected object.

    Notes:
        - The `prediction_results` object is expected to contain detections, each of which
          is an array with the structure:
          `[xyxy, mask, confidence, class_id, tracker_id, data, metadata]`.
        - The `labels` for each bounding box are fetched from the `COCO_CLASSES`
          dictionary using the `class_id` associated with the detection.
    """
    class_map = COCO_CLASSES
    if annotations_path:
        annotations_data = read_json_file(file_path=annotations_path)
        class_map = coco_categories_to_classes(coco_categories=annotations_data[RFDETRKeys.categories])

    labels = [f"{class_map[class_id]}" for class_id in prediction_results.class_id]

    annotations = []
    for idx, detection in enumerate(prediction_results):
        x, y, w, h = bbox_xyxy_to_xywh(detection[0])
        ann = ImageAnnotations(
            label=detection[3],
            label_str=labels[idx],
            bbox=BoundingBox(x, y, w, h),
            confidence_score=detection[2],
        )
        annotations.append(ann)
    return annotations


def initialize_output_dir(params: dict | BaseModel) -> dict | BaseModel:
    """
    Initializes and creates the output directory for RF-DETR results.

    This function checks if an output directory is specified in the provided parameters.
    If the output directory is not specified, it defaults to a directory. It ensures the directory exists
    by creating it if necessary.

    Args:
        params (dict | BaseModel): A dictionary or BaseModel object containing parameters for RF-DETR.
            The function expects this object to possibly include an `output_dir` key or attribute.

    Returns:
        dict | BaseModel: The same `params` object, updated with the `output_dir` key/attribute set to the
            appropriate directory path.

    """
    default_output_dir = Path(SINAPSIS_CACHE_DIR) / RFDETRKeys.rfdetr / RFDETRKeys.output
    output_dir = getattr(params, RFDETRKeys.output_dir, None)

    if output_dir is None or output_dir == RFDETRKeys.output:
        output_dir = str(default_output_dir)
    makedirs(output_dir, exist_ok=True)

    setattr(params, RFDETRKeys.output_dir, output_dir) if not isinstance(params, dict) else params.update(
        {RFDETRKeys.output_dir: output_dir}
    )

    return params
