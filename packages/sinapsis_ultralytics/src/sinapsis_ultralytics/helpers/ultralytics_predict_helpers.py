# -*- coding: utf-8 -*-

import numpy as np
from sinapsis_core.data_containers.annotations import (
    BoundingBox,
    ImageAnnotations,
    KeyPoint,
    OrientedBoundingBox,
    Segmentation,
)
from sinapsis_data_visualization.helpers.detection_utils import bbox_xyxy_to_xywh
from ultralytics.engine.results import Boxes, Results
from ultralytics.utils.ops import scale_image


def get_labels_from_boxes(boxes: Boxes) -> np.ndarray:
    """
    Extract labels from Ultralytics Boxes.

    Args:
        boxes (Boxes): Ultralytics Boxes object containing detections.

    Returns:
        np.ndarray: Array of labels corresponding to detected objects.
    """
    labels: np.ndarray = boxes.cls.cpu().int().numpy()
    return labels


def get_keypoints_list(result: Results, idx: int) -> list[KeyPoint]:
    """
    Extract keypoints for specific detection from an Ultralytics result.

    Args:
        result (Results): Ultralytics Results object containing detections.
        idx (int): Index of the detection.

    Returns:
        list[KeyPoint]: List of keypoints for the detection.
    """
    kp_array = result.keypoints.xy.cpu().numpy()[idx, :, :]
    conf_array = result.keypoints.conf.cpu().numpy()[idx, :]

    n_keypoints = kp_array.shape[0]
    keypoints = []
    for idx_kpt in range(n_keypoints):
        kpt = KeyPoint(
            x=float(kp_array[idx_kpt, 0]),
            y=float(kp_array[idx_kpt, 1]),
            score=float(conf_array[idx_kpt]),
        )
        keypoints.append(kpt)
    return keypoints


def get_segmentation_mask(result: Results, idx: int) -> np.ndarray:
    """
    Extract the segmentation mask for a specific detection.

    Args:
        result (Results): Ultralytics Results object containing detections.
        idx (int): Index of the detection.

    Returns:
        np.ndarray: Segmentation mask as a binary array.
    """
    mask: np.ndarray = result.masks.data[idx].cpu().numpy().astype(np.uint8)
    scaled_mask = scale_image(mask, result.masks.orig_shape)
    squeezed_mask = np.squeeze(scaled_mask)
    return squeezed_mask


def get_annotations_from_bbox(result: Results) -> list[ImageAnnotations]:
    """
    Generate annotations from bounding box detections.

    Args:
        result (Results): Ultralytics Results object containing detections.

    Returns:
        list[ImageAnnotations]: List of annotations derived from bounding boxes.
    """
    annotations = []
    labels = get_labels_from_boxes(result.boxes)
    n_detections = labels.shape[0]

    xyxy_boxes = result.boxes.xyxy.cpu().numpy()

    for idx in range(n_detections):
        label = labels[idx]
        box_confidence = result.boxes.conf[idx]
        x, y, w, h = bbox_xyxy_to_xywh(xyxy_boxes[idx])
        ann = ImageAnnotations(
            label=label,
            label_str=result.names.get(label),
            bbox=BoundingBox(x, y, w, h),
            confidence_score=box_confidence,
        )
        if result.masks:
            mask = get_segmentation_mask(result, idx)
            ann.segmentation = Segmentation(mask=mask)
        if result.keypoints:
            ann.keypoints = get_keypoints_list(result, idx)

        annotations.append(ann)
    return annotations


def get_annotations_from_oriented_bbox(result: Results) -> list[ImageAnnotations]:
    """
    Generate annotations from oriented bounding box detections.

    Args:
        result (Results): Ultralytics Results object containing detections.

    Returns:
        list[ImageAnnotations]: List of annotations with oriented bounding boxes.
    """
    annotations = []
    labels = get_labels_from_boxes(result.obb)

    xyxyxyxy_boxes = result.obb.xyxyxyxy.cpu().int().numpy()
    xyxy_boxes = result.obb.xyxy.cpu().int().numpy()

    for idx in range(labels.shape[0]):
        # Oriented Bounding Box Points in
        # [ [x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
        x1y1, x2y2, x3y3, x4y4 = xyxyxyxy_boxes[idx]

        # Aligned Bounding Box in [x,y,w,h] format
        x, y, w, h = bbox_xyxy_to_xywh(xyxy_boxes[idx])

        ann = ImageAnnotations(
            label=labels[idx],
            label_str=result.names.get(labels[idx]),
            oriented_bbox=OrientedBoundingBox(
                x1y1[0],
                x1y1[1],
                x2y2[0],
                x2y2[1],
                x3y3[0],
                x3y3[1],
                x4y4[0],
                x4y4[1],
            ),
            bbox=BoundingBox(x, y, w, h),
            confidence_score=result.obb.conf[idx],
        )

        annotations.append(ann)
    return annotations


def get_annotations_from_masks(result: Results) -> list[ImageAnnotations]:
    """
    Generate annotations from segmentation masks.

    Args:
        result (Results): Ultralytics Results object containing segmentation masks.

    Returns:
        list[ImageAnnotations]: List of annotations with segmentation masks.
    """
    n_masks = result.masks.shape[0]
    annotations = []
    for i in range(n_masks):
        mask = get_segmentation_mask(result, i)
        annotations.append(ImageAnnotations(segmentation=Segmentation(mask=mask)))

    return annotations


def get_annotations_from_probs(result: Results) -> list[ImageAnnotations]:
    """
    Generate annotations from classification probabilities.

    Args:
        result (Results): Ultralytics Results object containing classification results.

    Returns:
        list[ImageAnnotations]: List of annotations with classification labels and confidence scores.
    """
    label = result.probs.top5[0]
    label_str = result.names[label]
    confidence_score = result.probs.top5conf[0]

    extra_labels = {}
    for pred_id, pred_conf in zip(result.probs.top5, result.probs.top5conf):
        extra_labels[result.names[pred_id]] = pred_conf
    return [
        ImageAnnotations(
            label=label,
            label_str=label_str,
            confidence_score=confidence_score,
            extra_labels=extra_labels,
        )
    ]


def get_annotations_from_ultralytics_result(
    results: Results,
) -> list[ImageAnnotations] | None:
    """Get Annotations from an ultralytics Results object.

    Args:
        result (Results): ultralytics Results object.

    Returns:
        list[ImageAnnotations] | None: list of image annotations. If no annotations are found, return None.
    """
    if results.boxes:
        return get_annotations_from_bbox(results)

    if results.obb:
        return get_annotations_from_oriented_bbox(results)

    if results.masks:
        return get_annotations_from_masks(results)

    if results.probs:
        return get_annotations_from_probs(results)

    return None
