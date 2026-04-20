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
from torch import Tensor
from ultralytics.engine.results import OBB, Boxes, Results
from ultralytics.utils.ops import scale_masks


def scale_image(masks: np.ndarray, im0_shape: tuple[int, int]) -> np.ndarray:
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    tensor_masks = Tensor(masks).unsqueeze(0).unsqueeze(1)
    scaled_masks = scale_masks(tensor_masks, im0_shape)
    masks = scaled_masks.numpy()

    return masks


def get_labels_from_boxes(boxes: Boxes | OBB) -> np.ndarray:
    """
    Extract labels from Ultralytics Boxes.

    Args:
        boxes (Boxes): Ultralytics Boxes object containing detections.

    Returns:
        np.ndarray: Array of labels corresponding to detected objects.
    """
    cls = boxes.cls
    if isinstance(cls, Tensor):
        labels: np.ndarray = cls.cpu().int().numpy()
    else:
        labels = cls.astype(np.int32)
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
    if result.keypoints is not None and result.keypoints.xy is not None and result.keypoints.conf is not None:
        if isinstance(result.keypoints.xy, Tensor):
            kp_array = result.keypoints.xy.cpu().numpy()[idx, :, :]
        else:
            kp_array = result.keypoints.xy[idx, :, :]
        if isinstance(result.keypoints.conf, Tensor):
            conf_array = result.keypoints.conf.cpu().numpy()[idx, :]
        else:
            conf_array = result.keypoints.conf[idx, :]

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
    return []


def get_segmentation_mask(result: Results, idx: int) -> np.ndarray | None:
    """
    Extract the segmentation mask for a specific detection.

    Args:
        result (Results): Ultralytics Results object containing detections.
        idx (int): Index of the detection.

    Returns:
        np.ndarray: Segmentation mask as a binary array.
    """
    if result.masks is not None:
        data = result.masks.data[idx]
        if isinstance(data, Tensor):
            mask: np.ndarray = data.cpu().numpy().astype(np.uint8)
        else:
            mask = data.astype(np.uint8)
        if result.masks.orig_shape is not None:
            scaled_mask = scale_image(mask, result.masks.orig_shape)
            squeezed_mask = np.squeeze(scaled_mask)
            return squeezed_mask
    return None


def get_annotations_from_bbox(result: Results) -> list[ImageAnnotations]:
    """
    Generate annotations from bounding box detections.

    Args:
        result (Results): Ultralytics Results object containing detections.

    Returns:
        list[ImageAnnotations]: List of annotations derived from bounding boxes.
    """
    annotations = []
    if result.boxes is not None:
        labels = get_labels_from_boxes(result.boxes)
        n_detections = labels.shape[0]

        if result.boxes.xyxy is not None:
            xyxy = result.boxes.xyxy
            xyxy_boxes = xyxy.cpu().numpy() if isinstance(xyxy, Tensor) else xyxy

        for idx in range(n_detections):
            label = labels[idx]
            box_confidence = float(result.boxes.conf[idx]) if result.boxes.conf is not None else 0.0
            x, y, w, h = bbox_xyxy_to_xywh(xyxy_boxes[idx])
            ann = ImageAnnotations(
                label=label,
                label_str=result.names.get(label),
                bbox=BoundingBox(x, y, w, h),
                confidence_score=box_confidence,
            )
            mask = get_segmentation_mask(result, idx)
            if mask is not None:
                ann.segmentation = Segmentation(mask=mask)

            kpt_list = get_keypoints_list(result, idx)
            if kpt_list:
                ann.keypoints = kpt_list

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
    if result.obb is not None and result.obb.xyxyxyxy is not None and result.obb.xyxy is not None:
        labels = get_labels_from_boxes(result.obb)

        obb_xyxyxyxy = result.obb.xyxyxyxy
        obb_xyxy = result.obb.xyxy
        xyxyxyxy_boxes = (
            obb_xyxyxyxy.cpu().int().numpy() if isinstance(obb_xyxyxyxy, Tensor) else obb_xyxyxyxy.astype(np.int32)
        )
        xyxy_boxes = obb_xyxy.cpu().int().numpy() if isinstance(obb_xyxy, Tensor) else obb_xyxy.astype(np.int32)

        for idx in range(labels.shape[0]):
            # Oriented Bounding Box Points in
            # [ [x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
            x1y1, x2y2, x3y3, x4y4 = xyxyxyxy_boxes[idx]

            # Aligned Bounding Box in [x,y,w,h] format
            x, y, w, h = bbox_xyxy_to_xywh(xyxy_boxes[idx])

            conf = float(result.obb.conf[idx]) if result.obb.conf is not None else 0.0
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
                confidence_score=conf,
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
    annotations = []
    if result.masks is not None:
        n_masks = result.masks.shape[0]

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
    annotations = []
    if result.probs is not None and result.names is not None:
        label = result.probs.top5[0]  # ty: ignore[unresolved-attribute]
        label_str = result.names[label]
        confidence_score = result.probs.top5conf[0]  # ty: ignore[unresolved-attribute]

        extra_labels = {}
        for pred_id, pred_conf in zip(result.probs.top5, result.probs.top5conf):  # ty: ignore[unresolved-attribute]
            extra_labels[result.names[pred_id]] = pred_conf
        annotations = [
            ImageAnnotations(
                label=label,
                label_str=label_str,
                confidence_score=float(confidence_score),
                extra_labels=extra_labels,
            )
        ]
    return annotations


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
