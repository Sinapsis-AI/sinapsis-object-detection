# -*- coding: utf-8 -*-
from types import SimpleNamespace

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor
from transformers.image_transforms import center_to_corners_format
from transformers.trainer_utils import EvalPrediction


class MAPEvaluator:
    """A class for computing object detection metrics during evaluation.

    This class is designed to be used as the `compute_metrics` function in a
    Hugging Face `Trainer`. It takes the raw model predictions and ground truth
    labels, formats them appropriately, and calculates Mean Average Precision (mAP),
    Mean Average Recall (mAR), and their per-class variants.
    """

    def __init__(self, image_processor: AutoImageProcessor, id2label: dict[int, str], threshold: float) -> None:
        """Initializes the MAPEvaluator.

        Args:
            image_processor (AutoImageProcessor): The image processor used during training,
                which is required for post-processing model outputs.
            id2label (dict[int, str]): A mapping from class IDs to human-readable label names,
                used for formatting the final metrics dictionary.
            threshold (float, optional): The confidence score threshold for filtering detections
                before metric calculation.
        """
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold

    def _collect_and_format_predictions(
        self, predictions_batches: list[tuple], label_batches: list[list[dict]]
    ) -> list[dict]:
        """Formats raw model predictions into the structure required by torchmetrics.

        This method iterates through batches and individual predictions, using the image
        processor to apply the confidence threshold and correctly scale bounding boxes.

        Args:
            predictions_batches (list[tuple]): A list of prediction batches from the model.
            label_batches (list[list[dict]]): The corresponding ground truth label batches,
                needed to access the original image sizes for scaling.

        Returns:
            list[dict]: A flat list of prediction dictionaries, one for each image, ready
                for the `torchmetrics` evaluator.
        """
        processed_predictions = []
        for pred_batch, label_batch in zip(predictions_batches, label_batches):
            logits_tuple, boxes_tuple = pred_batch[1], pred_batch[2]

            for logits, boxes, label in zip(logits_tuple, boxes_tuple, label_batch):
                target_sizes = torch.tensor(np.array([label["orig_size"]]))
                outputs_object = SimpleNamespace(
                    logits=torch.from_numpy(logits).unsqueeze(0),
                    pred_boxes=torch.from_numpy(boxes).unsqueeze(0),
                )

                post_processed_output = self.image_processor.post_process_object_detection(
                    outputs=outputs_object,
                    threshold=self.threshold,
                    target_sizes=target_sizes,
                )
                processed_predictions.append(post_processed_output[0])
        return processed_predictions

    @staticmethod
    def _collect_and_format_targets(label_batches: list[list[dict]]) -> list[dict]:
        """Formats ground truth labels into the structure required by torchmetrics.

        This method converts bounding boxes from the `center_to_corners_format` (xywh)
        to the required `xyxy` format and scales them to absolute pixel coordinates.

        Args:
            label_batches (list[list[dict]]): A list of ground truth label batches.

        Returns:
            list[dict]: A flat list of ground truth dictionaries, one for each image,
                ready for the `torchmetrics` evaluator.
        """
        processed_targets = []
        for batch in label_batches:
            for label in batch:
                height, width = label["orig_size"]
                boxes = center_to_corners_format(torch.from_numpy(label["boxes"]))
                boxes = boxes * torch.tensor([width, height, width, height])
                labels = torch.from_numpy(label["class_labels"])
                processed_targets.append({"boxes": boxes, "labels": labels})
        return processed_targets

    def __call__(self, eval_pred: EvalPrediction) -> dict:
        """The main entry point called by the `Trainer` during evaluation.

        This method orchestrates the entire evaluation process: it formats the
        predictions and targets, computes the metrics using `torchmetrics`, and
        formats the final output dictionary.

        Args:
            eval_pred (EvalPrediction): An object containing the model's predictions
                and the ground truth labels for the entire evaluation dataset.

        Returns:
            dict: A dictionary of computed metrics, including overall and per-class
                mAP and mAR scores.
        """
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids

        formatted_predictions = self._collect_and_format_predictions(predictions, label_ids)
        formatted_targets = self._collect_and_format_targets(label_ids)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(formatted_predictions, formatted_targets)
        metrics = evaluator.compute()

        if "classes" in metrics:
            map_per_class = metrics.pop("map_per_class").tolist()
            mar_100_per_class = metrics.pop("mar_100_per_class").tolist()
            classes = metrics.pop("classes").tolist()
            for cls_id, map_val, mar_val in zip(classes, map_per_class, mar_100_per_class):
                class_name = self.id2label.get(cls_id, f"class_{cls_id}")
                metrics[f"map_{class_name}"] = map_val
                metrics[f"mar_100_{class_name}"] = mar_val

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
