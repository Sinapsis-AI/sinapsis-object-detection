# -*- coding: utf-8 -*-
from typing import Any, cast

import cv2
import torch
import torchvision.transforms as T
from dfine.core import YAMLConfig
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import (
    DataContainer,
    ImageColor,
    ImagePacket,
)
from sinapsis_core.template_base.base_models import TemplateAttributeType
from sinapsis_data_visualization.helpers.detection_utils import bbox_xyxy_to_xywh

from sinapsis_dfine.helpers.load_labels import coco_id2label, objects365_id2label
from sinapsis_dfine.templates.dfine_base import DFINEBase, DFINEBaseAttributes

DetectionOutputs = list[tuple[list[list[float]], list[float], list[int]]]


class DFINEInferenceAttributes(DFINEBaseAttributes):
    """Attributes for the D-FINE inference workflow.

    Attributes:
        threshold (float): Confidence score threshold for filtering detections.
        warmup_iterations (int): Number of warm-up iterations to optimize model performance.
            The default value is 10.
        id2label (dict[int, str] | None): Mapping of class indices to label strings. Required
            if using custom weights_path. Defaults to None.
    """

    threshold: float
    warmup_iterations: int = 10
    id2label: dict[int, str] | None = None


class DFINEInference(DFINEBase):
    """
    Template designed to perform inference on an image using the D-FINE model.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: DFINEInference
          class_name: DFINEInference
          template_input: InputTemplate
          attributes:
            config_file: '/path/to/config/file/for/dfine'
            pretrained_model: null
            device: 'cuda'
            weights_path: null
            output_dir: /sinapsis/cache/dir
            threshold: 0.4
            warmup_iterations: 10
            id2label: null
    """

    AttributesBaseModel = DFINEInferenceAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.device = torch.device(self.attributes.device)
        self.model, self.postprocessor = self._initialize_model_and_postprocessor()
        self.width, self.height = self.model.decoder.eval_spatial_size
        self.transforms = self._setup_transforms()
        self._validate_inference_attributes()
        self.id2label = self._set_id2label()
        self._warmup_model()

    def _validate_inference_attributes(self) -> None:
        """Validates the attributes for inference workflows.

        Raises:
            ValueError: If neither weights_path nor pretrained_model is provided.
            ValueError: If id2label is missing for custom weights.
        """
        self._validate_config_file()

        if not (self.attributes.weights_path or self.attributes.pretrained_model):
            raise ValueError("For inference, either 'weights_path' or 'pretrained_model' must be provided.")

        if self.attributes.weights_path and not self.attributes.id2label:
            raise ValueError("When using 'weights_path', 'id2label' must be provided to map class indices to labels.")

        if self.attributes.pretrained_model:
            self._validate_pretrained_model()

        self._validate_id2label()

    def _validate_id2label(self) -> None:
        """Validates that the id2label dictionary matches the number of model classes.

        Raises:
            ValueError: If id2label does not match the model's number of classes.
        """
        if self.attributes.id2label:
            num_classes = self.model.decoder.num_classes
            if len(self.attributes.id2label) != num_classes:
                raise ValueError(
                    f"The provided id2label dictionary has {len(self.attributes.id2label)}"
                    f" entries, but the model expects {num_classes} classes."
                )

    def _initialize_model_and_postprocessor(self) -> tuple[Any, Any]:
        """Loads the model and postprocessor based on the configuration.

        Returns:
            tuple[Any, Any]: Loaded model and postprocessor instances.
        """
        cfg = YAMLConfig(self.attributes.config_file)
        cfg.yaml_cfg[self.KEYS.HGNET_V2]["pretrained"] = False
        weights_path = self._get_weights_path()

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("ema", {}).get("module", checkpoint["model"])
        cfg.model.load_state_dict(state)

        return (
            cfg.model.deploy().to(self.device),
            cfg.postprocessor.deploy().to(self.device),
        )

    def _setup_transforms(self) -> T.Compose:
        """Sets up the image transformations for preprocessing input images.

        Returns:
            T.Compose: A torchvision transformation pipeline that processes input images.
        """
        transforms = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.height, self.width)),
                T.ToTensor(),
            ]
        )
        return transforms

    def _get_weights_path(self) -> str:
        """Resolves the path to model weights, downloading if necessary.

        Returns:
            str: Path to the resolved weights file.
        """
        if self.attributes.weights_path:
            return cast(str, self.attributes.weights_path)
        return self._download_dfine_weights()

    def _set_id2label(self) -> dict[int, str]:
        """Sets the index to label mapping used for annotations.

        Returns:
            dict[int, str]: Mapping of class indices to label strings.
        """
        if self.attributes.id2label:
            return cast(dict[int, str], self.attributes.id2label)
        if self.attributes.pretrained_model["variant"] == "coco":
            return coco_id2label
        return objects365_id2label

    def _warmup_model(self) -> None:
        """Perform a warm-up for the model to optimize performance during actual inference."""
        dummy_data = torch.randn((1, 3, self.height, self.width)).to(device=self.device)
        for _ in range(self.attributes.warmup_iterations):
            with torch.inference_mode():
                _ = self.model(dummy_data)

    def _preprocess_images(self, image_packets: list[ImagePacket]) -> torch.Tensor:
        """Preprocesses a batch of images into tensors suitable for model inference.

        Args:
            image_packets (list[ImagePacket]): A list of image packets containing raw images.

        Returns:
            torch.Tensor: A tensor containing preprocessed images ready for inference.
        """
        images = torch.stack(
            [
                self.transforms(
                    cv2.cvtColor(packet.content, cv2.COLOR_RGB2BGR)
                    if packet.color_space == ImageColor.RGB
                    else packet.content
                )
                for packet in image_packets
            ]
        ).to(self.device)

        return images

    def _create_annotations(
        self, bboxes: list[list[float]], scores: list[float], labels: list[int]
    ) -> list[ImageAnnotations]:
        """Creates annotations from bounding boxes, scores, and labels.

        Args:
            bboxes (list[list[float]]): List of bounding boxes in [x, y, width, height] format.
            scores (list[float]): List of confidence scores for each detection.
            labels (list[int]): List of class labels for each detection.

        Returns:
            list[ImageAnnotations]: List of annotations, each containing a bounding box,
                confidence score, and label information.
        """
        annotations = []
        for bbox, score, label in zip(bboxes, scores, labels):
            annotations.append(
                ImageAnnotations(
                    bbox=BoundingBox(*bbox),
                    confidence_score=score,
                    label=label,
                    label_str=self.id2label.get(label),
                )
            )
        return annotations

    def _convert_outputs_to_annotations(self, processed_outputs: DetectionOutputs) -> list[list[ImageAnnotations]]:
        """Converts processed model outputs to annotations for a batch of images.

        Args:
            processed_outputs (DetectionOutputs):
                Processed outputs containing bounding boxes, confidence scores, and labels.

        Returns:
            list[list[ImageAnnotations]]: Annotations for each image in the batch.
        """
        annotations_batch = []
        for bboxes, scores, labels in processed_outputs:
            annotations = self._create_annotations(bboxes, scores, labels)
            annotations_batch.append(annotations)
        return annotations_batch

    @torch.inference_mode()
    def _run_inference(self, image_packets: list[ImagePacket]) -> list[list[ImageAnnotations]]:
        """Performs inference on a batch of images.

        Args:
            image_packets (list[ImagePacket]): List of input image packets to process.

        Returns:
            list[list[ImageAnnotations]]: A batch of annotations, with each element corresponding
                to annotations for a single image.
        """
        orig_target_sizes = torch.tensor([packet.shape[:2][::-1] for packet in image_packets], device=self.device)
        preprocessed_images = self._preprocess_images(image_packets)
        outputs = self.model(preprocessed_images)
        processed_outputs = self._postprocess_outputs(outputs, orig_target_sizes)
        annotations_batch = self._convert_outputs_to_annotations(processed_outputs)

        return annotations_batch

    def _postprocess_outputs(self, outputs: dict[str, Any], orig_target_sizes: torch.Tensor) -> DetectionOutputs:
        """Processes model outputs for a batch of images and filters predictions.

        Args:
            outputs (dict[str, Any]): Raw model outputs for a batch.
            orig_target_sizes (torch.Tensor): Original image dimensions for rescaling the
                detections.

        Returns:
            DetectionOutputs: A list containing tuples of
                bounding boxes, confidence scores, and class labels for each image.
        """
        labels, bboxes, scores = self.postprocessor(outputs, orig_target_sizes)

        batch_size = labels.size(0)
        keep_mask = scores >= self.attributes.threshold

        filtered_results = []
        for i in range(batch_size):
            valid_indices = keep_mask[i]
            valid_bboxes = bboxes[i][valid_indices].cpu().numpy()
            valid_scores = scores[i][valid_indices].tolist()
            valid_labels = labels[i][valid_indices].tolist()

            converted_bboxes = [bbox_xyxy_to_xywh(bbox) for bbox in valid_bboxes]

            filtered_results.append((converted_bboxes, valid_scores, valid_labels))

        return filtered_results

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes inference on all images in the data container.

        Args:
            container (DataContainer): Container holding input image packets.

        Returns:
            DataContainer: Container updated with generated annotations for each image.
        """
        if not container.images:
            return container

        annotations_batch = self._run_inference(container.images)
        for image_packet, annotations in zip(container.images, annotations_batch):
            image_packet.annotations.extend(annotations)

        return container
