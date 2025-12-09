# -*- coding: utf-8 -*-

import torch
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import (
    DataContainer,
    ImagePacket,
)
from sinapsis_data_visualization.helpers.detection_utils import bbox_xyxy_to_xywh
from transformers import AutoImageProcessor, DFineForObjectDetection
from transformers.models.d_fine.modeling_d_fine import DFineObjectDetectionOutput

from sinapsis_dfine.templates.dfine_base import DFINEBase, DFINEBaseAttributes


class DFINEInferenceAttributes(DFINEBaseAttributes):
    """Defines configuration attributes for D-FINE inference template.

    Extends the attributes from DFINEBaseAttributes.

    Attributes:
        batch_size (int): The number of images to process in a single batch. If set to a non-positive value,
            all images will be processed in one batch. Defaults to 8.
    """

    batch_size: int = 8


class DFINEInference(DFINEBase):
    """Template designed to perform object detection inference on images using D-FINE.

    This template handles the entire inference workflow: configuring the model and processor,
    processing the images from the container, running the model on them, postprocessing the model's outputs
    and saving the annotations.

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
            model_path: ustc-community/dfine-small-coco
            batch_size: 16
            threshold: 0.5
            device: cuda
    """

    AttributesBaseModel = DFINEInferenceAttributes

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state. Can be overriden by subclasses for specific behaviour.
        """
        super().initialize()
        self._initialize_processor()
        self._initialize_model()

    def _initialize_processor(self) -> None:
        """Loads D-FINE image processor."""
        self.processor = AutoImageProcessor.from_pretrained(
            self.attributes.model_path, cache_dir=self.attributes.model_cache_dir, use_fast=True
        )

    def _initialize_model(self) -> None:
        """Loads the pre-trained model onto the resolved device."""
        self.model = DFineForObjectDetection.from_pretrained(
            self.attributes.model_path, cache_dir=self.attributes.model_cache_dir
        ).to(self.device)

    def _preprocess_images(self, image_packets: list[ImagePacket]) -> torch.Tensor:
        """Preprocesses a batch of images into a tensor for the model.

        Args:
            image_packets (list[ImagePacket]): A list of image packets to process.

        Returns:
            torch.Tensor: A tensor batch of images ready for inference.
        """
        images = [image_packet.content for image_packet in image_packets]
        processed_images = self.processor(images=images, return_tensors="pt", device=self.device)
        return processed_images

    def _postprocess_outputs(
        self, raw_outputs: DFineObjectDetectionOutput, orig_target_sizes: torch.Tensor
    ) -> list[dict]:
        """Processes model outputs for a batch of images and filters predictions.

        Args:
            raw_outputs (DFineObjectDetectionOutput): Raw model outputs for a batch.
            orig_target_sizes (torch.Tensor): Original image dimensions for rescaling the
                detections.

        Returns:
            list[dict]: A list containing tuples of
                bounding boxes, confidence scores, and class labels for each image.
        """
        return self.processor.post_process_object_detection(raw_outputs, self.attributes.threshold, orig_target_sizes)

    def _create_annotations(self, detection: dict) -> list[ImageAnnotations]:
        """Creates annotations from bounding boxes, scores, and labels.

        Args:
            detection (dict): Dictionary holding the detections for a single image

        Returns:
            list[ImageAnnotations]: List of annotations, each containing a bounding box,
                confidence score, and label information.
        """
        annotations = []
        for score, label, bbox in zip(
            detection[self.KEYS.SCORES], detection[self.KEYS.LABELS], detection[self.KEYS.BOXES]
        ):
            xywh_bbox = bbox_xyxy_to_xywh(bbox.cpu().numpy())
            annotations.append(
                ImageAnnotations(
                    bbox=BoundingBox(*xywh_bbox),
                    confidence_score=score.item(),
                    label=label.item(),
                    label_str=self.model.config.id2label.get(label.item()),
                )
            )
        return annotations

    @torch.inference_mode()
    def _run_inference(self, image_packets: list[ImagePacket]) -> list[list[ImageAnnotations]]:
        """Performs inference on a batch of images.

        Args:>
            image_packets (list[ImagePacket]): List of input image packets to process.

        Returns:
            list[list[ImageAnnotations]]: A batch of annotations, with each element corresponding
                to annotations for a single image.
        """
        orig_target_sizes = torch.tensor([packet.shape[:2] for packet in image_packets], device=self.device)
        processed_images = self._preprocess_images(image_packets)
        raw_outputs = self.model(**processed_images)
        detections = self._postprocess_outputs(raw_outputs, orig_target_sizes)
        annotations_batch = [self._create_annotations(det) for det in detections]

        return annotations_batch

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes inference on all images in the data container.

        Args:
            container (DataContainer): Container holding input image packets.

        Returns:
            DataContainer: Container updated with generated annotations for each image.
        """
        if not container.images:
            return container

        image_packets = container.images
        all_annotations = []

        if self.attributes.batch_size > 0:
            for i in range(0, len(image_packets), self.attributes.batch_size):
                image_batch = image_packets[i : i + self.attributes.batch_size]
                annotations_for_batch = self._run_inference(image_batch)
                all_annotations.extend(annotations_for_batch)
        else:
            all_annotations = self._run_inference(image_packets)

        for image_packet, annotations in zip(image_packets, all_annotations):
            image_packet.annotations.extend(annotations)

        return container
