# -*- coding: utf-8 -*-

from typing import Any

from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from ultralytics import models
from ultralytics.engine.results import Results

from sinapsis_ultralytics.helpers.ultralytics_predict_helpers import (
    get_annotations_from_ultralytics_result,
)
from sinapsis_ultralytics.templates.ultralytics_base import UltralyticsBase


class UltralyticsPredict(UltralyticsBase):
    """
    Template used to produce inference predictions using pre-trained or saved Ultralytics models. The template takes an
    input image coming from the DataContainer and stores the annotations and inference results in the ImagePacket.

    Usage example:

    agent:
      name: test_agent
    templates:
    - template_name: UltralyticsPredict
      class_name: UltralyticsPredict
      attributes:
        model_class: YOLO
        model: yolo11n-cls.pt
        task: classify
        verbose: 0
    """

    class AttributesBaseModel(UltralyticsBase.AttributesBaseModel):
        """
        Attributes for Ultralytics Predict Template

        predict_params (Dict[str, Any]): Dict containing the parameters
        required to run inference using an Ultralytics model. If not specified,
        model predictions will be obtained using default parameters.

        Detailed description of available predict parameters can be found in:
        https://docs.ultralytics.com/modes/predict/#inference-arguments
        """

        predict_params: dict[str, Any] = Field(default_factory=dict)
        use_detections_as_sam_prompt: bool = False

    def is_sam_model(self) -> bool:
        """
        Check if the loaded model is a SAM (Segment Anything Model) or FastSAM model.

        Returns:
            bool: `True` if the model is a SAM or FastSAM model, `False` otherwise.
        """
        return isinstance(self.model, (models.SAM, models.FastSAM))

    def predict_params_updated(self, image_packet: ImagePacket) -> bool:
        """
        Updates the `predict_params` to include bounding boxes as part of the prediction parameters
        if `use_detections_as_sam_prompt` is enabled.

        Args:
            image_packet (ImagePacket): The image packet containing annotations and metadata for the image.

        Returns:
            bool: `True` if the `predict_params` were updated with bounding boxes, `False` otherwise.
        """

        if self.attributes.use_detections_as_sam_prompt:
            bbox_list = self.get_bbox_list(image_packet)
            if bbox_list:
                self.attributes.predict_params["bboxes"] = bbox_list
                return True
        return False

    @staticmethod
    def get_bbox_list(image_packet: ImagePacket) -> list[list[float]]:
        """
        Extracts the bounding boxes from the annotations in the provided `ImagePacket`.

        Args:
            image_packet (ImagePacket): The image packet containing the annotations to extract bounding boxes from.

        Returns:
            list[list[float]]: A list of bounding boxes represented by four values:
                               [x_min, y_min, x_max, y_max].
        """
        return [
            [
                ann.bbox.x,
                ann.bbox.y,
                ann.bbox.x + ann.bbox.w,
                ann.bbox.y + ann.bbox.h,
            ]
            for ann in image_packet.annotations
            if ann.bbox is not None
        ]

    def process_images(self, container: DataContainer) -> None:
        """
        Processes all images in the provided `DataContainer` by running inference and storing the results.

        Args:
            container (DataContainer): A container holding the image packets to process.

        This method processes each image in the container and updates its annotations with the
        inference results from the model.
        """
        if container.images:
            for image_packet in container.images:
                self.attributes.predict_params["source"] = image_packet.content
                if self.is_sam_model():
                    if self.predict_params_updated(image_packet):
                        predict_results = self.model.predict(**self.attributes.predict_params)
                        self.process_results(predict_results, image_packet)
                    continue

                predict_results = self.model.predict(**self.attributes.predict_params)
                self.process_results(predict_results, image_packet)

    @staticmethod
    def process_results(predict_results: list[Results], image_packet: ImagePacket) -> None:
        """
        Processes the prediction results from the model and adds the resulting annotations
        to the provided `ImagePacket`.

        Args:
            predict_results: (list[Results]) The results returned by the model's prediction function.
            image_packet (ImagePacket): The image packet where the annotations will be stored.

        This method converts the model results into annotations and appends them to the
        `annotations` attribute of the `ImagePacket`.
        """
        for result in predict_results:
            annotations = get_annotations_from_ultralytics_result(result)

            if annotations:
                if isinstance(image_packet.annotations, list):
                    image_packet.annotations.extend(annotations)
                else:
                    image_packet.annotations = annotations

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the Ultralytics model inference on the provided `DataContainer` and
        updates the image packets with prediction results.

        Args:
            container (DataContainer): The container holding the images to process.

        Returns:
            DataContainer: The input container with updated annotations after processing the images.

        If the container does not contain any images, a log message is generated and the process terminates.
        """
        if not container.images:
            self.logger.info(f"No images to process. {self.class_name} execution finished")
            return container

        self.process_images(container)
        return container
