# -*- coding: utf-8 -*-
from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_rfdetr.helpers.rfdetr_helpers import get_annotations
from sinapsis_rfdetr.templates.rfdetr_model_base import RFDETRModelBase, RFDETRModelLarge


class RFDETRInference(RFDETRModelBase):
    """
    A class that handles the inference process for the RFDETRBase model on a batch of images.

    This class extends the `RFDETRModelBase` class and is responsible for processing
    a `DataContainer` holding image packets, running inference on each image,
    and updating the annotations with the model's predictions.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: RFDETRInference
          class_name: RFDETRInference
          template_input: InputTemplate
          attributes:
            threshold: 0.5
            model_params:
                resolution: 560
                pretrain_weights: 'path/to/checkpoint'
    """

    class AttributesBaseModel(RFDETRModelBase.AttributesBaseModel):
        """
        Attributes for the RFDETRBase inference template:

        Args:
            annotations_path (str): The file path to a JSON file containing annotations.
                If provided, this file will be used to map class IDs to labels.
            threshold (float): A threshold for the confidence score used to filter the model's predictions.
                Default is 0.5.
        """

        annotations_path: str = ""
        threshold: float = 0.5

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
                predict_results = self.model.predict(image_packet.content, threshold=self.attributes.threshold)
                annotations = get_annotations(predict_results, self.attributes.annotations_path)
                if annotations:
                    if isinstance(image_packet.annotations, list):
                        image_packet.annotations.extend(annotations)
                    else:
                        image_packet.annotations = annotations

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes inference on all images in the data container.

        Args:
            container (DataContainer): Container holding input image packets.

        Returns:
            DataContainer: Container updated with generated annotations for each image.
        """
        if not container.images:
            return container

        self.process_images(container)
        return container


class RFDETRLargeInferenceAttributes(RFDETRModelLarge.AttributesBaseModel, RFDETRInference.AttributesBaseModel):
    """
    Attributes for the RFDETRLarge inference template:

    Args:
        model_params (RFDETRLargeConfig): An instance of `RFDETRLargeConfig` containing the model parameters
            for initializing the RF-DETR model. If not provided, default parameters from `RFDETRLargeConfig`
            will be used.
        annotations_path (str): The file path to a JSON file containing annotations.
            If provided, this file will be used to map class IDs to labels.
        threshold (float): A threshold for the confidence score used to filter the model's predictions.
            Default is 0.5.
    """


class RFDETRLargeInference(RFDETRInference):
    """
    A class that handles the inference process for the RFDETRLarge model on a batch of images.

    This class extends the `RFDETRInference` class and is responsible for processing
    a `DataContainer` holding image packets, running inference on each image,
    and updating the annotations with the model's predictions.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: RFDETRLargeInference
          class_name: RFDETRLargeInference
          template_input: InputTemplate
          attributes:
            threshold: 0.5
            model_params:
                resolution: 560
                pretrain_weights: 'path/to/checkpoint'
    """

    MODEL_CLASS = "RFDETRLarge"
    AttributesBaseModel = RFDETRLargeInferenceAttributes
