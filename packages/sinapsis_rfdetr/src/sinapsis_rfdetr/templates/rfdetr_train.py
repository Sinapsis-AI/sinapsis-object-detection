# -*- coding: utf-8 -*-
from typing import Literal

import pandas as pd
from pydantic import Field
from rfdetr.config import TrainConfig
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import (
    TemplateAttributeType,
)

from sinapsis_rfdetr.helpers.rfdetr_helpers import RFDETRKeys, initialize_output_dir
from sinapsis_rfdetr.templates.rfdetr_model_base import RFDETRModelBase, RFDETRModelLarge


class RFDETRTrain(RFDETRModelBase):
    """
    A class that handles the training process for the RF-DETR model.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: RFDETRTrain
          class_name: RFDETRTrain
          template_input: InputTemplate
          attributes:
            training_params:
                dataset_dir: '/path/to/dataset'
                epochs: 10
                batch_size: 4
                grad_accum_steps: 4
                lr: 1e-4
                early_stopping: True
                resume: 'path/to/checkpoint'
    """

    class AttributesBaseModel(RFDETRModelBase.AttributesBaseModel):
        """
        Attributes for configuring the RF-DETR training template.

        This class encapsulates the configuration parameters for training the RF-DETR model,
        including the callback to be used during training and the training parameters.

        Args:
            callback (Literal["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]):
                Specifies the callback function to be executed at specific stages during the training process.
                Default is "on_fit_epoch_end".
            training_params (TrainConfig): An instance of `TrainConfig` containing the training parameters
            for training the RF-DETR model. If not specified, default parameters will be used.

        Key parameters that can be included in `training_params` are:
            - `dataset_dir`: Path to the COCO-formatted dataset directory, containing `train`, `valid`, and `test`
              folders, each containing an `_annotations.coco.json` file.
            - `epochs`: Total number of training epochs.
            - `batch_size`: Number of samples per training iteration. Adjust based on available GPU memory and use in
              conjunction with `grad_accum_steps` to maintain the intended effective batch size.
            - `grad_accum_steps`: Number of mini-batches over which gradients are accumulated. This effectively
              increases the total batch size without requiring as much memory at once, making it useful for smaller GPUs
            - `lr`: Learning rate for optimization.
            - `resume`: Path to a saved checkpoint for resuming training.

        Note on memory usage: Adjust `batch_size` and `grad_accum_steps` based on GPU VRAM. For example:
            - On powerful GPUs like the A100, you can use `batch_size=16` and `grad_accum_steps=1`.
            - On smaller GPUs like the T4, you may want to use `batch_size=4` and `grad_accum_steps=4`.

        Complete documentation for the available training parameters and an example dataset structure can be
        found on the RF-DETR GitHub:
            https://github.com/roboflow/rf-detr/tree/main
        """

        callback: Literal["on_fit_epoch_end", "on_train_batch_start", "on_train_end"] = "on_fit_epoch_end"
        training_params: TrainConfig = Field(default_factory=dict)  # type: ignore[arg-type]

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the RF-DETR templates with the given attributes."""
        super().__init__(attributes)
        self.history: list[dict] = []
        self.attributes.training_params = initialize_output_dir(self.attributes.training_params)

    def _check_dataset_path(self) -> bool:
        """
        Verifies the existence of the `dataset_dir` in `training_params`.

        This method checks if the `dataset_dir` key exists in `self.attributes.training_params`.
        If it does, the method returns `True`; otherwise, it logs an error and returns `False`.

        Returns:
            bool: `True` if `dataset_dir` is present, `False` otherwise.
        """

        if not hasattr(self.attributes.training_params, RFDETRKeys.dataset_dir):
            self.logger.error(f"{RFDETRKeys.dataset_dir} argument must be provided in training_params attribute")
            return False
        return True

    def _history_callback(self, data: dict) -> None:
        """
        Callback function to store training history data.

        This method is invoked at the end of each epoch during training to save
        relevant metrics and training state.

        Args:
            data (dict): A dictionary containing the training metrics for the current epoch.
        """
        self.history.append(data)

    def save_metrics(self, container: DataContainer) -> None:
        """
        Converts the collected training history into a pandas DataFrame and saves it
        into the container, making the metrics accessible for analysis and visualization.

        Args:
            container (DataContainer): The data container to which the metrics will be added.
        """
        df = pd.DataFrame(self.history)
        self._set_generic_data(container, df)

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the training process for the RF-DETR model and saves the training metrics
        in the provided DataContainer.
        """
        if not self._check_dataset_path():
            return container
        self.model.callbacks[self.attributes.callback].append(self._history_callback)
        self.model.train(**self.attributes.training_params.model_dump(exclude_none=True))
        self.save_metrics(container)

        return container


class RFDETRLargeTrainAttributes(RFDETRModelLarge.AttributesBaseModel, RFDETRTrain.AttributesBaseModel):
    """
    Attributes for the RFDETRLarge train template:
    Args:
        model_params (RFDETRLargeConfig): An instance of `RFDETRLargeConfig` containing the model parameters
            for initializing the RF-DETR model. If not provided, default parameters from `RFDETRLargeConfig`
            will be used.
        callback (Literal["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]):
            Specifies the callback function to be executed at specific stages during the training process.
            Default is "on_fit_epoch_end".
        training_params (TrainConfig): An instance of `TrainConfig` containing the training parameters
            for training the RF-DETR model. If not specified, default parameters will be used.
    """


class RFDETRLargeTrain(RFDETRTrain):
    """
    A class that handles the training process for the RFDETRLarge model.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: RFDETRLargeTrain
          class_name: RFDETRLargeTrain
          template_input: InputTemplate
          attributes:
            training_params:
                dataset_dir: '/path/to/dataset'
                epochs: 10
                batch_size: 4
                grad_accum_steps: 4
                lr: 1e-4
                early_stopping: True
                resume: 'path/to/checkpoint'
    """

    MODEL_CLASS = "RFDETRLarge"
    AttributesBaseModel = RFDETRLargeTrainAttributes
