# -*- coding: utf-8 -*-

from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer
from ultralytics.utils.files import WorkingDirectory

from sinapsis_ultralytics.templates.ultralytics_base import UltralyticsBase


class UltralyticsVal(UltralyticsBase):
    """
    Template to perform ultralytics model validation. The template exports the metrics for the loaded model and stores
    them in the DataContainer.

    Usage example:

    agent:
      name: ultralytics_val
    templates:
    - template_name: UltralyticsVal
      class_name: UltralyticsVal
      template_input: null
      attributes:
        model_class: YOLO
        model: yolo11n-cls.pt
        task: detect
        verbose: 0
        validation_params:
          data: "caltech101"
          imgsz: 128
          batch: 16

    """

    class AttributesBaseModel(UltralyticsBase.AttributesBaseModel):
        """
        Attributes for UltralyticsVal Template

        Args:
            validation_params (dict[str, Any]): A dictionary containing the validation parameters for the Ultralytics
            model. If not specified, default parameters will be used.
            The full documentation for available validation parameters can be found in the Ultralytics docs:
            https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation
        """

        validation_params: dict = Field(default_factory=dict)

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes Ultralytics model validation according to the specified validation parameters.

        Args:
            container (DataContainer): A container holding the data to be processed.

        Returns:
            DataContainer: The container with updated metrics and model path after validation.
        """
        with WorkingDirectory(self.attributes.working_dir):
            validation_results = self.model.val(**self.attributes.validation_params)

            model_info = {
                "metrics": validation_results,
                "results_path": validation_results.save_dir,
            }
            self.logger.info(f"Validation results stored in {validation_results.save_dir}")
            self._set_generic_data(container, model_info)

        return container
