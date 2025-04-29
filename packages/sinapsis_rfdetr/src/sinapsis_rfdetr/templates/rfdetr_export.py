# -*- coding: utf-8 -*-
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import (
    TemplateAttributeType,
)

from sinapsis_rfdetr.helpers.rfdetr_helpers import initialize_output_dir
from sinapsis_rfdetr.templates.rfdetr_model_base import RFDETRModelBase, RFDETRModelLarge


class RFDETRExport(RFDETRModelBase):
    """
    A class that handles the export process for the RFDETRBase trained model to ONNX format.

    This class is used to configure and execute the export of a trained RFDETRBase model to the ONNX format.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: RFDETRExport
          class_name: RFDETRExport
          template_input: InputTemplate
          attributes:
            model_params:
                pretrain_weights: 'path/to/checkpoint'
            export_params:
                output_dir: 'path/to/save/export/model'
    """

    class AttributesBaseModel(RFDETRModelBase.AttributesBaseModel):
        """
        Attributes for configuring the RF-DETR export template.

        Args:
            export_params (dict[str, Any]):
                A dictionary containing the export parameters for the RF-DETR model. If not specified, default
                parameters will be used.

        Key parameters that can be included in `export_params` are:
            - `output_dir`: The directory where the exported ONNX model will be saved. Defaults to
              `SINAPSIS_CACHE_DIR/rfdetr`.
        """

        export_params: dict = Field(default_factory=dict)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the RF-DETR templates with the given attributes."""
        super().__init__(attributes)
        self.attributes.export_params = initialize_output_dir(self.attributes.export_params)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the export process for the RF-DETR model to ONNX format.

        Args:
            container (DataContainer): Container holding input image packets.

        Returns:
            DataContainer: Container updated with generated annotations for each image.
        """
        self.model.export(**self.attributes.export_params)
        return container


class RFDETRLargeExportAttributes(RFDETRModelLarge.AttributesBaseModel, RFDETRExport.AttributesBaseModel):
    """
    Attributes for the RFDETRLarge inference template:
    Args:
        model_params (RFDETRLargeConfig): An instance of `RFDETRLargeConfig` containing the model parameters
            for initializing the RF-DETR model. If not provided, default parameters from `RFDETRLargeConfig`
            will be used.
        export_params (dict[str, Any]): A dictionary containing the export parameters for the RF-DETR model.
            If not specified, default parameters will be used.
    """


class RFDETRLargeExport(RFDETRExport):
    """
    A class that handles the export process for the RFDETRLarge trained model to ONNX format.

    This class is used to configure and execute the export of a trained RFDETRLarge model to the ONNX format.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: RFDETRLargeExport
          class_name: RFDETRLargeExport
          template_input: InputTemplate
          attributes:
            model_params:
                pretrain_weights: 'path/to/checkpoint'
            export_params:
                output_dir: 'path/to/save/export/model'
    """

    MODEL_CLASS = "RFDETRLarge"
    AttributesBaseModel = RFDETRLargeExportAttributes
