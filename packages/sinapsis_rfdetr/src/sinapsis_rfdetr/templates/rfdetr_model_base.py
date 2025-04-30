# -*- coding: utf-8 -*-
from os import path
from typing import Literal

import rfdetr
from pydantic import Field
from rfdetr.config import RFDETRBaseConfig, RFDETRLargeConfig
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

from sinapsis_rfdetr.helpers.rfdetr_helpers import RFDETRKeys


class RFDETRModelBase(Template):
    """
    Base class for RFDETRBase model templates.

    This class defines the base functionality for creating and initializing the RFDETRBase model.
    """

    MODEL_CLASS: Literal["RFDETRBase", "RFDETRLarge"] = "RFDETRBase"
    UIProperties = UIPropertiesMetadata(category="RFDetr", output_type=OutputTypes.IMAGE)

    class AttributesBaseModel(TemplateAttributes):
        """
        Attributes for the RFDETRBase model base template.

        This class allows for specifying model parameters and customization options.

        Args:
            model_params (RFDETRBaseConfig): An instance of `RFDETRBaseConfig` containing the model parameters
            for initializing the RF-DETR model. If not provided, default parameters from `RFDETRBaseConfig`
            will be used.

            The parameters in `model_params` can include:
            - `resolution`: Defines the resolution of the input images. It must be divisible by 56.
            - `pretrain_weights`: Specifies the path to pre-trained weights, allowing you to load a
              fine-tuned model.
            - `num_classes`: Specifies the number of classes in the dataset for object detection.
            - `encoder`: Defines which encoder to use (`dinov2_windowed_small` or `dinov2_windowed_base`).
            - `hidden_dim`: The dimension of the hidden layers.
            - `sa_nheads`: The number of attention heads for self-attention.
            - `ca_nheads`: The number of attention heads for cross-attention.
            - `dec_n_points`: The number of points in the decoder for sampling.
            - `num_queries`: The number of queries for the object detection model.
            - `projector_scale`: Specifies the feature pyramid scales used in the model.

            Additional parameters and configurations can be found in the original repository's `config.py` file:
            https://github.com/roboflow/rf-detr/blob/main/rfdetr/config.py
        """

        model_params: RFDETRBaseConfig = Field(default_factory=dict)  # type: ignore[arg-type]

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the RF-DETR templates with the given attributes."""
        super().__init__(attributes)
        self._check_pretrained_path()
        self.model = self._initialize_model()

    def _check_pretrained_path(self) -> None:
        """
        Checks if the `pretrain_weights` path in `model_params` exists. If it doesn't exist,
        the method constructs a path by appending the weights file to the default directory
        (`SINAPSIS_CACHE_DIR/rfdetr`). If the constructed path exists, it updates the `model_params`
        with the new full path.
        """

        _pretrain_weights = getattr(self.attributes.model_params, RFDETRKeys.pretrain_weights, None)

        if _pretrain_weights and not path.exists(_pretrain_weights):
            _pretrain_weights = path.join(SINAPSIS_CACHE_DIR, RFDETRKeys.rfdetr, RFDETRKeys.output, _pretrain_weights)
            if path.exists(_pretrain_weights):
                setattr(self.attributes.model_params, RFDETRKeys.pretrain_weights, _pretrain_weights)

    def _initialize_model(self) -> rfdetr:
        """
        Initialize the model according to the specified model class and configuration.

        This method dynamically loads the model specified in the attributes and
        sets the necessary parameters such as resolution and pre-trained weights.

        Returns:
            rfdetr: The initialized RF-DETR model with the specified configuration.
        """
        model_class = getattr(rfdetr, self.MODEL_CLASS)
        model_params = (
            self.attributes.model_params.model_dump(exclude_none=True) if self.attributes.model_params else {}
        )
        model_instance = model_class(**model_params)

        return model_instance


class RFDETRModelLarge(RFDETRModelBase):
    """
    Base class for RFDETRLarge model base templates.

    This class defines the base functionality for creating and initializing the RFDETRLarge model.
    """

    MODEL_CLASS = "RFDETRLarge"

    class AttributesBaseModel(TemplateAttributes):
        """
        Attributes for the RFDETRLarge model base template.

        This class allows for specifying model parameters and customization options.

        Args:
            model_params (RFDETRLargeConfig): An instance of `RFDETRLargeConfig` containing the model parameters
            for initializing the RF-DETR model. If not provided, default parameters from `RFDETRLargeConfig`
            will be used.

            The parameters in `model_params` can include:
            - `resolution`: Defines the resolution of the input images. It must be divisible by 56.
            - `pretrain_weights`: Specifies the path to pre-trained weights, allowing you to load a
              fine-tuned model.
            - `num_classes`: Specifies the number of classes in the dataset for object detection.
            - `encoder`: Defines which encoder to use (`dinov2_windowed_small` or `dinov2_windowed_base`).
            - `hidden_dim`: The dimension of the hidden layers. Default is 384 for this large model.
            - `sa_nheads`: The number of attention heads for self-attention in the encoder. Default is 12.
            - `ca_nheads`: The number of attention heads for cross-attention in the decoder. Default is 24.
            - `dec_n_points`: The number of points used in the decoder for sampling. Default is 4.
            - `projector_scale`: Specifies the feature pyramid scales used in the model. Default is `["P3", "P5"]`.
            - `num_queries`: The number of queries for object detection. Default is 300.

            Additional parameters and configurations can be found in the original repository's `config.py` file:
            https://github.com/roboflow/rf-detr/blob/main/rfdetr/config.py
        """

        model_params: RFDETRLargeConfig | dict = Field(default_factory=dict)  # type: ignore[arg-type]
