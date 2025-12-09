# -*- coding: utf-8 -*-
import gc
from abc import abstractmethod
from typing import Literal

import torch
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

from sinapsis_dfine.helpers.schemas import DFINEKeys
from sinapsis_dfine.helpers.tags import Tags


class DFINEBaseAttributes(TemplateAttributes):
    """Defines common configuration attributes for D-FINE templates.

    Attributes:
        model_path (str): The model identifier from the Hugging Face Hub or a local path to the
            model and processor files. Defaults to `"ustc-community/dfine-nano-coco"`.
        model_cache_dir (str): Directory to cache downloaded model files. Defaults to the path
            specified by the `SINAPSIS_CACHE_DIR` environment variable.
        threshold (float): The confidence score threshold (from 0.0 to 1.0) for filtering detections.
            For inference, it discards all detections below this value from the final output.
            For training, it is used on the validation dataset to filter predictions before calculating
            evaluation metrics like mAP.
        device (Literal["auto", "cuda", "cpu"]): The hardware device to run the model on.
            Defaults to `"auto"`, which automatically selects `"cuda"` if a compatible GPU
            is available, otherwise falls back to `"cpu"`.
    """

    model_path: str = "ustc-community/dfine-nano-coco"
    model_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    threshold: float
    device: Literal["auto", "cuda", "cpu"] = "auto"


class DFINEBase(Template):
    """Abstract base class for D-FINE training and inference templates.

    This class provides shared setup, teardown, and resource management logic.
    It handles the lifecycle of the model and processor, including initialization
    and memory cleanup, to ensure a consistent state for subclasses.
    """

    AttributesBaseModel = DFINEBaseAttributes
    UIProperties = UIPropertiesMetadata(
        category="D-FINE",
        output_type=OutputTypes.IMAGE,
        tags=[Tags.DFINE, Tags.IMAGE, Tags.INFERENCE, Tags.MODELS, Tags.TRAINING, Tags.OBJECT_DETECTION],
    )
    KEYS = DFINEKeys()

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.initialize()

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state. Can be overriden by subclasses for specific behaviour.
        """
        self.device = self._set_device()

    def _set_device(self) -> str:
        """Resolves the processing device based on the user's attribute setting.

        If the device attribute is set to `"auto"`, this method checks for CUDA
        availability and selects the appropriate device. Otherwise, it respects
        the user's explicit choice of `"cuda"` or `"cpu"`.

        Returns:
            str: The resolved device name, either `"cuda"` or `"cpu"`.
        """
        if self.attributes.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.attributes.device

    @abstractmethod
    def _initialize_processor(self) -> None:
        """Abstract method for loading the image processor.

        Subclasses must implement this to load their processor with the corresponding
        processor arguments.
        """

    @abstractmethod
    def _initialize_model(self) -> None:
        """Abstract method for loading the object detection model.

        Subclasses must implement this to load their specific model with the corresponding
        model arguments.
        """

    def _cleanup(self) -> None:
        """Unloads the model and processor to free up CPU/GPU memory.

        Subclasses can override this method to unload other resources as needed.
        """
        if hasattr(self, "model") and self.model is not None:
            self.model.to("cpu")
            del self.model
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor

    def reset_state(self, template_name: str | None = None) -> None:
        """Releases the heavy resources from memory and re-instantiates the template.

        Args:
            template_name (str | None, optional): The name of the template instance being reset.
                Defaults to None.
        """
        _ = template_name

        self._cleanup()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.initialize()
        self.logger.info(f"Reset template instance `{self.instance_name}`")
