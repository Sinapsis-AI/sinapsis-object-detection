# -*- coding: utf-8 -*-

import os
from typing import Literal

from pydantic.dataclasses import dataclass
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from sinapsis_generic_data_tools.helpers.file_downloader import download_file


@dataclass(frozen=True)
class DFINEKeys:
    """Defines constants used as keys in configurations and updates."""

    CONFIG: str = "config"
    RESUME: str = "resume"
    TUNING: str = "tuning"
    SEED: str = "seed"
    USE_AMP: str = "use_amp"
    OUTPUT_DIR: str = "output_dir"
    DEVICE: str = "device"
    HGNET_V2: str = "HGNetv2"


class DFINEBaseAttributes(TemplateAttributes):
    """Defines the general attributes required for D-FINE workflows.

    Attributes:
        config_file (str): Path to the model configuration file. It must be provided and follow
            the general structure as defined in the original D-FINE repository.
        pretrained_model (dict | None): Specifies the size and variant of
            the pretrained model, if any. Defaults to None.
        device (Literal["cpu", "cuda"]): Device to run the model ('cpu' or 'cuda').
        weights_path (str | None): Path to custom weights file, if provided. Defaults to None.
        output_dir (str): Directory for storing outputs and downloaded weights. Defaults to
            SINAPSIS_CACHE_DIR.

    """

    config_file: str
    pretrained_model: dict[Literal["size", "variant"], str] | None = None
    device: Literal["cpu", "cuda"]
    weights_path: str | None = None
    output_dir: str = str(SINAPSIS_CACHE_DIR)


class DFINEBase(Template):
    """
    Base class for shared logic in D-FINE Training and Inference workflows.

    The template validates that the configuration for the model is correct, and that
    the model has the correct size and weights to perform both inference and training.

    Raises:
        ValueError: If one of the attributes is defined incorrectly or missing for the wanted
            mode.
        FileNotFoundError: If the config file provided does not exist in the provided path.
    """

    AttributesBaseModel = DFINEBaseAttributes
    CATEGORY = "D-FINE"
    SUPPORTED_VARIANTS = ("coco", "obj365")
    SUPPORTED_SIZES = ("n", "s", "m", "l", "x")
    SUPPORTED_HGNET_BACKBONES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
    WEIGHTS_BASE_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/"
    KEYS = DFINEKeys()

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        if not os.path.exists(self.attributes.output_dir):
            os.makedirs(self.attributes.output_dir)
        if not self.attributes.output_dir.endswith("/"):
            self.attributes.output_dir += "/"

    def _validate_config_file(self) -> None:
        """Ensures the configuration file exists.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if not os.path.exists(self.attributes.config_file):
            raise FileNotFoundError(f"Config file not found: {self.attributes.config_file}")

    def _validate_pretrained_model(self) -> None:
        """Validates the pretrained model attributes.

        Raises:
            ValueError: If the model variant or size is unsupported.
        """
        model_variant = self.attributes.pretrained_model.get("variant")
        model_size = self.attributes.pretrained_model.get("size")

        if model_variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported pretrained model variant '{model_variant}'. Supported variants: {self.SUPPORTED_VARIANTS}"
            )

        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported pretrained model size '{model_size}'. Supported sizes: {self.SUPPORTED_SIZES}"
            )

        if model_variant == "obj365" and model_size == "n":
            raise ValueError("The 'n' model size is not available for 'obj365' pretrained models.")

    def _download_dfine_weights(self) -> str:
        """Downloads D-FINE weights based on the pretrained model configuration.

        Returns:
            str: Path to the downloaded D-FINE weights.
        """
        model_size = self.attributes.pretrained_model["size"]
        model_variant = self.attributes.pretrained_model["variant"]

        dfine_weights_filename = f"dfine_{model_size}_{model_variant}.pth"
        dfine_weights_path = os.path.join(self.attributes.output_dir, dfine_weights_filename)
        dfine_weights_url = self.WEIGHTS_BASE_URL + dfine_weights_filename
        download_file(dfine_weights_url, dfine_weights_path, f"D-FINE {model_variant.upper()} weights ({model_size})")

        return dfine_weights_path
