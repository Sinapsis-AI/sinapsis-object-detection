# -*- coding: utf-8 -*-

from os import makedirs, path
from pathlib import Path
from typing import Any, Literal

from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from ultralytics import models, settings
from ultralytics.utils.files import WorkingDirectory


class UltralyticsBase(Template):
    """
    Base template for Ultralytics models.

    The base template includes functionality to initialize ultralytics models associated with classification, detection,
    segmentation, pose and obb tasks. Initialized models can be used for training, prediction, validation or export
    operations.
    """

    CATEGORY = "Ultralytics"

    class AttributesBaseModel(TemplateAttributes):
        """
        Attributes for Ultralytics Base Class:

            model_class (Literal["YOLO", "YOLOWorld", "SAM", "FastSAM", "NAS", "RTDETR"]): The Ultralytics model
            class name.
            model (str | Path): The model name or model path to be loaded.
                Check https://docs.ultralytics.com/models/ to consult the full list of models supported by ultralytics.
            task (Literal["classify", "detect", "segment", "pose", "obb"] | None): The task to be performed by the
                model. Only needed by YOLO models if the task can't be obtained from the specified model name. Defaults
                to None.
            verbose (bool): Whether to print verbose logs. Defaults to False.
            working_dir (str | Path): The working directory for ultralytics. Ultralytics default models are
                downloaded to this directory. Defaults to SINAPSIS_CACHE_DIR/ultralytics.
            datasets_dir (str | Path): The directory where the datasets are located. Ultralytics datasets are
                downloaded to this directory. Defaults to working_dir/datasets.
            runs_dir (str | Path): The directory where the training experiment artifacts are saved.
                Defaults to working_dir/runs.
            checkpoint_path (str | Path | None): Optional explicit path to checkpoint (pre-trained) model.
                Defaults to None.
        """

        model_class: Literal["YOLO", "YOLOWorld", "SAM", "FastSAM", "NAS", "RTDETR"]
        model: str | Path
        task: Literal["classify", "detect", "segment", "pose", "obb"] | None = None
        verbose: bool = False
        working_dir: str | Path = path.join(SINAPSIS_CACHE_DIR, "ultralytics")
        datasets_dir: str | Path = path.join(working_dir, "datasets")
        runs_dir: str | Path = path.join(working_dir, "runs")
        checkpoint_path: str | Path | None = None

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the Ultralytics templates with the given attributes."""
        super().__init__(attributes)
        self.model_class = getattr(models, self.attributes.model_class)

        self._update_ultralytics_settings()
        self.model_params = self._get_model_params_dict()

        self.initialize_working_directory()

        with WorkingDirectory(self.attributes.working_dir):
            self._initialize_model()

    def initialize_working_directory(self) -> None:
        """
        Verify if the working directory exists if not it is created.
        """

        if not path.exists(self.attributes.working_dir):
            makedirs(self.attributes.working_dir, exist_ok=True)

    def _initialize_model(self) -> None:
        """
        Initialize model according to the specified model class. If a checkpoint path is set
        the model is initialized using the loaded pre-trained weights.
        """
        self.model: models = self.model_class(**self.model_params)

        if self.attributes.checkpoint_path:
            self._load_pretrained_model()

    def _get_model_params_dict(self) -> dict[str, Any]:
        """Returns the parameters to initialize the Ultralytics model."""
        return dict(
            self.attributes.model_dump(
                exclude_unset=True,
                exclude_defaults=True,
                exclude_none=True,
                include={"model", "task", "verbose"},
            )
        )

    def _update_ultralytics_settings(self) -> None:
        """Updates the settings for the Ultralytics library."""
        settings.update({"runs_dir": self.attributes.runs_dir, "datasets_dir": self.attributes.datasets_dir})

    def _load_pretrained_model(self) -> None:
        """Loads a pretrained model from the given checkpoint"""
        self.logger.info(
            f"Loading from checkpoint: {self.attributes.checkpoint_path}",
        )
        self.model.load(self.attributes.checkpoint_path)
