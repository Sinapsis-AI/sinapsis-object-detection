# -*- coding: utf-8 -*-


import os
import resource
from typing import Any, Literal

import torch
from dfine.core import YAMLConfig
from dfine.misc import dist_utils
from dfine.solver import DetSolver
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import TemplateAttributeType
from sinapsis_generic_data_tools.helpers.file_downloader import download_file

from sinapsis_dfine.templates.dfine_base import DFINEBase, DFINEBaseAttributes


@dataclass
class TrainingArtifacts:
    """
    Dataclass to contain the best_model, last_model and log_file paths produced during the D-Fine model training.
    """

    best_model: str
    last_model: str
    log_file: str


class DFINETrainingAttributes(DFINEBaseAttributes):
    """Attributes for the D-FINE training workflow. Extend the DFINEBaseAttributes.

    Attributes:
        training_mode (Literal["scratch", "tune"]): Specifies the training mode.
            'scratch' for training the model from scratch; 'tune' for fine-tuning the model
            with provided or downloaded weights.
        seed (int | None): Seed for reproducibility. Defaults to None.
        use_amp (bool): Whether to enable Automatic Mixed Precision (AMP). Defaults to False.
        print_rank (int): Rank of the process for printing logs in distributed setups. Defaults
            to 0.
        print_method (Literal["builtin", "rich"]): Method used for printing logs. Defaults to
            "builtin".
    """

    training_mode: Literal["scratch", "tune"]
    seed: int | None = None
    use_amp: bool = False
    print_rank: int = 0
    print_method: Literal["builtin", "rich"] = "builtin"


class DFINETraining(DFINEBase):
    """This module implements the training pipeline for the D-FINE model.

    It includes logic for initializing configurations, downloading weights,
    and setting up the training solver.

    Usage example:

        agent:
        name: my_test_agent
        templates:
        - template_name: InputTemplate
        class_name: InputTemplate
        attributes: {}
        - template_name: DFINETraining
        class_name: DFINETraining
        template_input: InputTemplate
        attributes:
            config_file: '/path/to/config/file/for/dfine_training'
            pretrained_model: null
            device: 'cuda'
            weights_path: null
            output_dir: /sinapsis/cache/dir
            training_mode: 'tune'
            use_amp: false
            print_rank: 0
            print_method: builtin

    """

    AttributesBaseModel = DFINETrainingAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.is_dist = self._initialize_distributed_training()
        self._validate_training_attributes()
        self._adjust_file_descriptor_limit()
        self._set_seed()
        self.cfg = self._initialize_config()
        self._prepare_hgnetv2_weights()
        self.solver = DetSolver(self.cfg)

    def _initialize_distributed_training(self) -> bool:
        """Initializes distributed training if multiple GPUs are available.

        Returns:
            bool: True if distributed training is initialized, False otherwise.
        """
        if torch.cuda.device_count() > 1:
            dist_utils.setup_distributed(
                self.attributes.print_rank,
                self.attributes.print_method,
                seed=self.attributes.seed,
            )
            return True
        return False

    @staticmethod
    def _adjust_file_descriptor_limit(target_limit: int = 65535) -> None:
        """Adjusts the file descriptor limit for the training process.

        Args:
            target_limit (int): Desired limit for file descriptors. Defaults to 65535.
        """
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = min(target_limit, hard_limit)

        if soft_limit < new_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))

    def _validate_training_attributes(self) -> None:
        """Validates the attributes for training workflows."""
        self._validate_config_file()

        if self.attributes.training_mode == "scratch":
            self._log_ignored_weights_in_scratch_mode()
        elif self.attributes.training_mode == "tune":
            self._validate_tune_mode()

    def _log_ignored_weights_in_scratch_mode(self) -> None:
        """Logs a warning if pretrained_model or weights are provided in 'scratch' mode."""
        if self.attributes.weights_path or self.attributes.pretrained_model:
            self.logger.warning("'weights_path' and 'pretrained_model' will be ignored in 'scratch' mode.")

    def _validate_tune_mode(self) -> None:
        """Validates attributes for 'tune' mode.

        Raises:
            ValueError: If required weights_path or pretrained_model is missing.
        """
        if not (self.attributes.weights_path or self.attributes.pretrained_model):
            raise ValueError("For 'tune' mode, either 'weights_path' or 'pretrained_model' must be provided.")
        if self.attributes.pretrained_model:
            self._validate_pretrained_model()

    def _initialize_config(self) -> YAMLConfig:
        """Prepares and initializes the training configuration.

        Returns:
            YAMLConfig: The initialized training configuration object.
        """
        update_dict = self._build_solver_update_dict()
        update_dict = self._prepare_dfine_weights(update_dict)
        cfg = YAMLConfig(self.attributes.config_file, **update_dict)
        cfg.yaml_cfg[self.KEYS.HGNET_V2]["local_model_dir"] = self.attributes.output_dir
        if self.attributes.training_mode == "tune":
            cfg.yaml_cfg[self.KEYS.HGNET_V2]["pretrained"] = False
        return cfg

    def _prepare_hgnetv2_weights(self) -> None:
        """Ensures the HGNetv2 backbone weights are downloaded if necessary.

        Raises:
            ValueError: If the HGNetv2 backbone name is not found in the configuration.
        """
        if not self.attributes.weights_path:
            hgnetv2_cfg = self.cfg.yaml_cfg.get(self.KEYS.HGNET_V2, {})
            hgnetv2_name = hgnetv2_cfg.get("name")
            if hgnetv2_name not in self.SUPPORTED_HGNET_BACKBONES:
                raise ValueError("HGNetv2 model name not found in configuration.")
            self._download_hgnetv2_weights(hgnetv2_name)

    def _build_solver_update_dict(self) -> dict[str, Any]:
        """Builds the dictionary to update solver configuration.

        Returns:
            dict[str, Any]: Dictionary with updated solver parameters.
        """
        return {
            self.KEYS.OUTPUT_DIR: self.attributes.output_dir,
            self.KEYS.DEVICE: self.attributes.device,
            self.KEYS.USE_AMP: self.attributes.use_amp,
            self.KEYS.SEED: self.attributes.seed,
        }

    def _prepare_dfine_weights(self, update_dict: dict[str, Any]) -> dict[str, Any]:
        """Ensures that the D-FINE weights are downloaded if necessary and updates the solver configuration.

        Args:
            update_dict (dict[str, Any]): Initial solver configuration dictionary

        Returns:
            dict[str, Any]: Updated solver configuration dictionary
        """
        if self.attributes.training_mode == "tune":
            if self.attributes.weights_path:
                update_dict[self.KEYS.TUNING] = self.attributes.weights_path
            else:
                dfine_weights_path = self._download_dfine_weights()
                update_dict[self.KEYS.TUNING] = dfine_weights_path

        return update_dict

    def _download_hgnetv2_weights(self, hgnetv2_name: Literal["B0", "B1", "B2", "B3", "B4", "B5", "B6"]) -> None:
        """Downloads HGNetv2 backbone weights.

        Args:
            hgnetv2_name (str): Name of the HGNetv2 backbone to download.
        """
        hgnetv2_weights_filename = f"PPHGNetV2_{hgnetv2_name}_stage1.pth"
        hgnetv2_weights_path = os.path.join(self.attributes.output_dir, hgnetv2_weights_filename)
        hgnetv2_weights_url = self.WEIGHTS_BASE_URL + hgnetv2_weights_filename
        download_file(hgnetv2_weights_url, hgnetv2_weights_path, f"HGNetV2_{hgnetv2_name} weights")

    def _set_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        if self.attributes.seed is not None:
            torch.manual_seed(self.attributes.seed)
            torch.cuda.manual_seed_all(self.attributes.seed)
            self.logger.info(f"Seed set to: {self.attributes.seed}")

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the D-fine model training.

        Args:
            container (DataContainer): Input data container.

        Returns:
            DataContainer: The data container containing the model training artifacts.
        """
        self.solver.fit()

        best_model_path = os.path.join(self.attributes.output_dir, "best_stg2.pth")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(self.attributes.output_dir, "best_stg1.pth")

            training_artifats = TrainingArtifacts(
                best_model=best_model_path,
                last_model=os.path.join(self.attributes.output_dir, "last.pth"),
                log_file=os.path.join(self.attributes.output_dir, "log.txt"),
            )

            self._set_generic_data(container, training_artifats)

        if self.is_dist:
            dist_utils.cleanup()

        return container
