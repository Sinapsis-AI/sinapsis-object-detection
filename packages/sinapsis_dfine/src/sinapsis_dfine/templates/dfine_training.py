# -*- coding: utf-8 -*-
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from pydantic import Field
from sinapsis_core.data_containers.data_packet import (
    DataContainer,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    DFineForObjectDetection,
    Trainer,
    TrainingArguments,
)

from sinapsis_dfine.helpers.dataset_utils import collate_fn, prepare_detection_batch, validate_hub_dataset_license
from sinapsis_dfine.helpers.metrics import MAPEvaluator
from sinapsis_dfine.helpers.schemas import AnnotationKeys, DatasetMappingArgs, TrainingArgs, TrainingImageSize
from sinapsis_dfine.templates.dfine_base import DFINEBase, DFINEBaseAttributes


class DFINETrainingAttributes(DFINEBaseAttributes):
    """Defines configuration attributes for the D-FINE training template.

    Extends the attributes from DFINEBaseAttributes.

    Attributes:
        training_mode (Literal["fine-tune", "from-scratch"]): Specifies the training strategy.
            `"fine-tune"` (default) loads pre-trained weights and adapts them to the new task.
            `"from-scratch"` initializes the model with random weights.
        dataset_path (str): Path to the dataset to be loaded.
        id2label (dict[int, str] | None): An optional mapping from class ID to label name.
            It's recommended to let the template infer this from the dataset. This attribute
            should only be used as a fallback if the dataset features are non-standard and
            automatic inference fails. Defaults to None.
        annotation_keys (AnnotationKeys): A configuration object that specifies the dictionary
            keys for accessing annotation data (e.g., 'bbox', 'category') within the dataset.
            This allows the template to adapt to different dataset schemas.
        validation_split_size (float): The proportion of the dataset to reserve for validation.
        mapping_args (DatasetMappingArgs): Parameters for the dataset preprocessing step.
        image_size (TrainingImageSize): The target width and height for image resizing.
        training_args (TrainingArgs): A nested configuration object for all Hugging Face
            `Trainer` hyperparameters.
        save_dir (str): Path to the directory where the fine-tuned model will be saved.
    """

    training_mode: Literal["fine-tune", "from-scratch"] = "fine-tune"
    dataset_path: str
    id2label: dict[int, str] | None = None
    annotation_keys: AnnotationKeys = Field(default_factory=AnnotationKeys)
    validation_split_size: float = 0.15
    mapping_args: DatasetMappingArgs = Field(default_factory=DatasetMappingArgs)
    image_size: TrainingImageSize = Field(default_factory=TrainingImageSize)
    training_args: TrainingArgs = Field(default_factory=TrainingArgs)
    save_dir: str


class DFINETraining(DFINEBase):
    """Fine-tunes a D-FINE object detection model on a custom dataset.

    This template handles the entire training workflow: loading and preparing the
    dataset, configuring the model and processor, setting up the Hugging Face `Trainer`,
    running the training loop, and saving the resulting model.

    Usage example:

        agent:
          name: my_test_agent
        templates:
        - template_name: DFINETraining
            class_name: DFINETraining
            template_input: InputTemplate
            attributes:
            model_path: ustc-community/dfine-nano-coco
            threshold: 0.1
            device: cuda
            dataset_path: cppe-5
            mapping_args:
                batch_size: 64
                num_proc: 1
            image_size:
                width: 480
                height: 480
            training_args:
                output_dir: "outputs"
                num_train_epochs: 3
                max_grad_norm: 0.1
                learning_rate: 5e-5
                warmup_steps: 0
                per_device_train_batch_size: 1
                dataloader_num_workers: 0
                fp16: true
                metric_for_best_model: "eval_map"
                greater_is_better: true
                load_best_model_at_end: true
                eval_strategy: "epoch"
                save_strategy: "epoch"
                save_total_limit: 2
            save_dir: "cppe-5-dfine-nano"
    """

    AttributesBaseModel = DFINETrainingAttributes

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state. Can be overriden by subclasses for specific behaviour.
        """
        torch.set_num_threads(1)
        super().initialize()
        validate_hub_dataset_license(self.attributes.dataset_path)
        self._initialize_processor()
        raw_train_dataset, raw_validation_dataset = self._load_datasets()
        self.id2label, self.label2id = self._set_label_mappings(raw_train_dataset)
        self.train_dataset, self.validation_dataset = self._apply_transformations(
            raw_train_dataset, raw_validation_dataset
        )
        self._initialize_model()
        self._initialize_trainer()

    def _initialize_processor(self) -> None:
        """Loads D-FINE image processor."""
        self.processor = AutoImageProcessor.from_pretrained(
            self.attributes.model_path,
            cache_dir=self.attributes.model_cache_dir,
            do_resize=True,
            size={"width": self.attributes.image_size.width, "height": self.attributes.image_size.height},
            use_fast=True,
        )

    def _initialize_model(self) -> None:
        """Loads the model based on the specified training_mode (fine-tune or from-scratch)."""
        if self.attributes.training_mode == "from-scratch":
            config = AutoConfig.from_pretrained(
                self.attributes.model_path,
                cache_dir=self.attributes.model_cache_dir,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            self.model = DFineForObjectDetection(config=config).to(self.device)
        else:
            self.model = DFineForObjectDetection.from_pretrained(
                self.attributes.model_path,
                cache_dir=self.attributes.model_cache_dir,
                ignore_mismatched_sizes=True,
                id2label=self.id2label,
                label2id=self.label2id,
            ).to(self.device)

    def _load_datasets(self) -> tuple[Dataset, Dataset]:
        """Loads the dataset and creates a train/validation split if needed.

        It loads a dataset from either a local path or the Hugging Face Hub. If
        no validation set is present, it will automatically split the training
        set to create one.

        Raises:
            ValueError: If a validation split is required but the split size is invalid.

        Returns:
            tuple[Dataset, Dataset]: A tuple containing the raw training and validation datasets.
        """
        dataset = load_dataset(self.attributes.dataset_path, cache_dir=SINAPSIS_CACHE_DIR)
        if "validation" not in dataset and "test" not in dataset:
            self.logger.warning(
                f"No 'validation' or 'test' split found. Splitting with ratio {self.attributes.validation_split_size}."
            )
            if self.attributes.validation_split_size <= 0.0:
                raise ValueError("'validation_split_size' must be > 0 when no validation set is provided.")

            split = dataset["train"].train_test_split(test_size=self.attributes.validation_split_size)
            return split["train"], split["test"]
        else:
            train_dataset = dataset["train"]
            validation_dataset = dataset.get("validation", dataset.get("test"))
            return train_dataset, validation_dataset

    def _set_label_mappings(self, train_dataset: Dataset) -> tuple[dict, dict]:
        """Infers class label mappings directly from the dataset's features.

        This makes the dataset the single source of truth for class labels,
        avoiding the need for manual configuration.

        Args:
            train_dataset (Dataset): The raw training dataset.

        Returns:
            tuple[dict, dict]: A tuple containing the id2label and label2id dictionaries.
        """
        try:
            class_label_feature = train_dataset.features[self.KEYS.OBJECTS][
                self.attributes.annotation_keys.category
            ].feature
            category_names = class_label_feature.names
            id2label = dict(enumerate(category_names))
        except (KeyError, AttributeError):
            self.logger.warning(
                "Could not automatically infer label mappings from the dataset features. "
                "Falling back to the `id2label` attribute."
            )
            if self.attributes.id2label is None:
                raise ValueError(
                    "Automatic label inference failed and no `id2label` attribute was provided. "
                    "Please provide an `id2label` mapping in your configuration."
                )
            id2label = self.attributes.id2label
            self.logger.info(f"Using the provided `id2label` with {len(id2label)} classes.")

        label2id = {name: id for id, name in id2label.items()}
        return id2label, label2id

    def _apply_transformations(self, train_dataset: Dataset, validation_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Applies batch transformations to the raw datasets using `.map()`.

        Args:
            train_dataset (Dataset): The raw training dataset.
            validation_dataset (Dataset): The raw validation dataset.

        Returns:
            tuple[Dataset, Dataset]: A tuple containing the processed and formatted
                training and validation datasets.
        """
        fn_kwargs = {
            "processor": self.processor,
            "keys": self.attributes.annotation_keys,
            "objects_key": self.KEYS.OBJECTS,
        }

        proc_train_dataset = train_dataset.map(
            prepare_detection_batch,
            batched=True,
            remove_columns=train_dataset.column_names,
            fn_kwargs=fn_kwargs,
            load_from_cache_file=False,
            **self.attributes.mapping_args.model_dump(),
        )
        proc_validation_dataset = validation_dataset.map(
            prepare_detection_batch,
            batched=True,
            fn_kwargs=fn_kwargs,
            load_from_cache_file=False,
            **self.attributes.mapping_args.model_dump(),
        )

        proc_train_dataset.set_format("torch")
        proc_validation_dataset.set_format("torch")

        self.logger.info(f"Train dataset size: {len(proc_train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(proc_validation_dataset)}")
        return proc_train_dataset, proc_validation_dataset

    def _initialize_trainer(self) -> None:
        """Configures and instantiates the Hugging Face `Trainer`."""
        training_args = TrainingArguments(**self.attributes.training_args.model_dump(exclude_none=True))
        map_evaluator = MAPEvaluator(
            image_processor=self.processor, id2label=self.id2label, threshold=self.attributes.threshold
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=collate_fn,
            compute_metrics=map_evaluator,
        )

    def train(self) -> None:
        """Starts the model training process using the configured `Trainer`."""
        self.logger.info("Starting model training...")
        self.trainer.train()
        self.logger.info("Training finished.")

    def save_model(self) -> None:
        """Saves the fine-tuned model and processor to the specified directory."""
        self.trainer.save_model(self.attributes.save_dir)
        self.processor.save_pretrained(self.attributes.save_dir)
        self.logger.info("Model saved successfully.")

    def _cleanup(self) -> None:
        """Unloads the model, processor, datasets and trainer to free up CPU/GPU memory."""
        super()._cleanup()
        if hasattr(self, "trainer") and self.trainer is not None:
            del self.trainer
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            self.train_dataset.cleanup_cache_files()
            del self.train_dataset
        if hasattr(self, "validation_dataset") and self.validation_dataset is not None:
            self.validation_dataset.cleanup_cache_files()
            del self.validation_dataset

    def execute(self, container: DataContainer) -> DataContainer:
        """Runs the complete training and saving workflow.

        Args:
            container (DataContainer): The input data container (empty for training).

        Returns:
            DataContainer: The data container with generic data pointing to the directory
                where the final model is saved.
        """
        self.train()
        self.save_model()
        self._set_generic_data(container, self.attributes.save_dir)
        return container
