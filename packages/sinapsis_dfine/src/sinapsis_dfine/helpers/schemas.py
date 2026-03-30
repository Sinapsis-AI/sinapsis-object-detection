# -*- coding: utf-8 -*-
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class DFINEKeys:
    """Centralizes string constants for object detection data keys with D-FINE."""

    SCORES: str = "scores"
    LABELS: str = "labels"
    BOXES: str = "boxes"
    OBJECTS: str = "objects"


class TrainingArgs(BaseModel):
    """Defines the most common training args for training.

    For a complete list of all possible arguments and their detailed descriptions,
    refer to the official Hugging Face documentation:
    https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    """

    output_dir: str = "trainer_output"
    eval_strategy: Literal["no", "steps", "epoch"] = "no"
    eval_steps: int | float | None = None
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int | None = None
    eval_delay: float | None = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    logging_strategy: Literal["no", "steps", "epoch"] = "no"
    logging_steps: int | float = 500
    save_strategy: Literal["no", "steps", "epoch", "best"] = "steps"
    save_steps: int | float = 500
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    dataloader_num_workers: int = 0
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    save_total_limit: int | None = None
    report_to: str | list[str] = "none"
    remove_unused_columns: bool = False
    eval_do_concat_batches: bool = False
    resume_from_checkpoint: str | None = None
    model_config = ConfigDict(extra="allow")


class TrainingImageSize(BaseModel):
    """Defines the target image size for resizing during training.

    Attributes:
        width (int): The target width for resizing.
        height (int): The target height for resizing.
    """

    width: int = 640
    height: int = 640


class DatasetMappingArgs(BaseModel):
    """Configuration for the dataset mapping (`.map()`) process.

    Attributes:
        batch_size (int): The batch size for applying transformations. A larger
            size can speed up preprocessing but requires more RAM.
        num_proc (int): The number of CPU processes to use for mapping. Set to > 1
            for parallel processing, which can significantly reduce preprocessing time.
            Defaults to 0 (no multiprocessing).
    """

    batch_size: int = 16
    num_proc: int = 0


class AnnotationKeys(BaseModel):
    """Defines the key names for accessing annotation data within a dataset sample.

    This allows the template to adapt to different dataset naming conventions
    without changing the core transformation logic.

    Attributes:
        bbox (str): The dictionary key for the bounding box annotations. Defaults to "bbox".
        category (str): The dictionary key for the category/class label annotations.
            Defaults to "category".
        area (str): The dictionary key for the bounding box area. If not provided, area
            will be calculated from the bbox. Defaults to "area".
    """

    bbox: str = "bbox"
    category: str = "category"
    area: str = "area"
    model_config = ConfigDict(extra="forbid")
