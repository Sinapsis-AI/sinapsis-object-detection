# -*- coding: utf-8 -*-
import os
from typing import Any

import torch
from huggingface_hub import dataset_info
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from sinapsis_core.utils.logging_utils import sinapsis_logger
from transformers import AutoImageProcessor

from sinapsis_dfine.helpers.env_var_keys import ALLOW_UNVETTED_DATASETS
from sinapsis_dfine.helpers.schemas import AnnotationKeys

PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "bsl-1.0",
    "isc",
    "ncsa",
    "zlib",
    "postgresql",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-2.5",
    "cc-by-2.0",
    "pddl",
    "odc-by",
    "wtfpl",
    "unlicense",
    "cdla-permissive-1.0",
    "cdla-permissive-2.0",
}


def validate_hub_dataset_license(dataset_path: str) -> None:
    """Inspects a Hub dataset's metadata to verify it has a permissive license.

    This function acts as a gatekeeper, fetching card data from the Hub to check
    the license against a pre-approved allowlist. This check can be skipped for
    development by setting the `ALLOW_UNVETTED_DATASETS` environment variable to `True`.
    Local datasets are always skipped.

    Args:
        dataset_path (str): The path or Hub ID of the dataset to validate.

    Raises:
        ValueError: If the license is missing, explicitly non-commercial,
            or not found in the `PERMISSIVE_LICENSES` allowlist.
    """
    if ALLOW_UNVETTED_DATASETS:
        sinapsis_logger.warning(
            "ALLOW_UNVETTED_DATASETS is set to `True`. "
            "Skipping dataset license validation. Ensure this is not a production environment."
        )
        return

    if os.path.isdir(dataset_path):
        sinapsis_logger.info(f"Skipping license check for local dataset path: '{dataset_path}'.")
        return

    try:
        repo_info = dataset_info(dataset_path)
        license_info = repo_info.card_data.get("license")

        if not license_info:
            raise ValueError(f"Dataset '{dataset_path}' has no license specified and cannot be used.")

        licenses = (
            [license_info.lower().strip()]
            if isinstance(license_info, str)
            else [lic.lower().strip() for lic in license_info]
        )

        if any("nc" in lic or "non-commercial" in lic for lic in licenses):
            raise ValueError(f"Dataset '{dataset_path}' license {licenses} is non-commercial.")

        if not any(lic in PERMISSIVE_LICENSES for lic in licenses):
            raise ValueError(f"Dataset '{dataset_path}' license {licenses} is not in the approved allowlist.")

        sinapsis_logger.info(f"License check passed: {licenses}")

    except (RevisionNotFoundError, RepositoryNotFoundError) as e:
        sinapsis_logger.error(f"License validation failed for '{dataset_path}': {e}")
        raise ValueError(f"Could not validate dataset '{dataset_path}': repository or revision not found.") from e


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """A custom data collator for object detection tasks.

    Args:
        batch (list[dict[str, Any]]): A list of individual samples from the dataset,
            where each sample is a dictionary.

    Returns:
        dict[str, Any]: A dictionary representing the collated batch, ready to be
            passed to the model.
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }


def prepare_detection_batch(
    batch: dict[str, Any], processor: AutoImageProcessor, keys: AnnotationKeys, objects_key: str
) -> dict[str, Any]:
    """Applies the image processor's transformations to a batch of data.

    Args:
        batch (dict[str, Any]): Input dataset's batch to be processed.
        processor (AutoImageProcessor): The Hugging Face image processor instance.
        keys (AnnotationKeys): The user-define attribute with key names for accessing annotation data.
        objects_key (str): The top-level key for the dictionary that holds all annotations.

    Returns:
        dict[str, Any]: Processed data ready for training.
    """
    images = [image.convert("RGB") for image in batch["image"]]
    annotations_list = batch[objects_key]

    batch_annotations = []
    for i, annotations in enumerate(annotations_list):
        if keys.area and keys.area in annotations:
            areas = annotations[keys.area]
        else:
            areas = [box[2] * box[3] for box in annotations[keys.bbox]]
        individual_annotations = [
            {"category_id": cat, "bbox": box, "area": area}
            for cat, box, area in zip(annotations[keys.category], annotations[keys.bbox], areas)
        ]
        batch_annotations.append({"image_id": i, "annotations": individual_annotations})

    return processor(images=images, annotations=batch_annotations, return_tensors="pt")
