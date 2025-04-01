# -*- coding: utf-8 -*-
import json

from huggingface_hub import hf_hub_download
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR


def load_label_mapping(filename: str, repo_id: str = "huggingface/label-files") -> dict[int, str]:
    """Loads a label mapping dictionary from a Hugging Face Hub repository.

    Args:
        filename (str): Name of the JSON file containing the label mappings
        repo_id (str, optional): Hugging Face repository ID containing label files. Defaults to
            "huggingface/label-files".

    Returns:
        dict[int, str]: Dictionary mapping integer class IDs to their corresponding class names.
    """
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=SINAPSIS_CACHE_DIR)
    with open(file_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {int(k): v for k, v in label_map.items()}


coco_id2label = load_label_mapping("coco-detection-mmdet-id2label.json")

objects365_id2label = load_label_mapping("object365-id2label.json")

__all__ = ["coco_id2label", "objects365_id2label"]
