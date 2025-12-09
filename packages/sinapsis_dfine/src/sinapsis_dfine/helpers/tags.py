# -*- coding: utf-8 -*-
from enum import Enum


class Tags(Enum):
    """An enumeration of standardized tags for categorizing templates."""

    DFINE = "dfine"
    IMAGE = "image"
    INFERENCE = "inference"
    OBJECT_DETECTION = "object_detection"
    MODELS = "models"
    TRAINING = "training"
