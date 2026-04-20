# -*- coding: utf-8 -*-
from enum import Enum


class Tags(str, Enum):
    EXPORT = "export"
    IMAGE = "image"
    INFERENCE = "inference"
    MODELS = "models"
    OBJECT_DETECTION = "object_detection"
    ONNX = "onnx"
    RFDETR = "rfdetr"
    TRAINING = "training"
