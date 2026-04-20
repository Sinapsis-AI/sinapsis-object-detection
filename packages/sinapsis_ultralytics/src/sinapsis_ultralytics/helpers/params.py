from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict


class TrainParams(BaseModel):
    """Defines the most common parameters for an Ultralytics training session.

    This holds settings that control the dataset, model architecture, training
    duration, hardware, and optimization. These parameters correspond to the arguments
    used in the `model.train()` method. It allows passing any other valid training
    arguments per Ultralytics docs
    """

    data: str | None = None
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    save: bool = True
    device: int | str | list[int] | None = None
    workers: int = 8
    seed: int = 0
    pretrained: bool | str = True
    optimizer: str = "auto"
    lr0: float = 0.01
    project: str | None = None
    name: str | None = None
    exist_ok: bool = False
    model_config = ConfigDict(extra="allow")


class ValidationParams(BaseModel):
    """Defines the most common parameters for an Ultralytics validation session.

    This configures how a model's performance is evaluated on a dataset. It controls
    settings like the evaluation set, batch size, and metric thresholds. These
    correspond to arguments for the `model.val()`method. It allows passing any other
    valid validation arguments per Ultralytics docs.
    """

    data: str | None = None
    batch: int = 16
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.7
    device: str | None = None
    save_json: bool = False
    plots: bool = False
    split: str = "val"
    model_config = ConfigDict(extra="allow")


class ExportParams(BaseModel):
    """Defines the most common parameters for exporting an Ultralytics model.

    This controls the target format and optimizations like quantization. These
    parameters map to the `model.export()`method. It allows passing any other valid
    export arguments per Ultralytics docs.
    """

    format: str = "torchscript"
    imgsz: int = 640
    half: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = False
    opset: int | None = None
    device: str | None = None
    batch: int = 1
    model_config = ConfigDict(extra="allow")


class PredictParams(BaseModel):
    """Defines the most common parameters for running inference.

    This holds settings for running a trained model on new data, such as an image, video,
    or directory. It controls the input source, detection thresholds, and output settings.
    These parameters correspond to the `model.predict()` method. It allows passing any
    other valid inference arguments per Ultralytics docs.
    """

    source: str | np.ndarray | Path = "ultralytics/assets"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str | None = None
    save: bool = False
    save_txt: bool = False
    classes: list[int] | None = None
    agnostic_nms: bool = False
    bboxes: list[list[float]] | None = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
