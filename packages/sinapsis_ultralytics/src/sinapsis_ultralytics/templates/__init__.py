# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_ultralytics.templates"

_template_lookup = {
    "UltralyticsTrain": f"{_root_lib_path}.ultralytics_train",
    "UltralyticsPredict": f"{_root_lib_path}.ultralytics_predict",
    "UltralyticsVal": f"{_root_lib_path}.ultralytics_val",
    "UltralyticsExport": f"{_root_lib_path}.ultralytics_export",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)

    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
