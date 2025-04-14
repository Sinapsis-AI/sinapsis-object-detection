# -*- coding: utf-8 -*-
import os

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header, init_image_inference
from sinapsis_core.utils.env_var_keys import GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR

model_configs = {
    "Detect": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/predict_detect.yaml",
    "Classify": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/predict_classify.yaml",
    "Segment": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/predict_segment.yaml",
    "Pose": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/predict_pose.yaml",
    "OBB": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/predict_obb.yaml",
}

with gr.Blocks(css=css_header()) as demo:
    add_logo_and_title("Sinapsis Ultralytics Inference")
    for task, config in model_configs.items():
        with gr.Tab(task):
            init_image_inference(config, "")


if __name__ == "__main__":
    demo.launch(
        share=GRADIO_SHARE_APP,
        allowed_paths=[
            SINAPSIS_CACHE_DIR,
            os.path.join(SINAPSIS_CACHE_DIR, "ultralytics/runs"),
            "runs/",
            "logs/",
        ],
    )
