# -*- coding: utf-8 -*-
import os

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.utils.env_var_keys import GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR

model_configs = {
    "detect": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/train_detect.yaml",
    "classify": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/train_classify.yaml",
    "segment": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/train_segment.yaml",
    "pose": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/train_pose.yaml",
    "obb": "packages/sinapsis_ultralytics/src/sinapsis_ultralytics/configs/train_obb.yaml",
}


def get_output_images(output_container: DataContainer) -> list[str]:
    """
    Retrieve the output images from the directory specified in the container.

    Args:
        output_container (str): The path to the container's output directory.

    Returns:
        list: A list of image file paths (PNG and JPG) within the output directory.
    """
    output_dir = output_container.generic_data.get("UltralyticsTrain").get("trained_model_path")
    return [os.path.join(output_dir, file) for file in os.listdir(output_dir) if (file.endswith(("png", "jpg")))]


def model_train(specific_task: str) -> list:
    """
    Train a model for the specified task.

    This function resolves the training configuration based on the task and uses it to initialize an agent
    to perform training and attach important metadata to a container.

    Args:
        specific_task (str): The task name for which the model will be trained
                    ('Detect', 'Segment', 'Classify', 'Pose', 'OBB').

    Returns:
        tuple: A tuple containing:
            - output_container: The resulting container with relevant metadata.
            - str: A success message indicating that the training was completed.
    """

    agent = generic_agent_builder(specific_task)
    output_container = agent(DataContainer())
    images = get_output_images(output_container)
    return images


def select_config(specific_task: str) -> str:
    """
    Retrieve the configuration file path for a given task.

    Args:
        specific_task (str): The task name ('detect', 'segment', 'classify', 'pose', 'obb').

    Returns:
        str: The file path to the corresponding configuration YAML file.
    """
    return model_configs[specific_task]


with gr.Blocks() as demo:
    add_logo_and_title("Sinapsis Training app")
    for task, config in model_configs.items():
        with gr.Tab(task):
            container = gr.State(DataContainer())
            task_btn = gr.Button("Train")
            gallery = gr.Gallery(selected_index=1)
            task_btn.click(model_train, inputs=gr.Textbox(config, visible=False), outputs=[gallery])

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
