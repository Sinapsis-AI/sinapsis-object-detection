# -*- coding: utf-8 -*-
import gradio as gr
from helpers.download_configs import download_configs_folder
from sinapsis.webapp.agent_gradio_helper import (
    add_logo_and_title,
    css_header,
    init_image_inference,
)
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

CONFIG_PATH = AGENT_CONFIG_PATH or "packages/sinapsis-dfine/src/sinapsis_dfine/configs/demo.yml"


def create_demo() -> gr.Blocks:
    """Creates a Gradio interface.

    Returns:
        gr.Blocks: A configured Gradio Blocks interface ready to launch.
    """
    with gr.Blocks(css=css_header(), title="Sinapsis D-FINE") as demo:
        add_logo_and_title("Sinapsis D-FINE: Object Detection")
        init_image_inference(CONFIG_PATH, app_message="""Detect objects in images using the D-FINE model.""")

    return demo


if __name__ == "__main__":
    download_configs_folder(path="artifacts/configs")
    demo = create_demo()
    demo.launch(share=GRADIO_SHARE_APP)
