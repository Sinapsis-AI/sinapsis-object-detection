<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis RF-DETR
<br>
</h1>

<h4 align="center">Templates for training, inference, and model export with RF-DETR</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage example</a> •
<a href="#webapp"> 🌐 Webapp</a> •
  <a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

The **Sinapsis RF-DETR** module provides templates for training, inference, and exporting the [RF-DETR](https://blog.roboflow.com/rf-detr/) model, enabling advanced object detection tasks.

<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-rfdetr --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-rfdetr --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-rfdetr[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-rfdetr[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3>🗂️ Templates Supported</h3>

- **RFDETRExport** and **RFDETRLargeExport**: Templates for exporting the RFDETRBase and RFDETRLarge models to ONNX format.
    <details>
    <summary>Attributes</summary>

    - `model_params`(Optional): A dictionary containing model parameters for initializing the RF-DETR model (default: None). The parameters in `model_params` can include:
        - `resolution`: Defines the resolution of the input images. It must be divisible by 56.
        - `pretrain_weights`: Specifies pretrained weights path for loading a fine-tuned model.
        - `num_classes`: Specifies the number of classes for the model.
    - `export_params`(Optional): A dictionary containing the export parameters for the RF-DETR model (default: None). Key parameters that can be included in `export_params` are:
        - `output_dir`: The directory where the exported ONNX model will be saved (default: `SINAPSIS_CACHE_DIR/rfdetr`).

    </details>

- **RFDETRInference** and **RFDETRLargeInference**: Templates designed to perform inference on a set of images using the RFDETRBase and RFDETRLarge models.

    <details>
    <summary>Attributes</summary>

    - `model_params`(Optional): A dictionary containing model parameters for initializing the RF-DETR model (default: None). The parameters in `model_params` can include:
        - `resolution`: Defines the resolution of the input images. It must be divisible by 56.
        - `pretrain_weights`: Specifies pretrained weights path for loading a fine-tuned model.
        - `num_classes`: Specifies the number of classes for the model.
    - `annotations_path`(Optional): The file path to a JSON file containing annotations (default: "").
    - `threshold`(Required): A threshold for the confidence score used to filter the model's predictions (default: 0.5).

    </details>

- **RFDETRTrain** and **RFDETRLargeTrain**: Templates for training the RFDETRBase and RFDETRLarge models.

    <details>
    <summary>Attributes</summary>

    - `model_params`(Optional): A dictionary containing model parameters for initializing the RF-DETR model (default: None). The parameters in `model_params` can include:
        - `resolution`: Defines the resolution of the input images. It must be divisible by 56.
        - `num_classes`: Specifies the number of classes for the model.
    - `callback`(Required): Specifies the callback that will be used during training (default: `on_fit_epoch_end`).
    - `training_params`(Required): A dictionary containing the training parameters for the RF-DETR model (default: None). The only required argument is `dataset_dir`, which is the path to the COCO-formatted dataset directory, including `train`, `valid`, and `test` folders, each containing an `_annotations.coco.json` file.

    You can find the complete documentation for the available training parameters on the [RF-DETR GitHub](https://github.com/roboflow/rf-detr/tree/main) page.

    </details>



> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***RFDETRTrain*** use ```sinapsis info --example-template-config RFDETRTrain``` to produce an example config like:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: RFDETRTrain
  class_name: RFDETRTrain
  template_input: InputTemplate
  attributes:
    model_params:
      encoder: dinov2_windowed_small
      out_feature_indexes:
      - 2
      - 5
      - 8
      - 11
      dec_layers: 3
      two_stage: true
      projector_scale:
      - P4
      hidden_dim: 256
      sa_nheads: 8
      ca_nheads: 16
      dec_n_points: 2
      bbox_reparam: true
      lite_refpoint_refine: true
      layer_norm: true
      amp: true
      num_classes: 90
      pretrain_weights: rf-detr-base.pth
      device: cuda
      resolution: 560
      group_detr: 13
      gradient_checkpointing: false
      num_queries: 300
    callback: on_fit_epoch_end
    training_params:
      lr: 0.0001
      lr_encoder: 0.00015
      batch_size: 4
      grad_accum_steps: 4
      epochs: 100
      ema_decay: 0.993
      ema_tau: 100
      lr_drop: 100
      checkpoint_interval: 10
      warmup_epochs: 0
      lr_vit_layer_decay: 0.8
      lr_component_decay: 0.7
      drop_path: 0.0
      group_detr: 13
      ia_bce_loss: true
      cls_loss_coef: 1.0
      num_select: 300
      dataset_file: roboflow
      square_resize_div_64: true
      dataset_dir: 'path/to/dataset'
      output_dir: output
      multi_scale: true
      expanded_scales: true
      use_ema: true
      num_workers: 2
      weight_decay: 0.0001
      early_stopping: false
      early_stopping_patience: 10
      early_stopping_min_delta: 0.001
      early_stopping_use_ema: false
      tensorboard: true
      wandb: false
      project: null
      run: null
      class_names: null
```

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">📈 Training the RF-DETR model</span></strong></summary>


The **RFDETRTrain** and **RFDETRLargeTrain** templates in `sinapsis-rfdetr` simplify the process of training RF-DETR models using custom datasets. Here’s a breakdown of the training process and how to use the attributes effectively:

1. **Dataset Requirements**: Your dataset must be in **COCO format**, split into three directories: `train`, `valid`, and `test`. Each directory should contain an `_annotations.coco.json` file, which holds annotations for the respective subset, along with the corresponding image files.

The [Roboflow Universe](https://universe.roboflow.com/) provides a diverse selection of pre-labeled datasets for various use cases. To access and download a dataset, simply [create a free account account](https://app.roboflow.com/login). Additionally, [Roboflow](https://roboflow.com/annotate) allows you to create custom object detection datasets from scratch or convert existing datasets (e.g., YOLO) into COCO JSON format for training.

2. **Key Training Parameters**: The following parameters in `training_params` help configure and fine-tune the training process:
- `dataset_dir`: The path to the COCO-formatted dataset directory, containing `train`, `valid`, and `test` folders, each of which contains an `_annotations.coco.json` file.
- `epochs`: Total number of training epochs.
- `batch_size`: The number of samples per training iteration. Adjust based on available GPU memory, and use  it alongside `grad_accum_steps` to maintain the intended effective batch size.
- `grad_accum_steps`: The number of mini-batches over which gradients are accumulated.  This increases the total batch size without requiring additional memory, making it useful for GPUs with less VRAM.
- `lr`: Learning rate for optimization.
- `resume`: Allows resuming training from a saved checkpoint by specifying the checkpoint file path. This is helpful for continuing interrupted training or fine-tuning a previously trained model.
- `early_stopping`: Halts training when the model's validation performance (mAP) shows no improvement over a specified number of epochs. The stopping behavior can be adjusted using parameters like `early_stopping_patience`, `early_stopping_min_delta`, and `early_stopping_use_ema`.

**Note on memory usage**: Adjust `batch_size` and `grad_accum_steps` according to GPU VRAM. For example:
- On powerful GPUs like the A100, you can use `batch_size=16` and `grad_accum_steps=1`.
- On smaller GPUs like the T4, you may want to use `batch_size=4` and `grad_accum_steps=4`.

 Detail documentation is available on the [RF-DETR GitHub](https://github.com/roboflow/rf-detr/tree/main) page.

3. **Checkpoints**: During training, two model checkpoints will be saved: one for regular weights (`checkpoint_best_regular.pth`) and another for the Exponential Moving Average (EMA) of the model’s weights (`checkpoint_best_total.pth`), which helps improve stability and generalization.

4. **Using the Fine-Tuned Model**:
After training, load the fine-tuned model by setting the path to the pre-trained weights in  `pretrain_weights` within the `model_params` argument. Use the **RFDETRInference** template to run predictions on images.

</details>


<h2 id='example'>📚 Usage example</h2>

The following example demonstrates how to use the **RFDETRLargeTrain** template for object detection. This setup perfoms training on the RF-DETR model with a custom dataset.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: rfdetr_train
  description: Agent that runs training on a dataset with pre-trained RF-DETR model

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: RFDETRTrain
    class_name: RFDETRTrain
    template_input: InputTemplate
    attributes:
      training_params:
        dataset_dir: datasets/COCO Dataset.v37i.coco
        epochs: 20
        batch_size: 4
        grad_accum_steps: 4
        lr: 1e-4
```
</details>

This configuration defines an **agent** and a sequence of **templates** to train a **RF-DETR** model for object detection using a custom dataset.


To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="webapp">🌐 Webapp</h2>

This module includes a webapp to interact with the model.

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-object-detection.git
cd sinapsis-object-detection
```
> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!NOTE]
> Agent configuration can be modified using the `AGENT_CONFIG_PATH` environment variable. You can find the available configurations in the package's configs folder.

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-object-detection image**:
```bash
docker compose -f docker/compose.yaml build
```

2. **Start the app container**:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-rfdetr-gradio -d
```
3. **Check the status**:
```bash
docker logs -f sinapsis-rfdetr-gradio
```
4. **The logs will display the URL to access the webapp, e.g.**:
```bash
Running on local URL:  http://127.0.0.1:7860
```

**NOTE**: The url may be different, check the output of logs.

5. **To stop the app**:
```bash
docker compose -f docker/compose_apps.yaml down
```

</details>


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">💻 UV</span></strong></summary>

To run the webapp using the <code>uv</code> package manager, follow these steps:

1. **Create the virtual environment and sync the dependencies**:
```bash
uv sync --frozen
```
2. **Install the wheel**:
```bash
uv pip install sinapsis-object-detection[all] --extra-index-url https://pypi.sinapsis.tech
```
3. **Specify the correct configuration file before running the app**:
```bash
export AGENT_CONFIG_PATH=packages/sinapsis_rfdetr/src/sinapsis_rfdetr/configs/rfdetr_demo.yml
```
4. **Run the webapp**:
```bash
uv run webapps/detection_demo.py
```
5. **The terminal will display the URL to access the webapp (e.g.)**:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The URL may vary; check the terminal output for the correct address.

</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
