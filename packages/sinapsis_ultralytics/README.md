<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis-Ultralytics
<br>
</h1>

<h4 align="center">Templates to perform training, inference, validation and export operations for different computer vision tasks using Ultralytics models.</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage example</a> •
<a href="#webapp"> 🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

The **Sinapsis Ultralytics** module provides templates for training, inference, validation and exporting models with [**Ultralytics**](https://docs.ultralytics.com/).


<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-ultralytics --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-ultralytics --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-ultralytics[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-ultralytics[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

- **UltralyticsTrain**: Template for training Ultralytics models.

    <details>
    <summary>Attributes</summary>

    - `model_class`(Required): The Ultralytics model class name. Options: `YOLO`, `YOLOWorld`, `SAM`, `FastSAM`, `NAS`, `RTDETR`.
    - `model`(Required): The model name or model path to be loaded.
    - `task`(Optional): The task to be performed by the model. Only needed by YOLO models if the task can't be obtained from the specified model name. Options: `classify`, `detect`, `segment`, `pose`, `obb` (default: `None`).
    - `verbose`(Optional): Whether to print verbose logs (default: `False`).
    - `working_dir`(Optional): The working directory for ultralytics. Ultralytics default models are downloaded to this directory (default :`SINAPSIS_CACHE_DIR/ultralytics`).
    - `datasets_dir`(Optional): The directory where the datasets are located. Ultralytics datasets ar downloaded to this directory (default: `working_dir/datasets`).
    - `runs_dir`(Optional): The directory where the training experiment artifacts are saved (default: `working_dir/runs`).
    - `checkpoint_path`(Optional): Optional explicit path to a checkpoint (pre-trained) model (default: `None`).
    - `training_params`(Optional): A dictionary containing the training parameters for the Ultralytics model. If not specified, default parameters will be used. The full documentation for available training parameters can be found in the Ultralytics [docs](https://docs.ultralytics.com/modes/train/#train-settings).

    </details>

- **UltralyticsVal**: Template for validating Ultralytics models.

    <details>
    <summary>Attributes</summary>

    - `model_class`(Required): The Ultralytics model class name. Options: `YOLO`, `YOLOWorld`, `SAM`, `FastSAM`, `NAS`, `RTDETR`.
    - `model`(Required): The model name or model path to be loaded.
    - `task`(Optional): The task to be performed by the model. Only needed by YOLO models if the task can't be obtained from the specified model name. Options: `classify`, `detect`, `segment`, `pose`, `obb` (default: `None`).
    - `verbose`(Optional): Whether to print verbose logs (default: `False`).
    - `working_dir`(Optional): The working directory for ultralytics. Ultralytics default models are downloaded to this directory (default :`SINAPSIS_CACHE_DIR/ultralytics`).
    - `datasets_dir`(Optional): The directory where the datasets are located. Ultralytics datasets ar downloaded to this directory (default: `working_dir/datasets`).
    - `runs_dir`(Optional): The directory where the training experiment artifacts are saved (default: `working_dir/runs`).
    - `checkpoint_path`(Optional): Optional explicit path to a checkpoint (pre-trained) model (default: `None`).
    - `validation_params`(Optional): A dictionary containing the validation parameters for the Ultralytics model. If not specified, default parameters will be used. The full documentation for available validation parameters can be found in the Ultralytics [docs](https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation).

    </details>

- **UltralyticsPredict**: Template for generating inference predictions with trained models.

    <details>
    <summary>Attributes</summary>

    - `model_class`(Required): The Ultralytics model class name. Options: `YOLO`, `YOLOWorld`, `SAM`, `FastSAM`, `NAS`, `RTDETR`.
    - `model`(Required): The model name or model path to be loaded.
    - `task`(Optional): The task to be performed by the model. Only needed by YOLO models if the task can't be obtained from the specified model name. Options: `classify`, `detect`, `segment`, `pose`, `obb` (default: `None`).
    - `verbose`(Optional): Whether to print verbose logs (default: `False`).
    - `working_dir`(Optional): The working directory for ultralytics. Ultralytics default models are downloaded to this directory (default :`SINAPSIS_CACHE_DIR/ultralytics`).
    - `datasets_dir`(Optional): The directory where the datasets are located. Ultralytics datasets ar downloaded to this directory (default: `working_dir/datasets`).
    - `runs_dir`(Optional): The directory where the training experiment artifacts are saved (default: `working_dir/runs`).
    - `checkpoint_path`(Optional): Optional explicit path to a checkpoint (pre-trained) model (default: `None`).
    - `predict_params`(Optional)**: A dictionary containing the inference parameters for the Ultralytics model. If not specified, default parameters will be used. The full documentation for available inference parameters can be found in the Ultralytics [docs](https://docs.ultralytics.com/modes/predict/#inference-arguments).
    - `use_detections_as_sam_prompt`(Optional): Whether to use the available detections as prompts for SAM model (default: `False`).

    </details>

- **UltralyticsExport**: Template for exporting models to deployment-ready format.

    <details>
    <summary>Attributes</summary>

    - `model_class`(Required): The Ultralytics model class name. Options: `YOLO`, `YOLOWorld`, `SAM`, `FastSAM`, `NAS`, `RTDETR`.
    - `model`(Required): The model name or model path to be loaded.
    - `task`(Optional): The task to be performed by the model. Only needed by YOLO models if the task can't be obtained from the specified model name. Options: `classify`, `detect`, `segment`, `pose`, `obb` (default: `None`).
    - `verbose`(Optional): Whether to print verbose logs (default: `False`).
    - `working_dir`(Optional): The working directory for ultralytics. Ultralytics default models are downloaded to this directory (default :`SINAPSIS_CACHE_DIR/ultralytics`).
    - `datasets_dir`(Optional): The directory where the datasets are located. Ultralytics datasets ar downloaded to this directory (default: `working_dir/datasets`).
    - `runs_dir`(Optional): The directory where the training experiment artifacts are saved (default: `working_dir/runs`).
    - `checkpoint_path`(Optional): Optional explicit path to a checkpoint (pre-trained) model (default: `None`).
    - `export_params`(Optional): A dictionary containing the export parameters for the Ultralytics model. If not specified, default parameters will be used. The full documentation for available export parameters can be found in the Ultralytics [docs](https://docs.ultralytics.com/modes/export/#arguments).

    </details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Ultralytics.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***UltralyticsPredict*** use ```sinapsis info --example-template-config UltralyticsPredict``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
- template_name: UltralyticsPredict
  class_name: UltralyticsPredict
  template_input: InputTemplate
  attributes:
    model_class: 'YOLO'
    model: null
    task: null
    verbose: false
    working_dir: /path/to/sinapsis/cache
    datasets_dir: /path/to/sinapsis/cache/datasets
    run_id: null
    checkpoint_name: last.pt
    checkpoint_path: null
    predict_params: {}
    use_detections_as_sam_prompt: false

```

<h2 id='example'>📚Usage example</h2>

The following example demonstrates how to use **Sinapsis Ultralytics** to use Ultralytics YOLO on a specific dataset. This setup loads a dataset of images, runs inferences with YOLO for detection, draws the bboxes and saves the results. Below is the full YAML configuration, followed by a breakdown of each component.
<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: inference_agent

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: FolderImageDatasetCV2
  class_name: FolderImageDatasetCV2
  template_input: InputTemplate
  attributes:
    data_dir: my_dataset

- template_name: UltralyticsPredict
  class_name: UltralyticsPredict
  template_input: FolderImageDatasetCV2
  attributes:
    model_class: YOLO
    model: yolo11n.pt
    task: "detect"
    predict_params:
      classes: [0]

- template_name: BBoxDrawer
  class_name: BBoxDrawer
  template_input: UltralyticsPredict
  attributes:
    overwrite: true
    draw_classification_label: true
    randomized_color: false

- template_name: ImageSaver
  class_name: ImageSaver
  template_input: BBoxDrawer
  attributes:
    save_dir: results
    extension: jpg
```
</details>

This configuration defines an **agent** and a sequence of **templates** to run inferences on a dataset.

> [!IMPORTANT]
> The FolderImageDatasetCV2, BBoxDrawer and ImageSaver templates correspond to the [sinapsis-data-readers](https://pypi.org/project/sinapsis-data-readers/), [sinapsis-data-visualization](https://pypi.org/project/sinapsis-data-visualization/) and [sinapsis-data-writers](https://pypi.org/project/sinapsis-data-writers/) packages respectively. If you want to use the example, please make sure you install these packages.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```


<h2 id="webapp">🌐 Webapp</h2>

The module includes two Gradio web apps for testing both training and inference Ultralytics modules.

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-ultralytics.git
cd sinapsis-ultralytics
```
> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-object-detection image**:
```bash
docker compose -f docker/compose.yaml build
```

2. **Start the app container**:

- For inference:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-ultralytics-inference -d
```

- For training:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-ultralytics-train -d
```

**NOTE**: if resources are not enough to test all tasks, please stop and restart the app.


3. **Check the status**:

- For inference:
```bash
docker logs -f sinapsis-ultralytics-inference
```

- For training:
```bash
docker logs -f sinapsis-ultralytics-train
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

3. **Run the webapp**:

- For inference:
```bash
uv run webapps/inference_app.py
```

- For training:
```bash
uv run webapps/training_app.py
```



4. **The terminal will display the URL to access the webapp, e.g.**:

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

