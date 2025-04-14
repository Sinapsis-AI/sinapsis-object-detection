<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis Object Detection
<br>
</h1>

<h4 align="center">Mono repo with packages for training and inference with various models for advanced object detection tasks.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#packages">📦 Packages</a> •
<a href="#webapp">🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>


<h2 id="installation"> 🐍  Installation </h2>

> [!IMPORTANT]
> Sinapsis projects requires Python 3.10 or higher.
>

This repo includes packages for performing object detection using different models:

* <code>sinapsis-dfine</code>
* <code>sinapsis-rfdetr</code>
* <code>sinapsis-ultralytics</code>

Install using your package manager of choice. We strongly encourage the use of <code>uv</code>. If you need to install <code>uv</code> please see the [official documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-dfine --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-dfine --extra-index-url https://pypi.sinapsis.tech
```
**Replace `sinapsis-dfine` with the name of the package you intend to install.**



> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-dfine[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-dfine[all] --extra-index-url https://pypi.sinapsis.tech
```
**Be sure to substitute `sinapsis-dfine` with the appropriate package name.**

> [!TIP]
> You can also install all the packages within this project:

```bash
  uv pip install sinapsis-object-detection[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="packages">📦 Packages</h2>

This repository is organized into modular packages, each built for integration with different object detection models. These packages offer ready-to-use templates for training and performing inference with advanced models. Below is an overview of the available packages:

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis D-FINE</span></strong></summary>

The package provides templates for **training** and **inference** with the D-FINE model, enabling advanced **object detection** tasks. It includes:

- **DFINETraining**: A template that implements the training pipeline for the D-FINE model, including logic for initializing configurations, downloading weights, and setting up the training solver.
- **DFINEInference**: A template designed for performing inference on a set of images using the different D-FINE architectures available.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-object-detection/blob/main/packages/sinapsis_dfine/README.md).

</details>

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis RF-DETR</span></strong></summary>

The package provides templates for **training**, **inference**, and **export** with the RF-DETR model, enabling advanced **object detection** tasks. It includes:

- **RFDETRExport** and **RFDETRLargeExport**: Templates for exporting the RFDETRBase and RFDETRLarge models to ONNX format.
- **RFDETRInference** and **RFDETRLargeInference**: Templates designed to perform inference on a set of images using the RFDETRBase and RFDETRLarge models.
- **RFDETRTrain** and **RFDETRLargeTrain**: Templates for training the RFDETRBase and RFDETRLarge models.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-object-detection/blob/main/packages/sinapsis_rfdetr/README.md).

</details>

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis Ultralytics</span></strong></summary>

The package provides templates for **training**, **inference**, **validation**, and **exporting** models with Ultralytics. It includes:

- **UltralyticsTrain**: Template for training Ultralytics models.
- **UltralyticsVal**: Template for validating Ultralytics models.
- **UltralyticsPredict**: Template for generating inference predictions with trained models.
- **UltralyticsExport**: Template for exporting models to deployment-ready format.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-object-detection/blob/main/packages/sinapsis_ultralytics/README.md).

</details>

<h2 id="webapp">🌐 Webapp</h2>

The webapps included in this project demonstrate the modularity of the templates, showcasing the capabilities of various object detection models for different tasks.

> [!IMPORTANT]
> To run the app, you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-object-detection.git
cd sinapsis-object-detection
```

> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!NOTE]
> Agent configuration can be changed through the `AGENT_CONFIG_PATH` env var. You can check the available configurations in each package configs folder.

> [!NOTE]
> When running the app with the **D-FINE** model, it defaults to a confidence threshold of `0.5`, uses **CUDA** for acceleration, and employs the **nano-sized** D-FINE model trained on the **COCO dataset**. These settings can be customized by modifying the `demo.yml` file inside the `configs` directory of the `sinapsis-dfine` package and restarting the webapp.


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-object-detection image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the app container**:

- For D-FINE:

```bash
docker compose -f docker/compose_apps.yaml up sinapsis-dfine-gradio -d
```

- For RF-DETR:

```bash
docker compose -f docker/compose_apps.yaml up sinapsis-rfdetr-gradio -d
```

- For Ultralytics Inference:

```bash
docker compose -f docker/compose_apps.yaml up sinapsis-ultralytics-inference -d
```

- For Ultralytics Training:

```bash
docker compose -f docker/compose_apps.yaml up sinapsis-ultralytics-train -d
```

3. **Check the logs**:

- For D-FINE:

```bash
docker logs -f sinapsis-dfine-gradio
```

- For RF-DETR:

```bash
docker logs -f sinapsis-rfdetr-gradio
```

- For Ultralytics Inference:

```bash
docker logs -f sinapsis-ultralytics-inference
```

- For Ultralytics Training:

```bash
docker logs -f sinapsis-ultralytics-train
```

4. **The logs will display the URL to access the webapp, e.g.**:

```bash
Running on local URL:  http://127.0.0.1:7860
```

**NOTE**: The url can be different, check the output of logs.

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
2. **Install the sinapsis-object-detection package**:
```bash
uv pip install sinapsis-object-detection[all] --extra-index-url https://pypi.sinapsis.tech
```
3. **Run the webapp**:

- For D-FINE:

```bash
uv run webapps/detection_demo.py
```

- For RF-DETR:

```bash
export AGENT_CONFIG_PATH=packages/sinapsis_rfdetr/src/sinapsis_rfdetr/configs/rfdetr_demo.yml
uv run webapps/detection_demo.py
```

- For Ultralytics Inference:

```bash
uv run webapps/inference_app.py
```

- For Ultralytics Training:

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
