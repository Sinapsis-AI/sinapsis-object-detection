<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis D-FINE
<br>
</h1>

<h4 align="center">Templates for training and inference with the D-FINE model</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage example</a> •
<a href="#webapp"> 🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

The **Sinapsis D-FINE** module provides templates for training and inference with the D-FINE model, enabling advanced object detection tasks.

<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-dfine --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-dfine --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3>Templates Supported</h3>

The **Sinapsis D-FINE** module provides two main templates for **inference** and **training**:

- **`DFINETraining`**: A highly flexible template for fine-tuning D-FINE models on custom data. It is designed for rapid setup while still offering deep control.
  - **Effortless Setup**: Automatically infers class labels directly from the dataset, eliminating the need to manually create id2label maps.
  - **Flexible Data Sources**: Seamlessly loads datasets from both local directories and the Hugging Face Hub.
  - **Adaptable to Your Data**: Easily adapts to different dataset schemas by allowing users to specify custom keys for annotations (bbox, category, etc.) via the annotation_keys attribute.
  - **Powerful Customization**: Provides granular control over every aspect of training through structured Pydantic models for hyperparameters, data mapping, and more.
- **`DFINEInference`**: A streamlined and efficient template for running trained D-FINE models.
  - **High-Performance**: Processes images in batches for maximum throughput on the target hardware.
  - **Structured Output**: Generates clear, structured annotations for each image, including bounding boxes, confidence scores, and class labels, ready for downstream tasks.

<details>
<summary><strong><span style="font-size: 1.25em;">🌍 General Attributes</span></strong></summary>

Both templates share the following attributes:
- **`model_path` (str, optional)**: The model identifier from the Hugging Face Hub or a local path to the model and processor files. Defaults to `"ustc-community/dfine-nano-coco"`.
- **`model_cache_dir` (str, optional)**: Directory to cache downloaded model files. Defaults to the path specified by the `SINAPSIS_CACHE_DIR` environment variable.
- **`threshold` (float, required)**: The confidence score threshold (from 0.0 to 1.0) for filtering detections. For inference, it discards all detections below this value from the final output. For training, it is used on the validation dataset to filter predictions before calculating evaluation metrics.
- **`device` (Literal["auto", "cuda", "cpu"], optional)**: The hardware device to run the model on. Defaults to `"auto"`, which automatically selects `"cuda"` if a compatible GPU is available, otherwise falls back to `"cpu"`.

</details>
<details>
<summary><strong><span style="font-size: 1.25em;">Specific Attributes</span></strong></summary>

There are some attributes specific to the templates used:
- `DFINEInference` has one additional attribute:
    - **`batch_size` (int, optional)**: The number of images to process in a single batch. Defaults to `8`.
- `DFINETraining` has nine additional attributes:
    - **`training_mode` (Literal["fine-tune", "from-scratch"], optional)**: Specifies the training strategy.
    - **`dataset_path` (str, required)**: Path to the dataset to be loaded.
    - **`id2label` (dict[int, str] | None, optional)**: An optional mapping from class ID to label name. It's recommended to let the template infer this from the dataset. This attribute should only be used as a fallback if the dataset features are non-standard.
    - **`annotation_keys` (AnnotationKeys, optional)**: A configuration object that specifies the dictionary keys for accessing annotation data within the dataset.
      - **`bbox` (str, optional)**: The dictionary key for the bounding box annotations. Defaults to `"bbox"`.
      - **`category` (str, optional)**: The dictionary key for the category/class label annotations. Defaults to `"category"`.
      - **`area` (str, optional)**: The dictionary key for the bounding box area. If not provided, area will be calculated from the bbox. Defaults to `"area"`.
    - **`validation_split_size` (float, optional)**: The proportion of the dataset to reserve for validation. Defaults to `0.15`
    - **`mapping_args` (DatasetMappingArgs, optional)**: Parameters for the dataset preprocessing step.
      - **`batch_size` (int, optional)**: The batch size for applying transformations. A larger size can speed up preprocessing but requires more RAM. Defaults to `16`.
      - **`num_proc` (int, optional)**: The number of CPU processes to use for mapping. Defaults to `0` (no multiprocessing).
    - **`image_size` (TrainingImageSize, optional)**: The target image size for image resizing.
      - **`width` (int, optional)**: The target width for image resizing. Defaults to `640`.
      - **`height` (int, optional)**: The target height for image resizing. Defaults to `640`.
    - **`training_args` (TrainingArgs, optional)**: A nested configuration object for all Hugging Face `Trainer` hyperparameters. Refer to the [official documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) for the full list of possible arguments.
    - **`save_dir` (str, required)**: Path to the directory where the fine-tuned model will be saved.

</details>

<details>
<summary><strong><span style="font-size: 1.25em;">📁 Supported Dataset Structure</span></strong></summary>

To ensure compatibility and smooth training, the `DFINETraining` template relies on a specific dataset structure. This format is inspired by the widely used COCO dataset, making it easy to adapt many existing object detection datasets.

**IMPORTANT**: The `DFINETraining` template expects datasets to follow a specific **nested (COCO-style) format**. This ensures consistency and reliability during the data transformation process.

Each example in your dataset must contain at least two features:
1.  **`image`**: A PIL Image object.
2.  **`objects`**: A dictionary that acts as a container for all annotations related to the image.

The `objects` dictionary must contain parallel lists for the annotations. The keys for these lists are configurable via the `annotation_keys` attribute.

**Example of a single dataset entry:**
```python
{
  'image': <PIL.Image object>,
  'objects': {
    'bbox': [[x, y, width, height], [x, y, width, height], ...],
    'category': [label_id_1, label_id_2, ...],
    'area': [area_1, area_2, ...]  # This is optional and will be calculated if not present
  }
}
```
<details>
<summary><strong><span style="font-size: 1.1em;">Preparing a Local Dataset</span></strong></summary>

To load a local dataset of images, the files must be structured with a `metadata.jsonl` file, which is the standard method for the Hugging Face `datasets` library.

1. The folder structure should be organized as follows:

```bash
my_dataset/
|--- train/
|   |--- image1.jpg
|   |--- image2.png
|   |--- metadata.jsonl
|--- validation/
    |--- image3.jpg
    |--- metadata.jsonl
```

2. A `metadata.jsonl` file must be created. Each line in this file is a JSON object describing one image and its annotations.

Example line in `train/metadata.jsonl`:

```json
{"file_name": "image1.jpg", "objects": {"bbox": [[22, 34, 100, 150]], "category": [3]}}
```

3. The dataset can be loaded by providing the path to the root folder (`my_dataset/`). The template will automatically find and parse the `metadata.jsonl` files.

For more detailed information on creating image datasets for object detection, refer to the official [Hugging Face documentation](https://huggingface.co/docs/datasets/image_dataset#object-detection).

</details>

</details>

<details>
<summary><strong><span style="font-size: 1.25em;">Advanced Configuration</span></strong></summary>

<h4>License Validation for Hub Datasets</h4>

For commercial safety, the `DFINETraining` template automatically validates that datasets from the Hugging Face Hub have a permissive license. This check can be managed using an environment variable.

- **`ALLOW_UNVETTED_DATASETS`**:
  - **Default Behavior (`True`):** By default, the license check is **skipped**. This is to provide a smooth experience for local development and testing.
  - **Production Behavior (`False`):** For production environments, this variable **must** be explicitly set to `False` to enforce the license validation and ensure only commercially safe datasets are used.

**Example (for production):**
```bash
export ALLOW_UNVETTED_DATASETS=False
```

</details>

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***DFINEInference*** use ```sinapsis info --example-template-config DFINEInference``` to produce an example config like:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: DFINEInference
  class_name: DFINEInference
  template_input: InputTemplate
  attributes:
    model_path: ustc-community/dfine-nano-coco
    model_cache_dir: '/path/to/sinapsis/cache'
    threshold: '`replace_me:<class ''float''>`'
    device: auto
    batch_size: 8
```

<h2 id='example'>📚 Usage example</h2>

The following example demonstrates how to use the **DFINEInference** template for object detection. This setup processes a folder of images, runs inference using the **D-FINE** model, and saves the results, including detected bounding boxes.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: dfine_inference
  description: "run inferences with D-FINE"

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: FolderImageDatasetCV2
    class_name: FolderImageDatasetCV2
    template_input: InputTemplate
    attributes:
      data_dir: datasets/coco

  - template_name: DFINEInference
    class_name: DFINEInference
    template_input: FolderImageDatasetCV2
    attributes:
      model_path: ustc-community/dfine-small-coco
      batch_size: 16
      threshold: 0.5
      device: cuda

  - template_name: BBoxDrawer
    class_name: BBoxDrawer
    template_input: DFINEInference
    attributes:
      overwrite: true
      randomized_color: false

  - template_name: ImageSaver
    class_name: ImageSaver
    template_input: BBoxDrawer
    attributes:
      root_dir: datasets
      save_dir: output
      extension: png
```
</details>

This configuration defines an **agent** and a sequence of **templates** to run object detection with **D-FINE**.

> [!IMPORTANT]
> The FolderImageDatasetCV2, BBoxDrawer and ImageSaver correspond to [sinapsis-data-readers](https://github.com/Sinapsis-AI/sinapsis-data-tools/tree/main/packages/sinapsis_data_readers), [sinapsis-data-visualization](https://github.com/Sinapsis-AI/sinapsis-data-tools/tree/main/packages/sinapsis_data_visualization) and [sinapsis-data-writers](https://github.com/Sinapsis-AI/sinapsis-data-tools/tree/main/packages/sinapsis_data_writers). If you want to use the example, please make sure you install the packages.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

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
> When running the app with the **D-FINE** model, it defaults to a confidence threshold of `0.5`, uses **CUDA** for acceleration, and employs the **nano-sized** D-FINE model trained on the **COCO dataset**. These settings can be customized by modifying the `demo.yml` file inside `packages/sinapsis_dfine/src/sinapsis_dfine/configs` directory and restarting the webapp.


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT**: This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-object-detection image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the app container**:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-dfine-gradio -d
```

3. **Check the status**:
```bash
docker logs -f sinapsis-dfine-gradio
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
```bash
uv run webapps/detection_demo.py
```

4. **The terminal will display the URL to access the webapp, e.g.**:

```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The url can be different, check the output of the terminal.

</details>



<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
