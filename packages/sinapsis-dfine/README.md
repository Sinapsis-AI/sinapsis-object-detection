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

- **DFINETraining**: This module implements the training pipeline for the D-FINE model. It includes logic for initializing configuration, downloading weights and setting up the training solver.
- **DFINEInference**: Template designed to perform inference on a set of images using the different D-FINE architectures available.

<details>
<summary><strong><span style="font-size: 1.25em;">🌍 General Attributes</span></strong></summary>

Both templates share the following attributes:
- **`config_file` (str, required)**: Path to the model configuration file. Refer to the [original repo](https://github.com/Peterande/D-FINE) for detailed instructions on using, creating and customizing these configuration files.
- **`pretrained_model` (dict | None, optional)**: Specifies the **size** and **variant** of the pretrained model.
- **`device` (Literal["cpu", "cuda"], required)**: Defines whether to run inference on **CPU** or **CUDA**.
- **`weights_path` (str | None, optional)**: Path to a custom weights file, if provided. Defaults to `None`.
- **`output_dir` (str, optional)**: Directory where downloaded weights will be stored. Defaults to **SINAPSIS_CACHE_DIR**.

</details>
<details>
<summary><strong><span style="font-size: 1.25em;">Specific Attributes</span></strong></summary>

There are some attributes specific to the templates used:
- `DFINEInference` has four additional attributes:
    - **`threshold` (float, required)**: Confidence score threshold for filtering detections.
    - **`batch_inference` (bool, optional)**: Whether to perform batch inference. Defaults to `False`.
    - **`warmup_iterations` (int, optional)**: Number of warm-up iterations to optimize model performance. Defaults to `10`.
    - **`id2label` (dict[int, str] | None, optional)**: Mapping of class indices to label strings. Required if using custom weights. Defaults to `None`.
- `DFINETraining` has five additional attributes:
    - **`training_mode` (Literal["scratch", "tune"], required)**: `"scratch"` trains the model from scratch, while `"tune"` is meant to be used to fine-tune the model with provided or downloaded weights.
    - **`seed` (int | None, optional)**: Random seed for reproducibility. Defaults to `None`.
    - **`use_amp` (bool, optional)**: Enables Automatic Mixed Precision (AMP) for improved performance. Defaults to `False`.
    - **`print_rank` (int, optional)**: Rank of the process for logging in distributed training. Defaults to `0`.
    - **`print_method` (Literal["builtin", "rich"], optional)**: Defines the logging method while training. Defaults to `"builtin"`.

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
    config_file: '/path/to/config.yml'
    pretrained_model: null
    device: 'cuda'
    weights_path: null
    output_dir: '/path/to/sinapsis/cache'
    threshold: 0.5
    warmup_iterations: 10
    id2label: null
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
      threshold: 0.5
      config_file: artifacts/configs/dfine/dfine_hgnetv2_n_coco.yml
      device: cuda
      output_dir: ./artifacts/dfine_hgnetv2_n_coco
      pretrained_model:
        size: n
        variant: coco

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



<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
