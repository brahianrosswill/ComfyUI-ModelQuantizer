# ComfyUI Quantization Node Pack

A custom node pack for ComfyUI that provides tools for quantizing model weights to lower precision formats like FP16, BF16, or true FP8 types.

## Overview

This node pack provides tools for quantizing models directly within ComfyUI. It includes:

1.  **Model To State Dict**: Extracts the state dictionary from a model object and attempts to normalize keys.
2.  **Quantize Model to FP8 Format**: Converts model weights directly to `float8_e4m3fn` or `float8_e5m2` format (requires CUDA).
3.  **Quantize Model Scaled**: Applies simulated FP8 scaling (per-tensor or per-channel) and then casts the model to `float16`, `bfloat16`, or keeps the original format.
4.  **Save As SafeTensor**: Saves the processed state dictionary to a `.safetensors` file at a specified path.

## Installation

1.  Clone or download this repository into your ComfyUI's `custom_nodes` directory.
    * Example using git:
        ```bash
        cd ComfyUI/custom_nodes
        git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git) ComfyUI_QuantNodes 
        # Replace with your actual repo URL and desired folder name
        ```
    * Alternatively, download the ZIP and extract it into `ComfyUI/custom_nodes/ComfyUI_QuantNodes`.
2.  Install dependencies (optional, check if already installed):
    ```bash
    pip install -r ComfyUI/custom_nodes/ComfyUI_QuantNodes/requirements.txt
    ```
3.  Restart ComfyUI.

## Usage

### Model To State Dict
* **Category:** `Model Quantization/Utils`
* **Function:** Extracts state dict from a MODEL object, stripping common prefixes.
* **Inputs**:
    * `model`: The input `MODEL` object.
* **Outputs**:
    * `model_state_dict`: The extracted state dictionary.

### Quantize Model to FP8 Format
* **Category:** `Model Quantization/FP8 Direct`
* **Function:** Converts model weights directly to a specific FP8 format. Requires CUDA.
* **Inputs**:
    * `model_state_dict`: The state dictionary to quantize.
    * `fp8_format`: The target FP8 format (`float8_e5m2` or `float8_e4m3fn`).
* **Outputs**:
    * `quantized_model_state_dict`: The state dictionary with FP8 tensors.

### Quantize Model Scaled
* **Category:** `Model Quantization`
* **Function:** Applies simulated FP8 value scaling and then casts to FP16, BF16, or keeps the original dtype. Useful for size reduction with good compatibility.
* **Inputs**:
    * `model_state_dict`: The state dictionary to quantize.
    * `scaling_strategy`: How to simulate scaling (`per_tensor` or `per_channel`).
    * `processing_device`: Where to perform calculations (`Auto`, `CPU`, `GPU`).
    * `output_dtype`: Final data type (`Original`, `float16`, `bfloat16`). Defaults to `float16`.
* **Outputs**:
    * `quantized_model_state_dict`: The processed state dictionary.

### Save As SafeTensor
* **Category:** `Model Quantization/Save`
* **Function:** Saves the processed state dictionary to a `.safetensors` file.
* **Inputs**:
    * `quantized_model_state_dict`: The state dictionary to save.
    * `absolute_save_path`: The full path (including filename) where the model will be saved.
* **Outputs**: None (Output node).

## Example Workflow

1.  Load a model using a standard loader (e.g., `Load Checkpoint`).
2.  Connect the `MODEL` output to the `Model To State Dict` node.
3.  Connect the `model_state_dict` output from `Model To State Dict` to `Quantize Model Scaled`.
4.  In `Quantize Model Scaled`, select your desired `scaling_strategy` and set `output_dtype` to `float16` (for size reduction).
5.  Connect the `quantized_model_state_dict` output from `Quantize Model Scaled` to the `Save Model as SafeTensor` node.
6.  Specify the `absolute_save_path` in the `Save Model as SafeTensor` node.
7.  Queue the prompt.
8.  Restart ComfyUI or refresh loaders to find the saved model.

## Requirements

* PyTorch (usually included with ComfyUI)
* `safetensors`
* `tqdm`
* CUDA-enabled GPU is required for the `Quantize Model scaled` node.

## License

MIT (Or your chosen license)
