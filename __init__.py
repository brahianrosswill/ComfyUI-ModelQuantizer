# __init__.py
# This file is necessary to make Python treat the directory as a package.
# It's also where ComfyUI looks for node mappings.

# Import node classes from nodes.py
# Renamed QuantizeScaled to QuantizeModel
from .nodes import ModelToStateDict, QuantizeFP8Format, QuantizeModel, SaveAsSafeTensor 

# A dictionary that ComfyUI uses to map node class names to node objects
NODE_CLASS_MAPPINGS = {
    # Node Utils
    "ModelToStateDict": ModelToStateDict,
    # Direct FP8 Conversion
    "QuantizeFP8Format": QuantizeFP8Format,
    # Scaled Quantization + Casting (FP16/BF16)
    "QuantizeModel": QuantizeModel,         # Renamed class QuantizeScaled -> QuantizeModel
    # Saving Node
    "SaveAsSafeTensor": SaveAsSafeTensor,
}

# A dictionary that ComfyUI uses to map node class names to display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelToStateDict": "Model To State Dict",
    "QuantizeFP8Format": "Quantize Model to FP8 Format",
    "QuantizeModel": "Quantize Model Scaled", # Display name for the renamed node
    "SaveAsSafeTensor": "Save Model as SafeTensor",
}

# Optional: Print a message to the console when the extension is loaded
print("----------------------------------------------------")
print("--- ComfyUI Quantization Node Pack Loaded ---")
print("--- Renamed QuantizeScaled to QuantizeModel ---")
print("--- Removed FP8 output from QuantizeModel   ---")
print("--- Using User-Provided Scaling Logic       ---") # Added note about scaling
print("--- Developed by [Lum3on]            ---") # Remember to change this!
print("--- Version 0.5.0                             ---") # Incremented version
print("----------------------------------------------------")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']