# __init__.py
# This file is necessary to make Python treat the directory as a package.
# It's also where ComfyUI looks for node mappings.

# Import node classes from nodes.py
# Renamed QuantizeScaled to QuantizeModel
from .nodes import ModelToStateDict, QuantizeFP8Format, QuantizeModel, SaveAsSafeTensor

# Import ControlNet FP8 quantization nodes
try:
    from .controlnet_fp8_node import ControlNetFP8QuantizeNode, ControlNetMetadataViewerNode
    CONTROLNET_NODES_AVAILABLE = True
    print("✅ ControlNet FP8 nodes imported successfully")
except Exception as e:
    print(f"❌ Failed to import ControlNet FP8 nodes: {e}")
    CONTROLNET_NODES_AVAILABLE = False

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

# Add ControlNet nodes if available
if CONTROLNET_NODES_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        # ControlNet FP8 Quantization Nodes
        "ControlNetFP8QuantizeNode": ControlNetFP8QuantizeNode,
        "ControlNetMetadataViewerNode": ControlNetMetadataViewerNode,
    })
    print("✅ ControlNet FP8 nodes registered in NODE_CLASS_MAPPINGS")

# A dictionary that ComfyUI uses to map node class names to display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelToStateDict": "Model To State Dict",
    "QuantizeFP8Format": "Quantize Model to FP8 Format",
    "QuantizeModel": "Quantize Model Scaled", # Display name for the renamed node
    "SaveAsSafeTensor": "Save Model as SafeTensor",
}

# Add ControlNet display names if available
if CONTROLNET_NODES_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # ControlNet FP8 Quantization Node Display Names
        "ControlNetFP8QuantizeNode": "ControlNet FP8 Quantizer",
        "ControlNetMetadataViewerNode": "ControlNet Metadata Viewer",
    })
    print("✅ ControlNet FP8 display names registered")

# Optional: Print a message to the console when the extension is loaded
print("----------------------------------------------------")
print("--- ComfyUI Quantization Node Pack Loaded ---")
print("--- Renamed QuantizeScaled to QuantizeModel ---")
print("--- Removed FP8 output from QuantizeModel   ---")
print("--- Using User-Provided Scaling Logic       ---") # Added note about scaling
print("--- NEW: ControlNet FP8 Quantization Nodes  ---") # New ControlNet nodes
print("--- NEW: Advanced Tensor Calibration        ---") # Advanced features
print("--- Developed by [Lum3on]            ---") # Remember to change this!
print("--- Version 0.6.0                             ---") # Incremented version for new features
print("----------------------------------------------------")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']