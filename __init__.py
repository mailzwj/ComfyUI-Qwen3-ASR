# ComfyUI-Qwen3-ASR Custom Nodes
# Based on the open-source Qwen3-ASR project by Alibaba Qwen team

# Import nodes
from .nodes import (
    Voice2TextNode,
)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3ASR": Voice2TextNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASR": "Qwen3-ASR",
}

# Version information
__version__ = "1.0.0"

print(f"✅ ComfyUI-Qwen3-ASR v{__version__} loaded")
