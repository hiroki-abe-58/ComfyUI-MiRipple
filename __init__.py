"""Mi-Ripple local diagnosis and repair for ComfyUI (unofficial integration of https://github.com/miyang-ai/Mi-Ripple)."""

import logging

try:
    from .miripple_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency (scikit-image)
    logging.error("[Mi-Ripple] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
