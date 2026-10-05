"""Test setup. Node-level tests need a ComfyUI checkout (v0.38.0) given by MIRIPPLE_COMFYUI_DIR and torch;
MIRIPPLE_REQUIRE_COMFYUI=1 (set in CI) makes a missing checkout a failure instead of a skip."""

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import upstream_synthetic as synthetic  # noqa: E402

COMFY = os.environ.get("MIRIPPLE_COMFYUI_DIR")
if COMFY and Path(COMFY).exists():
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
else:
    COMFY = None


@pytest.fixture(scope="session")
def comfy_dir():
    if COMFY:
        return COMFY
    if os.environ.get("MIRIPPLE_REQUIRE_COMFYUI") == "1":
        pytest.fail("a ComfyUI checkout is required (MIRIPPLE_COMFYUI_DIR, MIRIPPLE_REQUIRE_COMFYUI=1)")
    pytest.skip("ComfyUI checkout not available (MIRIPPLE_COMFYUI_DIR)")


@lru_cache(maxsize=None)
def synth(name: str, height: int = 512, width: int = 512, *args) -> np.ndarray:
    """Upstream's own synthetic generator at a given size (read-only result)."""
    old = synthetic.HEIGHT, synthetic.WIDTH
    synthetic.HEIGHT, synthetic.WIDTH = height, width
    try:
        image = getattr(synthetic, name)(*args)
    finally:
        synthetic.HEIGHT, synthetic.WIDTH = old
    image.setflags(write=False)
    return image


def smooth_image(size: int = 512) -> np.ndarray:
    """A clean smooth gradient with soft shapes (no artifact class)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = 90 + 120 * xx / size
    g = 70 + 100 * yy / size + 30 * np.exp(-((xx - 180) ** 2 + (yy - 300) ** 2) / (2 * 70.0 ** 2))
    b = 140 + 60 * np.exp(-((xx - 360) ** 2 + (yy - 160) ** 2) / (2 * 90.0 ** 2)) - 40 * yy / size
    return np.clip(np.rint(np.stack([r, g, b], -1)), 0, 255).astype(np.uint8)
