"""Thin runtime around the vendored official Mi-Ripple code. No ComfyUI imports, so it is testable on its own.

Every call goes through the official file-based API exactly as the upstream CLI does: the input is written as an
8-bit PNG named input.png in a fresh temporary folder, then `pipeline.run(...)` / `diagnosis.diagnose(...)` read it
with the official loader. Regeneration (MIYANG API) is never enabled: allow_regen=False, client=None, and an
authorize_regen that always says no. Nothing is kept between calls.
"""

from __future__ import annotations

import json
import platform
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .vendor.mi_ripple import pipeline
from .vendor.mi_ripple.common import load_rgb, write_json
from .vendor.mi_ripple.diagnosis import diag_board, diagnose
from .vendor.mi_ripple.reference import detect_faces

UPSTREAM = {
    "repository": "https://github.com/miyang-ai/Mi-Ripple",
    "commit": "865a1481acc22da80427bbebe11f1d7f00dc99be",
    "license": "MIT, Copyright (c) 2026 MIYANG Technology (Shanghai) Co., Ltd.",
    "local_patches": ["0001-detect-faces-best-effort (proposed upstream: miyang-ai/Mi-Ripple#2)"],
}
SIZE_512 = "512x512 only"
SIZE_ANY = "any size (unverified)"
SIZE_POLICIES = (SIZE_512, SIZE_ANY)
VERIFIED_SCOPE = "512x512 RGB, batch 1"
DELIVERED = "delivered"
VERIFY_FAILED = "delivered_with_verify_failure"
CANCELLED = "cancelled"
MASK_FULL_LEVELS = 8  # change_mask: a change of this many 8-bit levels (any channel) or more is 1.0
REGENERATION_ACTIONS = {"reference_clean", "regenerate"}
_PATH_KEYS = {"current", "output", "board", "scale_heatmap", "reference", "heatmap", "source", "diag", "final"}


class InputRefused(ValueError):
    """The input is outside what the node accepts. Nothing was processed, resized, cropped or dropped."""


def _never() -> bool:
    return False


def versions() -> dict:
    import scipy
    import skimage
    from PIL import __version__ as pillow

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-image": skimage.__version__, "Pillow": pillow}


def image_to_rgb8(batch: np.ndarray) -> np.ndarray:
    """ComfyUI IMAGE as numpy [B, H, W, C] in 0..1 -> one H x W x 3 uint8 image (round to nearest, clip).

    The official code works on 8-bit files, so the input is quantised exactly as an 8-bit PNG would be.
    """
    if batch.ndim != 4:
        raise InputRefused(f"expected an IMAGE batch [B, H, W, C], got shape {tuple(batch.shape)}")
    count, height, width, channels = batch.shape
    if count != 1:
        raise InputRefused(f"batch of {count} images: Mi-Ripple nodes process exactly one image per run "
                           "(nothing is dropped silently). Split the batch first, for example with an image-batch splitter.")
    if channels != 3:
        raise InputRefused(f"{channels} channels: only RGB is accepted (alpha or other channels are not handled). "
                           "Convert to RGB first.")
    if height < 1 or width < 1:
        raise InputRefused(f"empty image {width}x{height}")
    image = np.asarray(batch[0], dtype=np.float64)
    if not np.isfinite(image).all():
        raise InputRefused("the image contains NaN or infinite values")
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def check_size(rgb8: np.ndarray, size_policy: str) -> bool:
    """Returns True when the input is inside the verified scope. Refuses other sizes under the 512 policy."""
    if size_policy not in SIZE_POLICIES:
        raise InputRefused(f"unknown size_policy {size_policy!r}; choose one of {list(SIZE_POLICIES)}")
    height, width = rgb8.shape[:2]
    if (width, height) == (512, 512):
        return True
    if size_policy == SIZE_512:
        raise InputRefused(f"input is {width}x{height}; size_policy '{SIZE_512}' accepts 512x512 RGB only. "
                           f"Resize or crop it yourself, or choose '{SIZE_ANY}'. This node never resizes or crops.")
    return False


def _strip_paths(value):
    if isinstance(value, dict):
        return {k: _strip_paths(v) for k, v in value.items() if k not in _PATH_KEYS}
    if isinstance(value, list):
        return [_strip_paths(v) for v in value]
    return value


def _diagnosis_summary(diag: dict) -> dict:
    return {
        "lattice": {k: diag["lattice"].get(k) for k in ("detected", "max_peak_excess", "isolated_components_kept")},
        "granule": {"level": diag["granule"]["level"], "flagged_windows": diag["granule"]["flagged_windows"],
                    "flat_windows": len(diag["granule"]["flat_windows"]),
                    "oriented_windows": len(diag["granule"]["oriented_windows"])},
        "scale_index": {k: diag["scale_index"].get(k) for k in ("scale_tile_pct", "level", "flagged_tiles", "tiles",
                                                                "measured_on", "canvas_normalized_for_grading")},
        "flags": diag["flags"],
    }


def _keep(tmp: Path, records_dir: Path | None) -> str | None:
    if records_dir is None:
        return None
    records_dir = Path(records_dir)
    shutil.copytree(tmp, records_dir)
    return str(records_dir)


def change_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """max over RGB of |after - before| in 8-bit levels, divided by MASK_FULL_LEVELS, clipped to [0, 1]. float32 H x W."""
    delta = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    return np.clip(delta / float(MASK_FULL_LEVELS), 0.0, 1.0).astype(np.float32)


def repair(rgb8: np.ndarray, size_policy: str = SIZE_512, accept_verify_failure: bool = False,
           should_cancel: Callable[[], bool] | None = None, records_dir: Path | None = None) -> dict:
    """Run the official deterministic pipeline (diagnose -> notch / iso_clean -> verify -> finish) on one image.

    Returns a dict with: outcome (official, verbatim), candidate (uint8 or None: the official final image),
    use_candidate (the contract below), report (JSON-able). The caller returns the input unchanged unless
    use_candidate is True. Contract: delivered with a changed image -> candidate; delivered_with_verify_failure ->
    candidate only when accept_verify_failure; every other outcome -> input unchanged. cancelled is reported as is.
    """
    in_scope = check_size(rgb8, size_policy)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="miripple_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "input.png"
        Image.fromarray(rgb8).save(source)
        boxes, face_status = detect_faces(load_rgb(source))
        summary = pipeline.run(source, tmp / "repair", allow_regen=False, client=None,
                               authorize_regen=_never, should_cancel=should_cancel)
        actions = [step["action"] for step in summary["steps"]]
        if REGENERATION_ACTIONS & set(actions):
            raise RuntimeError(f"a regeneration step was reached ({actions}); this local-only build refuses it")
        candidate = None
        if summary["final"] is not None:
            with Image.open(summary["final"]) as image:
                candidate = np.asarray(image.convert("RGB")).copy()
        diag = json.loads(Path(summary["diag"]).read_text(encoding="utf-8")) if summary["diag"] else None
        records = _keep(tmp, records_dir)
    outcome = summary["outcome"]
    changed = candidate is not None and not summary["final_is_source_copy"]
    use_candidate = changed and (outcome == DELIVERED or (outcome == VERIFY_FAILED and accept_verify_failure))
    report = {
        "node": "Mi-Ripple Local Repair",
        "outcome": outcome,
        "restoration_applied": use_candidate,
        "image_output": "candidate" if use_candidate else "input unchanged",
        "human_note": summary["human_note"],
        "acceptance": summary["acceptance"],
        "accept_verify_failure": bool(accept_verify_failure),
        "candidate_available": candidate is not None,
        "final_is_source_copy": summary["final_is_source_copy"],
        "changed_pixels": int((change_mask(rgb8, candidate) > 0).sum()) if candidate is not None else 0,
        "verify": summary["verify"] and _strip_paths(summary["verify"]),
        "steps": [_strip_paths({k: v for k, v in step.items() if k != "n"}) for step in summary["steps"]],
        "diagnosis": _diagnosis_summary(diag) if diag else None,
        "face_detection": {"status": face_status, "boxes": len(boxes)},
        "size": [int(rgb8.shape[1]), int(rgb8.shape[0])],
        "size_policy": size_policy,
        "in_verified_scope": in_scope,
        "verified_scope": VERIFIED_SCOPE,
        "regeneration": "never (allow_regen=False, no client; MIYANG_* variables are ignored)",
        "records": records,
        "seconds": round(time.perf_counter() - started, 3),
        "upstream": UPSTREAM,
        "versions": versions(),
    }
    return {"outcome": outcome, "candidate": candidate, "use_candidate": use_candidate, "report": report}


def _boxes_mask(shape: tuple[int, int], boxes: list, scale: float = 1.0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        mask[int(round(y1 * scale)):int(round(y2 * scale)), int(round(x1 * scale)):int(round(x2 * scale))] = 1.0
    return mask


def diagnose_image(rgb8: np.ndarray, size_policy: str = SIZE_512, records_dir: Path | None = None) -> dict:
    """Official diagnosis only (no image change).

    Returns heatmap (uint8, the official scale heat map: canvas >1280 px wide is downscaled by upstream), board (uint8,
    the official diagnosis board), scale_mask / granule_mask (float32 H x W, input coordinates: union of flagged
    128 px scale tiles / flagged flat granule windows), lattice (bool), report.
    """
    in_scope = check_size(rgb8, size_policy)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="miripple_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "input.png"
        Image.fromarray(rgb8).save(source)
        rgb = load_rgb(source)
        (tmp / "diagnose").mkdir()
        (tmp / "board").mkdir()
        diag = diagnose(rgb, scale_heat_out=tmp / "diagnose" / "input_scaleheat.png")
        write_json(tmp / "diagnose" / "input_diag.json", diag)
        board_path = diag_board(rgb, diag, tmp / "board" / "input_input_diag_board.png", f"Diagnosis: {source.name}")
        next_action = _next_official_action(source, tmp, diag)
        with Image.open(diag["scale_index"]["heatmap"]) as image:
            heatmap = np.asarray(image.convert("RGB")).copy()
        with Image.open(board_path) as image:
            board = np.asarray(image.convert("RGB")).copy()
        records = _keep(tmp, records_dir)
    height, width = rgb8.shape[:2]
    measured_width = int(diag["scale_index"]["measured_on"].split("x")[0])
    scale_mask = _boxes_mask((height, width), diag["scale_index"]["flagged_tile_boxes"], width / measured_width)
    granule_mask = _boxes_mask((height, width), [row["box"] for row in diag["granule"]["flat_windows"] if row["flagged"]])
    report = {
        "node": "Mi-Ripple Diagnose",
        **_diagnosis_summary(diag),
        "next_official_action": next_action,
        "size": [int(width), int(height)],
        "size_policy": size_policy,
        "in_verified_scope": in_scope,
        "verified_scope": VERIFIED_SCOPE,
        "records": records,
        "seconds": round(time.perf_counter() - started, 3),
        "upstream": UPSTREAM,
        "versions": versions(),
    }
    return {"heatmap": heatmap, "board": board, "scale_mask": scale_mask, "granule_mask": granule_mask,
            "lattice": bool(diag["flags"]["lattice"]), "diagnosis": diag, "report": report}


def _next_official_action(source: Path, tmp: Path, diag: dict) -> dict:
    """What the official rules (pipeline.decide_rules) would do next after this diagnosis, regeneration disabled."""
    state = pipeline.State(source=source, out_dir=tmp, stem=source.stem, current=source, allow_regen=False,
                           max_regen=2, regen_model=pipeline.DEFAULT_MODEL, quality="high", authorize_regen=_never)
    state.diag = diag
    action, reason = pipeline.decide_rules(state)
    return {"action": action, "reason": reason, "note": state.human_note}
