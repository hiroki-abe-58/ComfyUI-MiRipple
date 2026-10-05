"""ComfyUI nodes for Mi-Ripple local diagnosis and repair (in-process, CPU, deterministic; no network)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import comfy.model_management as mm
import folder_paths
import numpy as np
import torch
from comfy_api.latest import ComfyExtension, io

from . import runtime

SIZE_TOOLTIP = (f"'{runtime.SIZE_512}' (verified scope) refuses other sizes. '{runtime.SIZE_ANY}' runs the official code "
                "on any size without resizing or cropping; results there were not compared with the official code.")
RECORDS_TOOLTIP = ("Copy the official records (JSON sidecars, diagnosis / verify boards, XML trace) to "
                   "output/mi-ripple/<run id>/ for inspection")


def _rgb8(image: torch.Tensor) -> np.ndarray:
    return runtime.image_to_rgb8(image.detach().cpu().numpy())


def _image(rgb8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb8.astype(np.float32) / 255.0).unsqueeze(0)


def _records_dir(keep: bool) -> Path | None:
    if not keep:
        return None
    return Path(folder_paths.get_output_directory()) / "mi-ripple" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


class MiRippleDiagnose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiRippleDiagnose",
            display_name="Mi-Ripple Diagnose",
            category="image/mi-ripple",
            description=("Official Mi-Ripple diagnosis (local, deterministic, CPU): periodic lattice in the frequency domain, "
                         "flat-region granules, whole-frame scale-like texture index. The image is not changed."),
            inputs=[
                io.Image.Input("image", tooltip="One RGB image (batch 1). Quantised to 8 bits like the official file input."),
                io.Combo.Input("size_policy", options=list(runtime.SIZE_POLICIES), default=runtime.SIZE_512, tooltip=SIZE_TOOLTIP),
                io.Boolean.Input("keep_records", default=False, tooltip=RECORDS_TOOLTIP),
            ],
            outputs=[
                io.Image.Output(display_name="heatmap", tooltip="Official scale heat map (red = flagged 128 px tiles)"),
                io.Image.Output(display_name="board", tooltip="Official diagnosis board (window crops and 3-8 px residual)"),
                io.Mask.Output(display_name="scale_mask", tooltip="1 inside flagged scale-index tiles"),
                io.Mask.Output(display_name="granule_mask", tooltip="1 inside flat windows flagged as granular"),
                io.Boolean.Output(display_name="lattice_detected"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, image, size_policy, keep_records) -> io.NodeOutput:
        out = runtime.diagnose_image(_rgb8(image), size_policy, _records_dir(keep_records))
        return io.NodeOutput(_image(out["heatmap"]), _image(out["board"]), torch.from_numpy(out["scale_mask"]).unsqueeze(0),
                             torch.from_numpy(out["granule_mask"]).unsqueeze(0), out["lattice"], json.dumps(out["report"], indent=1))


class MiRippleLocalRepair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiRippleLocalRepair",
            display_name="Mi-Ripple Local Repair",
            category="image/mi-ripple",
            description=("Official Mi-Ripple deterministic route: diagnose -> selective notch / masked granule reduction -> "
                         "verify. Only for periodic lattice and flat-region granule artifacts; not a general cleaner. "
                         "When the official outcome is not a verified delivery, the input is returned unchanged and "
                         "restoration_applied is false. Never regenerates and never uses the network."),
            inputs=[
                io.Image.Input("image", tooltip="One RGB image (batch 1). Quantised to 8 bits like the official file input."),
                io.Combo.Input("size_policy", options=list(runtime.SIZE_POLICIES), default=runtime.SIZE_512, tooltip=SIZE_TOOLTIP),
                io.Boolean.Input("accept_verify_failure", default=False,
                                 tooltip="Return the candidate even when the official verification failed "
                                         "(outcome delivered_with_verify_failure). Off: the input is returned unchanged."),
                io.Boolean.Input("keep_records", default=False, tooltip=RECORDS_TOOLTIP),
            ],
            outputs=[
                io.Image.Output(display_name="image", tooltip="The candidate when restoration_applied, otherwise the input unchanged"),
                io.Image.Output(display_name="candidate",
                                tooltip="The official final image when one was written (also after a failed verification), "
                                        "otherwise the input unchanged. For inspection."),
                io.Mask.Output(display_name="change_mask",
                               tooltip=f"max over RGB of |candidate - input| / {runtime.MASK_FULL_LEVELS} levels, clipped to 0..1"),
                io.Boolean.Output(display_name="restoration_applied"),
                io.String.Output(display_name="outcome",
                                 tooltip="Official outcome: delivered, delivered_with_verify_failure, needs_human_decision, "
                                         "failed or step_limit (cancelled raises an interrupt)"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, image, size_policy, accept_verify_failure, keep_records) -> io.NodeOutput:
        rgb8 = _rgb8(image)
        out = runtime.repair(rgb8, size_policy, accept_verify_failure, should_cancel=mm.processing_interrupted,
                             records_dir=_records_dir(keep_records))
        if out["outcome"] == runtime.CANCELLED:
            mm.throw_exception_if_processing_interrupted()
            raise mm.InterruptProcessingException()
        candidate = out["candidate"]
        mask = runtime.change_mask(rgb8, candidate) if candidate is not None else np.zeros(rgb8.shape[:2], np.float32)
        return io.NodeOutput(_image(candidate) if out["use_candidate"] else image,
                             _image(candidate) if candidate is not None else image,
                             torch.from_numpy(mask).unsqueeze(0), out["use_candidate"], out["outcome"],
                             json.dumps(out["report"], indent=1))


class MiRippleExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiRippleDiagnose, MiRippleLocalRepair]


async def comfy_entrypoint() -> MiRippleExtension:
    return MiRippleExtension()
