import json
from pathlib import Path

import numpy as np
import pytest
from conftest import smooth_image, synth

LATTICE_VERIFIED = ("make_lattice", 384, 512)
TEXTURED_512 = ("make_textured", 512, 512, True)


def _tensor(rgb8: np.ndarray):
    import torch

    return torch.from_numpy(rgb8.astype(np.float32) / 255.0).unsqueeze(0)


def test_node_schemas(comfy_dir):
    from miripple_comfy import nodes, runtime

    diagnose = nodes.MiRippleDiagnose.define_schema()
    repair = nodes.MiRippleLocalRepair.define_schema()
    assert (diagnose.node_id, repair.node_id) == ("MiRippleDiagnose", "MiRippleLocalRepair")
    assert (diagnose.display_name, repair.display_name) == ("Mi-Ripple Diagnose", "Mi-Ripple Local Repair")
    assert [i.id for i in diagnose.inputs] == ["image", "size_policy", "keep_records"]
    assert [i.id for i in repair.inputs] == ["image", "size_policy", "accept_verify_failure", "keep_records"]
    policy = repair.inputs[1]
    assert policy.options == list(runtime.SIZE_POLICIES) and policy.default == runtime.SIZE_512
    assert repair.inputs[2].default is False and repair.inputs[3].default is False
    assert [o.display_name for o in repair.outputs] == ["image", "candidate", "change_mask", "restoration_applied", "outcome", "report"]
    assert [o.display_name for o in diagnose.outputs] == ["heatmap", "board", "scale_mask", "granule_mask", "lattice_detected", "report"]


def test_repair_node_delivers_and_returns_input_otherwise(comfy_dir):
    from miripple_comfy import nodes, runtime

    lattice = _tensor(synth(*LATTICE_VERIFIED))
    out = nodes.MiRippleLocalRepair.execute(lattice, runtime.SIZE_ANY, False, False).result
    image, candidate, mask, applied, outcome, report = out
    assert outcome == "delivered" and applied is True
    assert image.shape == lattice.shape and image.dtype == lattice.dtype and not (image == lattice).all()
    assert (candidate == image).all() and mask.shape == (1, 384, 512) and float(mask.max()) > 0
    assert json.loads(report)["restoration_applied"] is True

    textured = _tensor(synth(*TEXTURED_512))
    image, candidate, mask, applied, outcome, report = nodes.MiRippleLocalRepair.execute(textured, runtime.SIZE_512, False, False).result
    assert outcome == "needs_human_decision" and applied is False
    assert image is textured and candidate is textured and float(mask.max()) == 0.0

    clean = _tensor(smooth_image())
    image, _, _, applied, outcome, _ = nodes.MiRippleLocalRepair.execute(clean, runtime.SIZE_512, False, False).result
    assert outcome == "delivered" and applied is False and image is clean


def test_refusals_and_interrupt(comfy_dir):
    import comfy.model_management as mm
    import torch

    from miripple_comfy import nodes, runtime

    with pytest.raises(runtime.InputRefused, match="batch of 2"):
        nodes.MiRippleLocalRepair.execute(torch.zeros(2, 512, 512, 3), runtime.SIZE_512, False, False)
    with pytest.raises(runtime.InputRefused, match="640x480"):
        nodes.MiRippleDiagnose.execute(torch.zeros(1, 480, 640, 3), runtime.SIZE_512, False)
    mm.interrupt_current_processing(True)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            nodes.MiRippleLocalRepair.execute(_tensor(smooth_image()), runtime.SIZE_512, False, False)
        assert mm.processing_interrupted() is False
    finally:
        mm.interrupt_current_processing(False)


def test_diagnose_node_and_records(comfy_dir, tmp_path, monkeypatch):
    import folder_paths

    from miripple_comfy import nodes, runtime

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    heatmap, board, scale_mask, granule_mask, lattice, report = nodes.MiRippleDiagnose.execute(
        _tensor(synth(*LATTICE_VERIFIED)), runtime.SIZE_ANY, True).result
    assert lattice is True and heatmap.shape == (1, 384, 512, 3) and board.ndim == 4
    assert scale_mask.shape == granule_mask.shape == (1, 384, 512)
    records = Path(json.loads(report)["records"])
    assert records.parent == tmp_path / "mi-ripple" and (records / "diagnose" / "input_diag.json").is_file()
