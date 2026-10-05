import json
import os
import socket
import sys
import types

import numpy as np
import pytest
from conftest import smooth_image, synth

from miripple_comfy import runtime
from miripple_comfy.vendor.mi_ripple import pipeline, regeneration

LATTICE_VERIFIED = ("make_lattice", 384, 512)  # upstream's own test size: official notch + verify passes
LATTICE_512 = ("make_lattice", 512, 512)  # official notch, verification fails at 512x512
TEXTURED_512 = ("make_textured", 512, 512, True)  # structured scale texture: official asks for a human decision


def test_image_to_rgb8_is_exact_for_8bit_input() -> None:
    levels = np.arange(256, dtype=np.uint8)
    image = np.stack([levels, levels[::-1], levels], -1).reshape(16, 16, 3)
    batch = (image.astype(np.float32) / 255.0)[None]
    assert np.array_equal(runtime.image_to_rgb8(batch), image)


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (np.zeros((2, 8, 8, 3), np.float32), "batch of 2"),
        (np.zeros((1, 8, 8, 4), np.float32), "4 channels"),
        (np.zeros((8, 8, 3), np.float32), "expected an IMAGE batch"),
        (np.full((1, 8, 8, 3), np.nan, np.float32), "NaN"),
    ],
)
def test_bad_input_is_refused(batch: np.ndarray, message: str) -> None:
    with pytest.raises(runtime.InputRefused, match=message):
        runtime.image_to_rgb8(batch)


def test_size_policy() -> None:
    other = np.zeros((480, 640, 3), np.uint8)
    with pytest.raises(runtime.InputRefused, match="640x480"):
        runtime.repair(other)
    with pytest.raises(runtime.InputRefused, match="640x480"):
        runtime.diagnose_image(other)
    assert runtime.check_size(other, runtime.SIZE_ANY) is False
    assert runtime.check_size(np.zeros((512, 512, 3), np.uint8), runtime.SIZE_512) is True
    with pytest.raises(runtime.InputRefused, match="unknown size_policy"):
        runtime.check_size(other, "resize")


def test_clean_image_is_delivered_unchanged() -> None:
    out = runtime.repair(smooth_image())
    assert out["outcome"] == "delivered"
    assert out["report"]["final_is_source_copy"]
    assert not out["use_candidate"]
    assert out["report"]["image_output"] == "input unchanged"
    assert [s["action"] for s in out["report"]["steps"]] == ["diagnose", "finish"]


def test_verified_lattice_is_delivered() -> None:
    image = synth(*LATTICE_VERIFIED)
    out = runtime.repair(image, runtime.SIZE_ANY)
    assert out["outcome"] == "delivered"
    assert out["use_candidate"]
    assert out["report"]["verify"]["passed"]
    assert out["report"]["in_verified_scope"] is False
    assert [s["action"] for s in out["report"]["steps"]] == ["diagnose", "notch", "verify", "finish"]
    assert out["candidate"].shape == image.shape and not np.array_equal(out["candidate"], image)
    mask = runtime.change_mask(image, out["candidate"])
    assert mask.dtype == np.float32 and 0 < mask.max() <= 1
    assert out["report"]["changed_pixels"] == int((mask > 0).sum())


def test_verify_failure_returns_input_unless_accepted() -> None:
    image = synth(*LATTICE_512)
    out = runtime.repair(image)
    assert out["outcome"] == "delivered_with_verify_failure"
    assert out["candidate"] is not None and not out["use_candidate"]
    assert out["report"]["verify"]["passed"] is False and out["report"]["verify"]["reasons"]
    accepted = runtime.repair(image, accept_verify_failure=True)
    assert accepted["use_candidate"] and np.array_equal(accepted["candidate"], out["candidate"])


def test_human_decision_returns_input() -> None:
    out = runtime.repair(synth(*TEXTURED_512))
    assert out["outcome"] == "needs_human_decision"
    assert out["candidate"] is None and not out["use_candidate"]
    assert "regeneration was not enabled" in out["report"]["human_note"]
    assert [s["action"] for s in out["report"]["steps"]] == ["diagnose", "request_human"]


@pytest.mark.parametrize("outcome", ["failed", "step_limit"])
def test_other_outcomes_return_input(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    def fake_run(source, out_dir, **kwargs):
        return {"outcome": outcome, "human_note": "x", "acceptance": "", "final": None, "final_is_source_copy": False,
                "diag": None, "verify": None, "steps": [{"n": 1, "action": "diagnose", "result": {}}]}

    monkeypatch.setattr(pipeline, "run", fake_run)
    out = runtime.repair(smooth_image())
    assert out["outcome"] == outcome and out["candidate"] is None and not out["use_candidate"]


def test_regeneration_step_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(source, out_dir, **kwargs):
        return {"outcome": "delivered", "human_note": "", "acceptance": "", "final": None, "final_is_source_copy": False,
                "diag": None, "verify": None, "steps": [{"n": 1, "action": "reference_clean", "result": {}}]}

    monkeypatch.setattr(pipeline, "run", fake_run)
    with pytest.raises(RuntimeError, match="regeneration step"):
        runtime.repair(smooth_image())


def test_cancel_between_steps() -> None:
    assert runtime.repair(smooth_image(), should_cancel=lambda: True)["outcome"] == "cancelled"
    calls = []

    def after_first_step() -> bool:
        calls.append(1)
        return len(calls) > 1

    out = runtime.repair(synth(*LATTICE_VERIFIED), runtime.SIZE_ANY, should_cancel=after_first_step)
    assert out["outcome"] == "cancelled" and out["candidate"] is None and not out["use_candidate"]
    assert [s["action"] for s in out["report"]["steps"]] == ["diagnose"]


def test_no_network_and_no_regeneration(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(args)
        raise OSError("network blocked by test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(regeneration, "client_from_env", blocked)
    monkeypatch.setattr(pipeline, "client_from_env", blocked)
    for key in ("MIYANG_API_KEY", "MIYANG_BASE_URL", "MIYANG_PROXY"):
        monkeypatch.setenv(key, "test-value-not-a-secret")
    for image, policy in ((synth(*LATTICE_VERIFIED), runtime.SIZE_ANY), (synth(*TEXTURED_512), runtime.SIZE_512)):
        out = runtime.repair(image, policy)
        assert "reference_clean" not in [s["action"] for s in out["report"]["steps"]]
        runtime.diagnose_image(image, policy)
    assert attempts == []
    assert os.environ["MIYANG_API_KEY"] == "test-value-not-a-secret"


def _fake_cv2(with_cascade_api: bool) -> types.ModuleType:
    module = types.ModuleType("cv2")
    module.COLOR_RGB2GRAY = 7
    module.cvtColor = lambda image, code: image[..., 0]
    module.equalizeHist = lambda gray: gray
    if with_cascade_api:
        module.CascadeClassifier = object
    return module


def test_broken_opencv_does_not_fail_the_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = runtime.repair(synth(*LATTICE_VERIFIED), runtime.SIZE_ANY)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(with_cascade_api=False))
    out = runtime.repair(synth(*LATTICE_VERIFIED), runtime.SIZE_ANY)
    assert out["report"]["face_detection"] == {"status": "cv2_error:AttributeError", "boxes": 0}
    assert out["outcome"] == "delivered" and np.array_equal(out["candidate"], reference["candidate"])
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert runtime.repair(smooth_image())["report"]["face_detection"]["status"] == "cv2_unavailable"


def test_records_and_report(tmp_path) -> None:
    out = runtime.repair(synth(*LATTICE_VERIFIED), runtime.SIZE_ANY, records_dir=tmp_path / "rec")
    rec = tmp_path / "rec"
    assert (rec / "input.png").is_file() and (rec / "repair" / "input_restored.json").is_file()
    assert (rec / "repair" / "input_verify_board.png").is_file()
    text = json.dumps(out["report"])
    assert "miripple_" not in text and out["report"]["records"] == str(rec)
    assert out["report"]["upstream"]["commit"] == "865a1481acc22da80427bbebe11f1d7f00dc99be"
    assert runtime.repair(smooth_image())["report"]["records"] is None


def test_diagnose_outputs() -> None:
    image = synth(*LATTICE_VERIFIED)
    out = runtime.diagnose_image(image, runtime.SIZE_ANY)
    assert out["lattice"] is True
    assert out["heatmap"].shape == image.shape
    assert out["scale_mask"].shape == out["granule_mask"].shape == image.shape[:2]
    assert out["report"]["next_official_action"]["action"] == "notch"
    textured = runtime.diagnose_image(synth(*TEXTURED_512))
    assert textured["report"]["next_official_action"]["action"] == "request_human"
    assert textured["scale_mask"].max() == 1.0
    assert textured["report"]["flags"]["scale_structured"]
