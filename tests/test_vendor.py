import hashlib
import json
import subprocess
import sys

from conftest import ROOT

VENDOR = ROOT / "miripple_comfy" / "vendor"
MANIFEST = json.loads((VENDOR / "VENDOR.json").read_text(encoding="utf-8"))


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_files_match_the_manifest() -> None:
    files = MANIFEST["files"]
    assert MANIFEST["commit"] == "865a1481acc22da80427bbebe11f1d7f00dc99be"
    assert sorted(p.name for p in (VENDOR / "mi_ripple").glob("*.py")) == sorted(files)
    for name, row in files.items():
        vendored = sha256(VENDOR / "mi_ripple" / name)
        assert vendored == row["vendored_sha256"], name
        assert (vendored == row["upstream_sha256"]) == (row["status"] == "unmodified"), name
    assert [n for n, r in files.items() if r["status"] == "patched"] == ["reference.py"]


def test_patch_license_and_test_generator_hashes() -> None:
    for path, row in MANIFEST["patches"].items():
        assert sha256(ROOT / path) == row["sha256"]
    assert sha256(VENDOR / "LICENSE.mi-ripple") == MANIFEST["license_file"]["sha256"]
    for path, row in MANIFEST["test_files"].items():
        assert sha256(ROOT / path) == row["sha256"]


def test_example_workflow_uses_only_known_nodes() -> None:
    known = {"LoadImage", "ImageScale", "SaveImage", "PreviewImage", "PreviewAny", "MaskToImage", "MiRippleLocalRepair", "MiRippleDiagnose"}
    for path in (ROOT / "workflows" / "api").glob("*.json"):
        prompt = json.loads(path.read_text(encoding="utf-8"))
        assert {node["class_type"] for node in prompt.values()} <= known, path.name
        repair = [n for n in prompt.values() if n["class_type"] == "MiRippleLocalRepair"]
        assert repair and all(n["inputs"]["accept_verify_failure"] is False for n in repair)


def test_no_top_level_mi_ripple_module() -> None:
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from miripple_comfy import runtime; "
            "print(sorted(m for m in sys.modules if m == 'mi_ripple' or m.startswith('mi_ripple.')))")
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
