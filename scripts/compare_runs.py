"""Compare two run trees case by case, layer by layer. usage: python scripts/compare_runs.py <A> <B> <out.json>

A is the reference (official run tree), B the run under test (node records or a second official run).
L0 input pixels; L1 summary (outcome, actions, final_is_source_copy, verify); L2 diagnosis JSON; L3/L5 every image in
repair/ and diagnose/ (decoded pixels); L4 verify JSON. Volatile keys (times, absolute paths, file-byte hashes) are
removed before JSON comparison and listed apart. When B has node_outputs.json, L5b (diagnosis board) and L6 (what the
nodes return, recomputed here from A's files by the documented contract) are checked too. Exit 1 unless all cases pass.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VOLATILE = {"time", "source", "source_before", "before", "after", "output", "board", "heatmap", "diag", "final",
            "reference", "prompt_file", "source_sha256", "final_sha256", "after_sha256", "current", "scale_heatmap"}
MASK_FULL_LEVELS = 8


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in VOLATILE}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def volatile_keys(v, prefix=""):
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            out += [prefix + k] if k in VOLATILE else volatile_keys(x, prefix + k + ".")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            out += volatile_keys(x, prefix + f"{i}.")
    return out


def pix(p: Path) -> np.ndarray:
    with Image.open(p) as image:
        return np.asarray(image.convert("RGB"))


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def boxes_mask(shape, boxes, scale=1.0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        mask[int(round(y1 * scale)):int(round(y2 * scale)), int(round(x1 * scale)):int(round(x2 * scale))] = 1.0
    return mask


def expected_outputs(a: Path) -> dict:
    """What the nodes must return, from the official files only (accept_verify_failure off)."""
    summary = json.loads((a / "repair" / "input_restored.json").read_text(encoding="utf-8"))
    source = pix(a / "input.png")
    final_path = a / "repair" / "input_restored.png"
    final = pix(final_path) if final_path.exists() else None
    applied = summary["outcome"] == "delivered" and final is not None and not summary["final_is_source_copy"]
    diff = np.zeros(source.shape[:2], np.float32) if final is None else np.clip(
        np.abs(final.astype(np.int16) - source.astype(np.int16)).max(axis=2) / float(MASK_FULL_LEVELS), 0.0, 1.0).astype(np.float32)
    diag = json.loads((a / "diagnose" / "input_diag.json").read_text(encoding="utf-8"))
    h, w = source.shape[:2]
    scale = w / int(diag["scale_index"]["measured_on"].split("x")[0])
    steps = [s["action"] for s in summary["steps"]]
    return {
        "outcome": summary["outcome"], "restoration_applied": applied,
        "image_sha256": sha(final if applied else source), "candidate_sha256": sha(final if final is not None else source),
        "change_mask_sha256": sha(diff), "input_sha256": sha(source),
        "diagnose": {"lattice_detected": diag["flags"]["lattice"],
                     "scale_mask_sha256": sha(boxes_mask((h, w), diag["scale_index"]["flagged_tile_boxes"], scale)),
                     "granule_mask_sha256": sha(boxes_mask((h, w), [r["box"] for r in diag["granule"]["flat_windows"] if r["flagged"]])),
                     "heatmap_sha256": sha(pix(a / "diagnose" / "input_scaleheat.png")),
                     "board_sha256": sha(pix(a / "repair" / "input_input_diag_board.png")),
                     "next_official_action": steps[1] if len(steps) > 1 else None},
    }


def compare_case(a: Path, b: Path) -> dict:
    res = {"layers": {}, "images": {}, "json": {}, "missing": []}
    res["layers"]["L0_input"] = bool(np.array_equal(pix(a / "input.png"), pix(b / "input.png")))
    for sub in ("repair", "diagnose"):
        fa = {p.name for p in (a / sub).glob("*")} if (a / sub).exists() else set()
        fb = {p.name for p in (b / sub).glob("*")} if (b / sub).exists() else set()
        res["missing"] += [f"{sub}/{n} only in A" for n in sorted(fa - fb)] + [f"{sub}/{n} only in B" for n in sorted(fb - fa)]
        for n in sorted(fa & fb):
            pa, pb = a / sub / n, b / sub / n
            if n.endswith(".png"):
                xa, xb = pix(pa), pix(pb)
                same_shape = xa.shape == xb.shape
                res["images"][f"{sub}/{n}"] = {"equal": bool(same_shape and np.array_equal(xa, xb)),
                                               "max_abs": int(np.abs(xa.astype(int) - xb.astype(int)).max()) if same_shape else None}
            elif n.endswith(".json"):
                ja, jb = json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8"))
                res["json"][f"{sub}/{n}"] = {"equal_without_volatile": strip(ja) == strip(jb),
                                             "volatile_keys": sorted(set(volatile_keys(ja)) | set(volatile_keys(jb)))[:12]}
    sa = json.loads((a / "repair" / "input_restored.json").read_text(encoding="utf-8"))
    sb = json.loads((b / "repair" / "input_restored.json").read_text(encoding="utf-8"))
    res["outcome"] = [sa["outcome"], sb["outcome"]]
    res["layers"]["L1_summary"] = all(sa[k] == sb[k] for k in ("outcome", "final_is_source_copy", "regen_rounds")) and \
        [s["action"] for s in sa["steps"]] == [s["action"] for s in sb["steps"]] and strip(sa.get("verify")) == strip(sb.get("verify"))
    res["layers"]["L2_diagnosis_json"] = all(v["equal_without_volatile"] for k, v in res["json"].items() if "diag" in k)
    res["layers"]["L3_L5_images"] = all(v["equal"] for v in res["images"].values()) and bool(res["images"])
    res["layers"]["L4_verify_json"] = all(v["equal_without_volatile"] for k, v in res["json"].items() if "verify" in k)
    res["layers"]["all_json"] = all(v["equal_without_volatile"] for v in res["json"].values())
    res["layers"]["same_file_set"] = not res["missing"]
    if (b / "node_outputs.json").exists():
        board = b / "board" / "input_input_diag_board.png"
        res["layers"]["L5b_board"] = board.exists() and bool(np.array_equal(pix(board), pix(a / "repair" / "input_input_diag_board.png")))
        got = json.loads((b / "node_outputs.json").read_text(encoding="utf-8"))
        want = expected_outputs(a)
        res["L6_mismatch"] = [k for k in want if k != "diagnose" and want[k] != got.get(k)] + \
            [f"diagnose.{k}" for k in want["diagnose"] if want["diagnose"][k] != got.get("diagnose", {}).get(k)]
        res["layers"]["L6_node_outputs"] = not res["L6_mismatch"]
        res["restoration_applied"] = got["restoration_applied"]
    res["ok"] = all(res["layers"].values())
    return res


def main() -> None:
    A, B, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cases = sorted(p.name for p in A.iterdir() if p.is_dir())
    results = {c: compare_case(A / c, B / c) for c in cases if (B / c).is_dir()}
    summary = {"reference": str(A.name), "under_test": str(B.name), "cases": len(results), "ok": sum(r["ok"] for r in results.values()),
               "not_ok": [c for c, r in results.items() if not r["ok"]], "missing_in_B": [c for c in cases if not (B / c).is_dir()],
               "images_compared": sum(len(r["images"]) for r in results.values()),
               "json_compared": sum(len(r["json"]) for r in results.values()),
               "outcomes": {c: r["outcome"][0] for c, r in results.items()},
               "restoration_applied": {c: r.get("restoration_applied") for c, r in results.items()}}
    out.write_text(json.dumps({"summary": summary, "cases": results}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    for c in summary["not_ok"]:
        r = results[c]
        print(c, {k: v for k, v in r["layers"].items() if not v}, r["missing"][:4], r.get("L6_mismatch"),
              [k for k, v in r["images"].items() if not v["equal"]][:4], [k for k, v in r["json"].items() if not v["equal_without_volatile"]][:4])
    sys.exit(0 if results and not summary["not_ok"] and not summary["missing_in_B"] else 1)


if __name__ == "__main__":
    main()
