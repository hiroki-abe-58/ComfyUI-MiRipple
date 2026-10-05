"""Node side: run the node runtime (vendored code) on every case, keeping the official-format records.

usage: python scripts/run_node_runtime.py <cases_dir> <out_dir> [case ...]
Per case: <out>/<case>/ = repair records (input.png, repair/), plus diagnose/ and board/ from the Diagnose path, plus
node_outputs.json = sha256 of what the nodes return (image / candidate / change_mask / Diagnose masks) and the flags.
Images: 512x512 cases use the default size policy; other sizes use "any size (unverified)".
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miripple_comfy import runtime  # noqa: E402


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    cases_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or sorted(p.stem for p in cases_dir.glob("*.png"))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"runtime_file": runtime.__file__, "upstream": runtime.UPSTREAM, "versions": runtime.versions(), "runs": {}}
    for name in names:
        case = out / name
        side = out / f"{name}.diagnose-tmp"
        for d in (case, side):
            if d.exists():
                shutil.rmtree(d)
        rgb8 = np.asarray(Image.open(cases_dir / f"{name}.png").convert("RGB"))
        policy = runtime.SIZE_512 if rgb8.shape[:2] == (512, 512) else runtime.SIZE_ANY
        started = time.perf_counter()
        rep = runtime.repair(rgb8, policy, records_dir=case)
        run_s = time.perf_counter() - started
        started = time.perf_counter()
        dia = runtime.diagnose_image(rgb8, policy, records_dir=side)
        diag_s = time.perf_counter() - started
        shutil.move(side / "diagnose", case / "diagnose")
        shutil.move(side / "board", case / "board")
        shutil.rmtree(side)
        candidate = rep["candidate"]
        image = candidate if rep["use_candidate"] else rgb8
        mask = runtime.change_mask(rgb8, candidate) if candidate is not None else np.zeros(rgb8.shape[:2], np.float32)
        outputs = {
            "outcome": rep["outcome"], "restoration_applied": rep["use_candidate"], "size_policy": policy,
            "image_sha256": sha(image), "candidate_sha256": sha(candidate if candidate is not None else rgb8),
            "change_mask_sha256": sha(mask), "input_sha256": sha(rgb8),
            "diagnose": {"lattice_detected": dia["lattice"], "scale_mask_sha256": sha(dia["scale_mask"]),
                         "granule_mask_sha256": sha(dia["granule_mask"]), "heatmap_sha256": sha(dia["heatmap"]),
                         "board_sha256": sha(dia["board"]), "next_official_action": dia["report"]["next_official_action"]["action"]},
            "face_detection": rep["report"]["face_detection"],
        }
        (case / "node_outputs.json").write_text(json.dumps(outputs, indent=1) + "\n", encoding="utf-8", newline="\n")
        meta["runs"][name] = {"outcome": rep["outcome"], "restoration_applied": rep["use_candidate"],
                              "run_s": round(run_s, 3), "diagnose_s": round(diag_s, 3)}
        print(name, rep["outcome"], rep["use_candidate"], f"{run_s:.2f}s / {diag_s:.2f}s", flush=True)
    (out / "node_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
