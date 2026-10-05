"""Real ComfyUI HTTP-queue tests (server started with --cpu --cache-none on 127.0.0.1). Needs numpy + Pillow.

usage: python scripts/queue_tests.py <base_url> <official_runs_dir> <out_dir> <case,case,...> [--interrupt-case NAME]
The case PNGs must be in <ComfyUI>/input/miripple/. Every run keeps the official-format records (keep_records), which
are compared with the official run tree layer by layer (compare_runs.compare_case), and every node output saved by
SaveImage / PreviewAny is compared with what the documented contract gives from the official files.
"""

import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_runs  # noqa: E402

SIZE_512, SIZE_ANY = "512x512 only", "any size (unverified)"


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def req(self, path: str, data=None, timeout: float = 60):
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(self.base + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw.decode(errors="replace")

    def submit(self, prompt: dict) -> tuple[int, dict]:
        return self.req("/prompt", {"prompt": prompt, "client_id": "miripple-queue-tests"})

    def wait(self, prompt_id: str, timeout: float = 900) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            code, history = self.req(f"/history/{prompt_id}")
            if code == 200 and isinstance(history, dict) and prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed") is not None or status.get("status_str") in ("success", "error"):
                    return history[prompt_id]
            time.sleep(0.1)
        raise TimeoutError(prompt_id)

    def png(self, info: dict) -> np.ndarray:
        query = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info["subfolder"], "type": info["type"]})
        _, raw = self.req(f"/view?{query}")
        with Image.open(BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"))


def workflow(case: str, policy: str, prefix: str, repair=True, diagnose=True, repeat=None, accept=False) -> dict:
    p = {"1": {"class_type": "LoadImage", "inputs": {"image": f"miripple/{case}.png"}}}
    src = ["1", 0]
    if repeat:
        p["9"] = {"class_type": "RepeatImageBatch", "inputs": {"image": src, "amount": repeat}}
        src = ["9", 0]
    if repair:
        p["2"] = {"class_type": "MiRippleLocalRepair",
                  "inputs": {"image": src, "size_policy": policy, "accept_verify_failure": accept, "keep_records": True}}
        p["3"] = {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": f"{prefix}/image"}}
        p["4"] = {"class_type": "SaveImage", "inputs": {"images": ["2", 1], "filename_prefix": f"{prefix}/candidate"}}
        p["5"] = {"class_type": "MaskToImage", "inputs": {"mask": ["2", 2]}}
        p["6"] = {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": f"{prefix}/change_mask"}}
        p["7"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 3]}}
        p["8"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 4]}}
        p["10"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 5]}}
    if diagnose:
        p["11"] = {"class_type": "MiRippleDiagnose", "inputs": {"image": src, "size_policy": policy, "keep_records": True}}
        p["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": f"{prefix}/heatmap"}}
        p["13"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 1], "filename_prefix": f"{prefix}/board"}}
        p["14"] = {"class_type": "MaskToImage", "inputs": {"mask": ["11", 2]}}
        p["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"{prefix}/scale_mask"}}
        p["16"] = {"class_type": "MaskToImage", "inputs": {"mask": ["11", 3]}}
        p["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": f"{prefix}/granule_mask"}}
        p["18"] = {"class_type": "PreviewAny", "inputs": {"source": ["11", 4]}}
        p["19"] = {"class_type": "PreviewAny", "inputs": {"source": ["11", 5]}}
    return p


def run(client: Client, prompt: dict, timeout: float = 900) -> dict:
    started = time.time()
    code, res = client.submit(prompt)
    if code != 200:
        return {"queued": False, "http": code, "error": res}
    history = client.wait(res["prompt_id"], timeout)
    status = history["status"]
    stamps = {m[0]: m[1].get("timestamp") for m in status.get("messages", [])}
    end = stamps.get("execution_success") or stamps.get("execution_error") or stamps.get("execution_interrupted") or 0
    return {"queued": True, "prompt_id": res["prompt_id"], "status": status.get("status_str"), "wall_s": round(time.time() - started, 3),
            "exec_ms": end - (stamps.get("execution_start") or 0), "stamps": stamps,
            "errors": [{k: m[1].get(k) for k in ("node_type", "exception_type", "exception_message")}
                       for m in status.get("messages", []) if m[0] == "execution_error"],
            "history": history}


def text(history: dict, node: str):
    t = history.get("outputs", {}).get(node, {}).get("text")
    return t[0] if t else None


def saved(client: Client, history: dict, node: str) -> np.ndarray:
    return client.png(history["outputs"][node]["images"][0])


def as_png(mask: np.ndarray) -> np.ndarray:
    """What MaskToImage + SaveImage write for a float32 mask."""
    return np.repeat(np.clip(255. * mask.astype(np.float32), 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def check_outputs(client: Client, history: dict, official: Path) -> dict:
    """L6 through the queue: every saved node output against the contract computed from the official files."""
    summary = json.loads((official / "repair" / "input_restored.json").read_text(encoding="utf-8"))
    source = compare_runs.pix(official / "input.png")
    final_path = official / "repair" / "input_restored.png"
    final = compare_runs.pix(final_path) if final_path.exists() else None
    applied = summary["outcome"] == "delivered" and final is not None and not summary["final_is_source_copy"]
    diff = np.zeros(source.shape[:2], np.float32) if final is None else np.clip(
        np.abs(final.astype(np.int16) - source.astype(np.int16)).max(axis=2) / 8.0, 0.0, 1.0).astype(np.float32)
    diag = json.loads((official / "diagnose" / "input_diag.json").read_text(encoding="utf-8"))
    h, w = source.shape[:2]
    scale = w / int(diag["scale_index"]["measured_on"].split("x")[0])
    checks = {
        "outcome": text(history, "8") == summary["outcome"],
        "restoration_applied": text(history, "7") == str(applied),
        "image": np.array_equal(saved(client, history, "3"), final if applied else source),
        "candidate": np.array_equal(saved(client, history, "4"), final if final is not None else source),
        "change_mask": np.array_equal(saved(client, history, "6"), as_png(diff)),
        "heatmap": np.array_equal(saved(client, history, "12"), compare_runs.pix(official / "diagnose" / "input_scaleheat.png")),
        "board": np.array_equal(saved(client, history, "13"), compare_runs.pix(official / "repair" / "input_input_diag_board.png")),
        "scale_mask": np.array_equal(saved(client, history, "15"),
                                     as_png(compare_runs.boxes_mask((h, w), diag["scale_index"]["flagged_tile_boxes"], scale))),
        "granule_mask": np.array_equal(saved(client, history, "17"), as_png(compare_runs.boxes_mask(
            (h, w), [r["box"] for r in diag["granule"]["flat_windows"] if r["flagged"]]))),
        "lattice_detected": text(history, "18") == str(diag["flags"]["lattice"]),
    }
    return {k: bool(v) for k, v in checks.items()}


def case_run(client: Client, case: str, policy: str, official_dir: Path, tree: Path, tag: str) -> dict:
    r = run(client, workflow(case, policy, f"miripple-q/{tag}/{case}"))
    row = {"case": case, "policy": policy, "status": r.get("status"), "exec_ms": r.get("exec_ms"), "errors": r.get("errors")}
    if r.get("status") != "success":
        return row
    history = r["history"]
    repair_report, diag_report = json.loads(text(history, "10")), json.loads(text(history, "19"))
    row["records"] = [repair_report["records"], diag_report["records"]]
    row["seconds"] = {"repair": repair_report["seconds"], "diagnose": diag_report["seconds"]}
    row["face_detection"] = repair_report["face_detection"]
    dest = tree / tag / case
    shutil.copytree(repair_report["records"], dest)
    shutil.copytree(Path(diag_report["records"]) / "diagnose", dest / "diagnose")
    shutil.copytree(Path(diag_report["records"]) / "board", dest / "board")
    layers = compare_runs.compare_case(official_dir / case, dest)
    board_ok = np.array_equal(compare_runs.pix(dest / "board" / "input_input_diag_board.png"),
                              compare_runs.pix(official_dir / case / "repair" / "input_input_diag_board.png"))
    row["records_vs_official"] = {**layers["layers"], "L5b_board": bool(board_ok)}
    row["outputs_vs_contract"] = check_outputs(client, history, official_dir / case)
    row["ok"] = all(row["records_vs_official"].values()) and all(row["outputs_vs_contract"].values())
    return row


def main() -> None:
    base, official_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    cases = sys.argv[4].split(",")
    interrupt_case = sys.argv[sys.argv.index("--interrupt-case") + 1] if "--interrupt-case" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    tree = out / "tree"
    client = Client(base)
    sizes = {c: Image.open(official_dir / c / "input.png").size for c in cases}
    policy = {c: SIZE_512 if sizes[c] == (512, 512) else SIZE_ANY for c in cases}
    res = {"base": base, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    code, info = client.req("/object_info")
    res["Q0_registered"] = {n: (code == 200 and n in info) for n in ("MiRippleDiagnose", "MiRippleLocalRepair")}
    if code == 200 and "MiRippleLocalRepair" in info:
        res["Q0_registered"]["repair_inputs"] = info["MiRippleLocalRepair"]["input"]["required"]["size_policy"]

    res["Q1_cases"] = [case_run(client, c, policy[c], official_dir, tree, "q1") for c in cases]

    first = cases[0]
    twice = [case_run(client, first, policy[first], official_dir, tree, f"q2_{i}") for i in (1, 2)]
    res["Q2_same_prompt_twice"] = {"runs": twice, "executed_twice": all(r.get("ok") for r in twice)
                                   and twice[0].get("records") != twice[1].get("records")}

    a, b = cases[0], next((c for c in cases if c.startswith("P")), cases[-1])
    res["Q3_A_B_A"] = [case_run(client, c, policy[c], official_dir, tree, f"q3_{i}") for i, c in enumerate((a, b, a))]

    refusals = []
    edge = next((c for c in cases if sizes[c] != (512, 512)), None)
    if edge:
        r = run(client, workflow(edge, SIZE_512, "miripple-q/q4", diagnose=False))
        what = f"{edge} {sizes[edge][0]}x{sizes[edge][1]} under '{SIZE_512}'"
        refusals.append({"what": what, "status": r.get("status"), "errors": r.get("errors")})
    r = run(client, workflow(first, policy[first], "miripple-q/q4", diagnose=False, repeat=2))
    refusals.append({"what": "batch of 2 (RepeatImageBatch)", "status": r.get("status"), "errors": r.get("errors")})
    res["Q4_refusals"] = refusals
    res["Q4_recovery"] = case_run(client, first, policy[first], official_dir, tree, "q4_recovery")

    if interrupt_case:
        full = run(client, workflow(interrupt_case, SIZE_ANY, "miripple-q/q5_full", diagnose=False))
        full_report = json.loads(text(full["history"], "10")) if full.get("status") == "success" else {}
        res["Q5_uninterrupted"] = {"status": full.get("status"), "exec_ms": full.get("exec_ms"),
                                   "outcome": full_report.get("outcome"), "steps": [s["action"] for s in full_report.get("steps", [])]}
        trials = []
        for delay in (0.5, 2.0, 4.0, 6.0):
            code, q = client.submit(workflow(interrupt_case, SIZE_ANY, "miripple-q/q5", diagnose=False))
            if code != 200:
                trials.append({"delay_s": delay, "queued": False, "error": q})
                continue
            deadline = time.time() + 60
            while time.time() < deadline:  # wait until this prompt is executing
                _, queue = client.req("/queue")
                if any(item[1] == q["prompt_id"] for item in queue.get("queue_running", [])):
                    break
                time.sleep(0.02)
            started = time.time()
            time.sleep(delay)
            sent = time.time()
            client.req("/interrupt", {})
            history = client.wait(q["prompt_id"])
            stamps = {m[0]: m[1].get("timestamp") for m in history["status"].get("messages", [])}
            stopped = stamps.get("execution_interrupted") or stamps.get("execution_success") or stamps.get("execution_error")
            trials.append({"delay_s": delay, "status": history["status"].get("status_str"),
                           "interrupted": "execution_interrupted" in stamps, "outputs": sorted(history.get("outputs", {})),
                           "response_ms": None if stopped is None else round(stopped - sent * 1000),
                           "executing_ms_before_interrupt": round((sent - started) * 1000)})
        res["Q5_interrupt"] = trials
        res["Q5_recovery"] = case_run(client, first, policy[first], official_dir, tree, "q5_recovery")

    code, stats = client.req("/system_stats")
    res["system_stats"] = stats if code == 200 else None
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "queue_results.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    summary = {
        "Q0": res["Q0_registered"],
        "Q1_ok": f"{sum(bool(r.get('ok')) for r in res['Q1_cases'])}/{len(res['Q1_cases'])}",
        "Q1_not_ok": [r["case"] for r in res["Q1_cases"] if not r.get("ok")],
        "Q2_executed_twice": res["Q2_same_prompt_twice"]["executed_twice"],
        "Q3_ok": [bool(r.get("ok")) for r in res["Q3_A_B_A"]],
        "Q4": [(r["what"], r["status"], [e["exception_message"][:120] for e in r["errors"]]) for r in refusals],
        "Q4_recovery_ok": res["Q4_recovery"].get("ok"),
        "Q5_uninterrupted": res.get("Q5_uninterrupted"),
        "Q5": [(t["delay_s"], t.get("status"), t.get("interrupted"), t.get("response_ms")) for t in res.get("Q5_interrupt", [])],
        "Q5_recovery_ok": res.get("Q5_recovery", {}).get("ok"),
    }
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
