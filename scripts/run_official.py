"""Reference side: run the OFFICIAL mi_ripple package (installed in its own environment) on every case.

usage: python scripts/run_official.py <cases_dir> <out_dir> [case ...]
Per case: <out>/<case>/input.png (copy), <out>/<case>/repair/ = pipeline.run(input.png, repair/) with the official
defaults (allow_regen False), <out>/<case>/diagnose/ = diagnose(load_rgb(input.png), scale_heat_out=...) + its JSON.
The input stem is "input" on both sides, so file names match the node runtime's records.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import mi_ripple
from mi_ripple import pipeline
from mi_ripple.common import load_rgb, write_json
from mi_ripple.diagnosis import diagnose


def main() -> None:
    cases_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or sorted(p.stem for p in cases_dir.glob("*.png"))
    meta = {"mi_ripple_file": mi_ripple.__file__, "version": mi_ripple.__version__, "runs": {}}
    for name in names:
        case = out / name
        if case.exists():
            shutil.rmtree(case)
        (case / "repair").mkdir(parents=True)
        (case / "diagnose").mkdir(parents=True)
        source = case / "input.png"
        shutil.copyfile(cases_dir / f"{name}.png", source)
        started = time.perf_counter()
        summary = pipeline.run(source, case / "repair")
        run_s = time.perf_counter() - started
        started = time.perf_counter()
        diag = diagnose(load_rgb(source), scale_heat_out=case / "diagnose" / "input_scaleheat.png")
        diag_s = time.perf_counter() - started
        write_json(case / "diagnose" / "input_diag.json", diag)
        meta["runs"][name] = {"outcome": summary["outcome"], "actions": [s["action"] for s in summary["steps"]],
                              "run_s": round(run_s, 3), "diagnose_s": round(diag_s, 3)}
        print(name, summary["outcome"], meta["runs"][name]["actions"], f"{run_s:.2f}s / {diag_s:.2f}s", flush=True)
    (out / "official_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
