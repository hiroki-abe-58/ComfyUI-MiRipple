"""Write the CI parity cases with upstream's own synthetic generators (tests/upstream_synthetic.py, unmodified).

usage: python scripts/make_ci_cases.py <out_dir>
U1-U4: make_clean / make_lattice / make_granule / make_textured(uniform) at 512x512 (verified size).
X1-X2: make_lattice / make_granule at upstream's own 384x512 (run with size_policy "any size").
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import upstream_synthetic as synthetic  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cases = {}
    synthetic.HEIGHT, synthetic.WIDTH = 512, 512
    cases["U1"] = synthetic.make_clean()
    cases["U2"] = synthetic.make_lattice()
    cases["U3"] = synthetic.make_granule()
    cases["U4"] = synthetic.make_textured(True)
    synthetic.HEIGHT, synthetic.WIDTH = 384, 512
    cases["X1"] = synthetic.make_lattice()
    cases["X2"] = synthetic.make_granule()
    rows = {}
    for name, image in cases.items():
        path = out / f"{name}.png"
        Image.fromarray(image, "RGB").save(path)
        assert np.array_equal(np.asarray(Image.open(path).convert("RGB")), image)
        rows[name] = {"size": [int(image.shape[1]), int(image.shape[0])],
                      "pixels_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()}
    (out / "cases.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
