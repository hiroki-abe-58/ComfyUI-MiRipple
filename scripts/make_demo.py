"""Demo figure from a real ComfyUI queue run (scripts/queue_tests.py). Every listed case is shown, whatever its outcome.

usage: python scripts/make_demo.py <comfy_output_dir> <queue_results.json> <out.png> <case=label,...>
Columns: input | candidate (official final image, if any) | change_mask (node output) | centre crops x3 of input and
candidate. The node's image output is the candidate only when restoration_applied, otherwise the input unchanged.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THUMB, CROP, ZOOM, LABEL_W, PAD = 192, 64, 3, 330, 8


def load(p: Path) -> np.ndarray:
    with Image.open(p) as image:
        return np.asarray(image.convert("RGB"))


def thumb(a: np.ndarray) -> Image.Image:
    image = Image.fromarray(a)
    image.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (THUMB, THUMB), (245, 245, 245))
    canvas.paste(image, ((THUMB - image.width) // 2, (THUMB - image.height) // 2))
    return canvas


def crop(a: np.ndarray) -> Image.Image:
    h, w = a.shape[:2]
    y, x = (h - CROP) // 2, (w - CROP) // 2
    return Image.fromarray(a[y:y + CROP, x:x + CROP]).resize((CROP * ZOOM, CROP * ZOOM), Image.Resampling.NEAREST)


def main() -> None:
    output_dir, results, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), Path(sys.argv[3])
    labels = dict(item.split("=", 1) for item in sys.argv[4].split(","))
    rows = {r["case"]: r for r in results["Q1_cases"]}
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    headers = ["input", "candidate", "change_mask", f"input centre x{ZOOM}", f"candidate centre x{ZOOM}"]
    width = LABEL_W + len(headers) * (THUMB + PAD) + PAD
    height = 40 + len(labels) * (THUMB + PAD) + PAD
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, h in enumerate(headers):
        draw.text((LABEL_W + i * (THUMB + PAD) + 4, 12), h, fill="#1C1F23", font=font)
    for n, (case, label) in enumerate(labels.items()):
        top = 40 + n * (THUMB + PAD)
        base = output_dir / "miripple-q" / "q1" / case
        source = load(Path(rows[case]["records"][0]) / "input.png")
        image, candidate, mask = (load(base / f"{k}_00001_.png") for k in ("image", "candidate", "change_mask"))
        summary = json.loads((Path(rows[case]["records"][0]) / "repair" / "input_restored.json").read_text(encoding="utf-8"))
        applied = not np.array_equal(image, source)
        has_candidate = not np.array_equal(candidate, source)
        changed = float((mask[..., 0] > 0).mean() * 100)
        lines = [f"{case}  {label}", f"{source.shape[1]}x{source.shape[0]}", f"outcome: {summary['outcome']}",
                 f"restoration_applied: {applied}", "image output: " + ("candidate" if applied else "input unchanged"),
                 f"changed pixels: {changed:.1f}%" if has_candidate else "no candidate (input returned)"]
        verify = summary.get("verify")
        if verify and not verify["passed"]:
            first = verify["reasons"][0].split("]: ", 1)[-1]
            lines += [f"verify failed ({len(verify['reasons'])} reasons), e.g.", first[:46]]
        for k, line in enumerate(lines):
            draw.text((PAD, top + 6 + k * 19), line, fill="#A44A2A" if k == 2 else "#1C1F23", font=font if k < 2 else small)
        tiles = [thumb(source), thumb(candidate) if has_candidate else None, thumb(mask), crop(source), crop(candidate) if has_candidate else None]
        for i, tile in enumerate(tiles):
            x = LABEL_W + i * (THUMB + PAD)
            if tile is None:
                draw.rectangle([x, top, x + THUMB - 1, top + THUMB - 1], outline="#C8C8C8")
                draw.text((x + 50, top + THUMB // 2 - 8), "no candidate", fill="#7A7A7A", font=small)
            else:
                sheet.paste(tile, (x, top))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
