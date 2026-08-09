"""Stage 1 — parse a layered PSD into layers.json + trimmed PNGs.

Deterministic. No LLM, no guessing about semantics: this stage records what is
actually in the file (name, bbox, opacity, blend mode, draw order, nesting) and
leaves interpretation to stage 2.

Reality check that shaped this module: both official Inochi2D sample PSDs
(Aka, Midori — real production files) are *completely flat*, zero groups, 76 and
81 layers. Group nesting is a nice signal when present but cannot be relied on.
Name and geometry are what actually carry structure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from psd_tools import PSDImage

# psd-tools exposes blend mode as a 4-byte tag; Inochi2D/Cubism want a name.
# Keys are the raw PSD tags (note the space padding — 'mul ', 'scrn').
BLEND_TAGS = {
    b"norm": "Normal",
    b"mul ": "Multiply",
    b"scrn": "Screen",
    b"over": "Overlay",
    b"dark": "Darken",
    b"lite": "Lighten",
    b"div ": "ColorDodge",
    b"lddg": "LinearDodge",
    b"idiv": "ColorBurn",
    b"hLit": "HardLight",
    b"sLit": "SoftLight",
    b"diff": "Difference",
    b"smud": "Exclusion",
    b"fsub": "Subtract",
    b"lbrn": "LinearBurn",
    b"pass": "Normal",  # pass-through group; parts are flattened anyway
}


def blend_name(raw) -> str:
    """Map a PSD blend tag to an Inochi2D blend_mode name. Unknown -> Normal."""
    if isinstance(raw, str):
        raw = raw.encode()
    if hasattr(raw, "value"):  # psd_tools BlendMode enum
        raw = raw.value
    return BLEND_TAGS.get(raw, "Normal")


def slugify(name: str, used: set[str]) -> str:
    """Filesystem-safe unique stem for a layer name.

    'Eye L Eyelid2 Shadow' -> 'eye_l_eyelid2_shadow'. Collisions get _2, _3 —
    PSDs genuinely do contain duplicate layer names.
    """
    s = re.sub(r"[^\w\-]+", "_", name.strip().lower()).strip("_")
    s = re.sub(r"_+", "_", s) or "layer"
    stem, n = s, 2
    while s in used:
        s = f"{stem}_{n}"
        n += 1
    used.add(s)
    return s


def parse(psd_path, out_dir, *, export_png: bool = True) -> dict:
    """Parse `psd_path`, writing trimmed PNGs into `out_dir/layers/`.

    Returns the manifest dict (also written to `out_dir/layers.json`).

    Draw order: `index` counts bottom-to-top in composite order, matching how
    the PSD stacks. psd-tools iterates a group bottom-up, so plain traversal
    order is already correct.
    """
    psd_path, out_dir = Path(psd_path), Path(out_dir)
    layer_dir = out_dir / "layers"
    if export_png:
        layer_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    psd = PSDImage.open(psd_path)
    layers: list[dict] = []
    used: set[str] = set()
    counter = [0]

    def visit(node, path: list[str]):
        for layer in node:
            here = path + [layer.name]
            if layer.is_group():
                visit(layer, here)
                continue

            bbox = tuple(layer.bbox)  # (l, t, r, b) in PSD pixel space
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue  # empty layer — no pixels, nothing to rig

            stem = slugify(layer.name, used)
            rec = {
                "index": counter[0],
                "name": layer.name,
                "file": f"layers/{stem}.png",
                "path": "/".join(here),          # PSD-style provenance
                "group": "/".join(path) or None,  # None when flat
                "bbox": list(bbox),
                "size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "opacity": round(layer.opacity / 255, 4),
                "blend_mode": blend_name(layer.blend_mode),
                "visible": bool(layer.visible),
                "clipping": bool(getattr(layer, "clipping", False)),
            }
            counter[0] += 1

            if export_png:
                img = layer.composite()  # trimmed to bbox, alpha preserved
                if img is not None:
                    img.convert("RGBA").save(layer_dir / f"{stem}.png")
                else:
                    rec["file"] = None
            layers.append(rec)

    visit(psd, [])

    manifest = {
        "source": psd_path.name,
        "canvas": [psd.width, psd.height],
        "layer_count": len(layers),
        "flat": all(l["group"] is None for l in layers),
        "layers": layers,
    }
    (out_dir / "layers.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Stage 1: PSD -> layers.json + PNGs")
    ap.add_argument("psd")
    ap.add_argument("-o", "--out", default="build", help="output directory")
    ap.add_argument("--no-png", action="store_true", help="manifest only, skip image export")
    a = ap.parse_args(argv)

    m = parse(a.psd, a.out, export_png=not a.no_png)
    print(
        f"{m['source']}: {m['layer_count']} layers, canvas {m['canvas'][0]}x{m['canvas'][1]}, "
        f"{'flat' if m['flat'] else 'grouped'} -> {a.out}/layers.json"
    )


if __name__ == "__main__":
    main()
