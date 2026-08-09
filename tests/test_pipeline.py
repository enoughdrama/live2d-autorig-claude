"""End-to-end checks that the binary validators cannot make.

csmHasMocConsistency proves the file is structurally sound. It says nothing
about whether the artwork lands where the UVs point, which is the failure that
renders as a garbled or invisible model.

Run: python3 tests/test_pipeline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from autorig.emit import _atlas_size, _uv_rescale  # noqa: E402

WORK = ROOT / "build" / "aka"
OUT = ROOT / "out" / "Aka"


def test_uv_atlas_agreement():
    """Every mesh's UVs must land on pixels the atlas actually painted.

    _uv_rescale and render_atlas independently compute the canvas->atlas
    mapping. If they ever disagree the model still validates and still loads --
    it just renders wrong. This samples real UVs against the real atlas.
    """
    if not (WORK / "layers.json").exists() or not OUT.exists():
        print("skip: run `python3 -m src.autorig samples/Aka_real.psd -o out/Aka` first")
        return

    from autorig.autorig import build_rig

    layers = json.loads((WORK / "layers.json").read_text())
    cw, ch = layers["canvas"]
    size = _atlas_size(cw, ch)

    tex = next(OUT.glob(f"Aka.{size}/texture_00.png"), None)
    assert tex is not None, f"no atlas at Aka.{size}/texture_00.png"
    alpha = np.array(Image.open(tex).convert("RGBA"))[:, :, 3]
    assert alpha.shape == (size, size), f"atlas is {alpha.shape}, expected {size}^2"

    rb = build_rig(WORK)
    sx, sy, ox, oy = _uv_rescale(cw, ch, size)

    checked = hits = 0
    for spec in rb.meshes:
        uv = spec.mesh.uvs.astype(np.float64).copy()
        uv[:, 0] = uv[:, 0] * sx + ox
        uv[:, 1] = uv[:, 1] * sy + oy
        assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6, \
            f"{spec.name}: atlas UV out of range [{uv.min():.4f},{uv.max():.4f}]"

        # UV origin is bottom-left; image rows are top-down
        px = np.clip((uv[:, 0] * size).astype(int), 0, size - 1)
        py = np.clip(((1.0 - uv[:, 1]) * size).astype(int), 0, size - 1)
        opaque = alpha[py, px] > 0
        checked += 1
        # A mesh traces its own alpha contour, so most sampled points must be
        # opaque. Boundary verts sit exactly on the edge, hence the margin.
        if opaque.mean() >= 0.5:
            hits += 1

    assert hits == checked, f"only {hits}/{checked} meshes land on painted atlas pixels"
    print(f"ok: {checked} meshes' UVs agree with the atlas")


def test_manifest_shape():
    if not OUT.exists():
        print("skip: manifest (no out/Aka)")
        return
    man = json.loads((OUT / "Aka.model3.json").read_text())
    refs = man["FileReferences"]
    for key in ("Moc", "Textures", "Physics", "DisplayInfo"):
        assert key in refs, f"model3.json missing {key}"
    for rel in [refs["Moc"], refs["Physics"], refs["DisplayInfo"], *refs["Textures"]]:
        assert (OUT / rel).exists(), f"model3.json references missing file {rel}"

    names = {g["Name"] for g in man["Groups"]}
    assert "EyeBlink" in names and "LipSync" in names, \
        "VTube Studio needs EyeBlink and LipSync groups for auto tracking"
    print("ok: model3.json references resolve; EyeBlink + LipSync present")


def test_physics_shape():
    if not OUT.exists():
        print("skip: physics (no out/Aka)")
        return
    ph = json.loads((OUT / "Aka.physics3.json").read_text())
    meta, settings = ph["Meta"], ph["PhysicsSettings"]
    assert meta["PhysicsSettingCount"] == len(settings)
    assert meta["TotalInputCount"] == sum(len(s["Input"]) for s in settings)
    assert meta["TotalOutputCount"] == sum(len(s["Output"]) for s in settings)
    assert meta["VertexCount"] == sum(len(s["Vertices"]) for s in settings)

    param_ids = {p["Id"] for p in json.loads((OUT / "Aka.cdi3.json").read_text())["Parameters"]}
    for s in settings:
        for o in s["Output"]:
            oid = o["Destination"]["Id"]
            assert oid in param_ids, f"physics drives unknown parameter {oid}"
            assert o["VertexIndex"] < len(s["Vertices"]), \
                f"{s['Id']}: VertexIndex out of range"
    print(f"ok: physics3.json consistent ({len(settings)} chains, "
          f"{meta['VertexCount']} vertices)")


if __name__ == "__main__":
    test_manifest_shape()
    test_physics_shape()
    test_uv_atlas_agreement()
    print("\nall checks passed")
