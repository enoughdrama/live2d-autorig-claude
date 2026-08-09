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




def test_keyform_grid_order():
    """Cubism reads the keyform grid with the FIRST bound axis varying FASTEST.

    Encodes each cell index into vertex positions and reads them back through
    the runtime. When this was row-major (itertools.product's natural order)
    the model still loaded, still deformed, and simply moved wrong -- every
    multi-parameter mesh read a transposed grid.
    """
    import ctypes  # noqa: F401  (load_core needs a working ctypes)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from render_model import Model, load_core
    except SystemExit as e:
        print(f"skip: grid order ({e})")
        return

    from autorig.build import ArtMeshSpec, RigBuilder
    from autorig.mesh import Mesh

    verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
    mesh = Mesh(verts=verts,
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
                indices=np.array([0, 1, 2, 0, 2, 3], np.uint16))
    rb = RigBuilder(100, 100, 1.0)
    pa = rb.add_param("ParamA", 0, 2, 0, [0.0, 1.0, 2.0])
    pb = rb.add_param("ParamB", 0, 1, 0, [0.0, 1.0])
    rb.add_part("Root")
    rb.add_mesh(ArtMeshSpec(
        name="Q", mesh=mesh, part_index=0, bound_params=[pa, pb],
        deform=lambda c: verts.astype(np.float64) + [c[0] * 10 + c[1] * 100, 0]))

    tmp = ROOT / "build" / "_gridorder.moc3"
    tmp.parent.mkdir(exist_ok=True)
    rb.build().to_file(tmp)

    m = Model(load_core(), tmp)
    for a in (0.0, 1.0, 2.0):
        for b in (0.0, 1.0):
            m.set_param("ParamA", a)
            m.set_param("ParamB", b)
            m.update()
            x = list(m.drawables())[0]["verts"][0][0]
            want = a * 10 + b * 100 - 1
            assert abs(x - want) < 0.01, \
                f"grid transposed: A={a} B={b} gave {x:.1f}, expected {want:.1f}"
    tmp.unlink()
    print("ok: keyform grid order (first axis fastest)")


def test_render_order_distinct():
    """self_group_idx must be -1, or the runtime reports render order 0 for
    every drawable and layers composite in arbitrary order."""
    if not OUT.exists():
        print("skip: render order (no out/Aka)")
        return
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from render_model import Model, load_core
    except SystemExit as e:
        print(f"skip: render order ({e})")
        return

    m = Model(load_core(), OUT / "Aka.moc3")
    orders = [d["order"] for d in m.drawables()]
    assert len(set(orders)) == len(orders), \
        f"only {len(set(orders))} distinct render orders for {len(orders)} drawables"
    print(f"ok: {len(orders)} drawables have distinct render order")


if __name__ == "__main__":
    test_manifest_shape()
    test_physics_shape()
    test_uv_atlas_agreement()
    test_keyform_grid_order()
    test_render_order_distinct()
    print("\nall checks passed")
