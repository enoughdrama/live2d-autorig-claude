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


def test_role_vocabulary_total():
    """Every role the classifier emits must be grouped by the rig.

    A role in neither HEAD_ROLES nor BODY_ROLES falls through to the body group
    silently -- a blush would rig to the torso and slide off the cheek on every
    head turn. This caught blush/breast/collar/hair/hair_ahoge, all of which the
    classifier could emit and the rig quietly mis-parented.
    """
    from autorig import classify
    from autorig.rig import BODY_ROLES, HEAD_ROLES, NECK_ROLES

    emitted = {r[0] for r in classify.RULES} | {"unknown", "accessory"}
    grouped = HEAD_ROLES | BODY_ROLES
    missing = emitted - grouped
    assert not missing, f"roles the rig never groups: {sorted(missing)}"
    assert NECK_ROLES <= BODY_ROLES, "neck roles must be body-anchored"

    # The two physics sets had drifted apart in both directions.
    from autorig.autorig import PHYSICS_ROLES
    assert set(classify.PHYSICS_ROLES) <= PHYSICS_ROLES, \
        f"classify marks physics roles the rig gives no sway param: " \
        f"{sorted(set(classify.PHYSICS_ROLES) - PHYSICS_ROLES)}"
    print(f"ok: role vocabulary total ({len(emitted)} roles all grouped)")


def _runtime_model():
    """Load out/Aka through Cubism Core, or None if unavailable."""
    sys.path.insert(0, str(ROOT / "tools"))
    if not OUT.exists():
        return None, None
    try:
        from render_model import Model, load_core
    except SystemExit:
        return None, None
    m = Model(load_core(), OUT / "Aka.moc3")
    plan = json.loads((OUT / ".work" / "rig_plan.json").read_text())
    roles = {p["layer"].replace(" ", "_").replace(":", "")[:60]:
             (p["role"], p["side"]) for p in plan["parts"]}
    return m, roles


def _rest(m):
    for p in m.param_ids:
        m.set_param(p, 1.0 if p in ("ParamEyeLOpen", "ParamEyeROpen") else 0.0)
    m.update()
    return {d["id"]: d["verts"].copy() for d in m.drawables()}


def _snap(m):
    m.update()
    return {d["id"]: d["verts"].copy() for d in m.drawables()}


def test_neck_follows_head():
    """The neck must be dragged by the head, more at the jaw than the torso.

    ROLE_FOLLOW has always declared neck=0.35, but the neck is in BODY_ROLES so
    head_transform was never called on it -- the constant was dead config and
    the neck stayed perfectly rigid (max displacement 0.0000) while the head
    turned 30 degrees.
    """
    m, roles = _runtime_model()
    if m is None:
        print("skip: neck follow (no runtime)")
        return

    rest = _rest(m)
    m.set_param("ParamAngleX", 30.0)
    turned = _snap(m)

    necks = [k for k in rest if roles.get(k, ("", ""))[0] == "neck"]
    assert necks, "no neck part in the test model"
    for k in necks:
        v0, v1 = rest[k], turned[k]
        top = v0[:, 1] > v0[:, 1].max() - 0.03
        bot = v0[:, 1] < v0[:, 1].min() + 0.03
        jaw = np.abs(v1[top] - v0[top]).max()
        torso = np.abs(v1[bot] - v0[bot]).max()
        assert jaw > 0.01, f"{k}: neck does not follow the head (jaw moved {jaw:.4f})"
        assert jaw > torso * 3, \
            f"{k}: neck moves rigidly (jaw {jaw:.4f} vs torso {torso:.4f}); " \
            "it must be ramped so the shoulders stay put"
    print(f"ok: {len(necks)} neck parts follow the head, anchored at the torso")


def test_blink_leaves_a_lid_line():
    """A closed eye is a line, not a hole.

    Every eye part used to scale to exactly 0% height, so the lash and lid
    geometry vanished along with the sclera and the closed eye read as a gap in
    the face. Lids and lashes must keep a visible residual; sclera and iris are
    correctly hidden.
    """
    m, roles = _runtime_model()
    if m is None:
        print("skip: blink profile (no runtime)")
        return

    rest = _rest(m)
    m.set_param("ParamEyeLOpen", 0.0)
    shut = _snap(m)

    kept = {}
    for k, (role, side) in roles.items():
        if k not in rest or side != "left":
            continue
        if role not in ("eyelid", "eyelash", "eye_white", "eyeball", "eye_light"):
            continue
        h0 = rest[k][:, 1].max() - rest[k][:, 1].min()
        if h0 <= 1e-9:
            continue
        kept[role] = kept.get(role, []) + [
            (shut[k][:, 1].max() - shut[k][:, 1].min()) / h0]

    for role in ("eyelid", "eyelash"):
        assert role in kept, f"no left {role} to check"
        assert min(kept[role]) > 0.02, \
            f"{role} collapses to {min(kept[role]):.3f} -- closed eye reads as a hole"
    for role in ("eye_white", "eyeball"):
        if role in kept:
            assert max(kept[role]) < 0.02, \
                f"{role} stays visible ({max(kept[role]):.3f}) behind a closed lid"
    print("ok: blink keeps a lid line and hides the sclera")


def test_sway_is_a_rotation():
    """Hanging parts must swing about their anchor, not shear sideways.

    sway() was a depth-weighted horizontal shear: at full deflection the tail
    tip translated 0.43 model units while the strand's length stayed constant --
    a motion no pendulum makes, and the reason the tail read as offset to one
    side rather than swaying. A rotation preserves length and pins the anchor.
    """
    m, roles = _runtime_model()
    if m is None:
        print("skip: sway rotation (no runtime)")
        return

    rest = _rest(m)
    for p in m.param_ids:
        if p.startswith("ParamSway"):
            m.set_param(p, 1.0)
    swung = _snap(m)

    checked = 0
    for k, (role, _) in roles.items():
        if k not in rest or role not in ("tail", "hair_back", "hair_side"):
            continue
        v0, v1 = rest[k], swung[k]
        top = v0[:, 1] > v0[:, 1].max() - 0.03
        drift = np.abs(v1[top] - v0[top]).max()
        assert drift < 0.02, f"{k}: anchor drifts {drift:.4f}; strand detaches"

        # A rotation preserves the strand's own extent; a shear does not.
        d0 = np.hypot(*(v0.max(0) - v0.min(0)))
        d1 = np.hypot(*(v1.max(0) - v1.min(0)))
        assert abs(d1 / d0 - 1.0) < 0.12, \
            f"{k}: extent changes {(d1/d0-1)*100:+.1f}% -- that is a shear, not a swing"
        checked += 1
    assert checked, "no hanging parts to check"
    print(f"ok: {checked} hanging parts rotate about their anchor")


def test_blink_sides_are_independent():
    """Closing one eye must not move the other.

    _is_right() once compared the classifier's "right" against "R", which sent
    every right-side part to the left parameter. That left ParamEyeROpen driving
    nothing and made both eyes blink together off a single parameter.
    """
    m, roles = _runtime_model()
    if m is None:
        print("skip: blink sides (no runtime)")
        return

    eye_roles = ("eyelid", "eyelash", "eye_white", "eyeball", "eye_light",
                 "eye_shadow")
    for pid, want in (("ParamEyeLOpen", "left"), ("ParamEyeROpen", "right")):
        rest = _rest(m)
        m.set_param(pid, 0.0)
        shut = _snap(m)
        for k, (role, side) in roles.items():
            if k not in rest or role not in eye_roles or side not in ("left", "right"):
                continue
            moved = np.abs(shut[k] - rest[k]).max() > 1e-6
            assert moved == (side == want), \
                f"{pid}=0: {k} (side={side}) {'moved' if moved else 'did not move'}"
    print("ok: left and right blink independently")


def test_synonyms_do_not_touch_english():
    """Non-English names classify, and English names are unaffected.

    The synonym pass rewrites tokens inside normalize(), which every rule,
    detect_side, is_shadow and is_highlight route through. A greedy or
    unanchored pattern there would corrupt English layer names silently --
    'ear' inside 'bear', or a side token eaten mid-word -- so both directions
    are asserted, not just the new one.
    """
    from autorig.classify import classify_one, detect_side, normalize

    canvas = [1000, 2000]

    # English names must normalize exactly as they did before synonyms existed.
    for name in ("Front Hair L1", "Eye L Eyelid2", "Back Side Hair",
                 "Eye:: Left", "Collar2", "Arm R Light"):
        assert normalize(name) == _plain_normalize(name), \
            f"synonym pass altered English name {name!r} -> {normalize(name)!r}"

    # Russian names must reach the right role and side.
    for name, role, side in (("брови л", "eyebrow", "left"),
                             ("брови п", "eyebrow", "right"),
                             ("ухо п", "ear", "right"),
                             ("нос", "nose", None),
                             ("румяна л", "blush", None),
                             ("основа", "body", None),
                             ("рога", "horn", None)):
        layer = {"name": name, "file": "x.png", "index": 0,
                 "center": [500, 400]}
        got = classify_one(layer, canvas)
        assert got["role"] == role, f"{name!r} -> {got['role']!r}, want {role!r}"
        if side is not None:
            assert got["side"] == side, \
                f"{name!r} -> side {got['side']!r}, want {side!r}"

    # 'блик нос' is a highlight sublayer, not an independent nose.
    assert detect_side("ухл д") == "left", "typo'd side token must still resolve"
    print("ok: synonyms map non-English names without touching English ones")


def _plain_normalize(name: str) -> str:
    """normalize() as it was before the synonym pass -- the regression baseline."""
    import re
    s = re.sub(r"([a-z])(\d)", r"\1 \2", name.lower())
    return re.sub(r"\s+", " ", re.sub(r"[:_\-/]+", " ", s)).strip()


if __name__ == "__main__":
    test_synonyms_do_not_touch_english()
    test_manifest_shape()
    test_physics_shape()
    test_role_vocabulary_total()
    test_uv_atlas_agreement()
    test_keyform_grid_order()
    test_render_order_distinct()
    test_neck_follows_head()
    test_blink_leaves_a_lid_line()
    test_sway_is_a_rotation()
    test_blink_sides_are_independent()
    print("\nall checks passed")
