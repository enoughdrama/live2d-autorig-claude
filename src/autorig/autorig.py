"""Stage 3 driver -- PSD layers + rig plan -> a fully rigged Moc3.

Decides, per layer, which parameters drive it and what its geometry looks like
at every cell of that parameter grid.

Binding width is deliberately conservative. The keyform count is the *product*
of the bound parameters' key counts, so binding one extra 3-key parameter
triples a mesh's keyform storage. Face parts get head angles plus their own
feature parameter; body parts get body angles only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import classify as _classify
from .build import ArtMeshSpec, RigBuilder
from .mesh import build_mesh
from .rig import (
    BODY_ROLES,
    HEAD_ROLES,
    NECK_ROLES,
    ROLE_FOLLOW,
    STANDARD_PARAMS,
    RigContext,
    body_transform,
    breath,
    blink,
    brow_form,
    brow_raise,
    eye_smile,
    eyeball_look,
    head_transform,
    mouth_form,
    mouth_open,
    mouth_x,
    neck_transform,
    sway,
)

# Which extra parameter a role gets, on top of the head/body angles.
ROLE_PARAM = {
    "eyelid": "ParamEyeLOpen",       # resolved to L/R by side below
    "eyelash": "ParamEyeLOpen",
    "eye_white": "ParamEyeLOpen",
    "eye_shadow": "ParamEyeLOpen",
    "eye_light": "ParamEyeLOpen",
    "eyeball": "ParamEyeBallX",
    "eyebrow": "ParamBrowLY",
    "mouth": "ParamMouthOpenY",
    "mouth_inner": "ParamMouthOpenY",
}

# Roles that breathe (torso and what sits on it).
BREATH_ROLES = {"body", "neck", "arm", "skirt", "skirt_frill", "collar", "breast"}

# Roles that get a physics-driven sway parameter. physics3.json writes into
# these; without a parameter that actually deforms, physics has nothing to drive.
#
# Sourced from classify.PHYSICS_ROLES rather than restated. These were two
# independent literals that had already drifted apart in both directions --
# classify marked breast/hair_ahoge as physics parts that never got a sway
# parameter, while the rig built sway for accessory/ear/hair_front that classify
# did not consider physical. The classifier owns the vocabulary; the extras here
# are roles the rig genuinely animates that classify treats as static.
PHYSICS_ROLES = set(_classify.PHYSICS_ROLES) | {
    "hair_front",   # bangs sway, just less than side/back hair
    "accessory",    # clips and pins hang off the head
    "ear",          # animal ears flick
}


def _bbox_model(verts: np.ndarray) -> tuple[float, float, float, float]:
    return (float(verts[:, 0].min()), float(verts[:, 1].min()),
            float(verts[:, 0].max()), float(verts[:, 1].max()))


def _is_right(side: str | None) -> bool:
    # stage 2 emits "left"/"right"/"center"/None -- not "L"/"R". Comparing
    # against "R" silently sent every right-side part to the left parameter,
    # leaving ParamEyeROpen and ParamBrowRY driving nothing.
    return (side or "").lower().startswith("r")


def _resolve_param(role: str, side: str | None) -> str | None:
    """Pick the side-correct parameter id for a role."""
    base = ROLE_PARAM.get(role)
    if base is None:
        return None
    if base == "ParamEyeLOpen":
        return "ParamEyeROpen" if _is_right(side) else "ParamEyeLOpen"
    if base == "ParamBrowLY":
        return "ParamBrowRY" if _is_right(side) else "ParamBrowLY"
    return base


def _brow_form_param(side: str | None) -> str:
    return "ParamBrowRForm" if _is_right(side) else "ParamBrowLForm"


def build_rig(build_dir: str | Path, pixels_per_unit: float = 1000.0,
              spacing: float = 90.0) -> RigBuilder:
    """Assemble a rigged model from stage 1/2 output in build_dir."""
    build_dir = Path(build_dir)
    layers = json.loads((build_dir / "layers.json").read_text())
    plan = json.loads((build_dir / "rig_plan.json").read_text())
    cw, ch = layers["canvas"]
    by_file = {l["file"]: l for l in layers["layers"]}

    rb = RigBuilder(cw, ch, pixels_per_unit)
    pidx = {}
    for pid, lo, hi, dflt, keys in STANDARD_PARAMS:
        pidx[pid] = rb.add_param(pid, lo, hi, dflt, keys)

    # Physics can only re-drive motion that already exists, so every part that
    # should have inertia needs its own sway parameter with real deformation
    # behind it. physics3.json then writes into these.
    sway_params: list[tuple[str, str, str]] = []

    # Mesh everything first: the head pivot is derived from where the face
    # parts actually are, not from a guess about canvas layout.
    meshed = []
    for part in plan["parts"]:
        rec = by_file[part["file"]]
        x0, y0, _, _ = rec["bbox"]
        mesh = build_mesh(str(build_dir / part["file"]), x0, y0, cw, ch,
                          pixels_per_unit=pixels_per_unit, spacing=spacing)
        if mesh is None:
            continue
        meshed.append((part, mesh))

    ctx = _make_context(meshed, cw, ch, pixels_per_unit)

    root = rb.add_part("Root")
    part_ids = {}
    for group in ("Head", "Body"):
        part_ids[group] = rb.add_part(group, root)

    for order, (part, mesh) in enumerate(meshed):
        role = part["role"]
        side = part.get("side")
        name = _safe_name(part["layer"])
        group = "Head" if role in HEAD_ROLES else "Body"
        part_index = rb.add_part(name, part_ids[group])

        sway_id = None
        if role in PHYSICS_ROLES:
            sway_id = f"ParamSway{len(sway_params):02d}"
            pidx[sway_id] = rb.add_param(sway_id, -1.0, 1.0, 0.0, [-1.0, 0.0, 1.0])
            sway_params.append((name.replace("_", " "), sway_id, role, side))

        bound, deform = _plan_deformation(role, side, mesh, ctx, pidx, sway_id)

        rb.add_mesh(ArtMeshSpec(
            name=name,
            mesh=mesh,
            part_index=part_index,
            draw_order=500 + order,
            bound_params=bound,
            deform=deform,
        ))

    rb.sway_params = sway_params
    return rb


def _make_context(meshed, cw: int, ch: int, ppu: float) -> RigContext:
    """Head pivot at the base of the head, body pivot at the hips."""
    face = [m for p, m in meshed if p["role"] in ("face", "neck")]
    head_parts = [m for p, m in meshed if p["role"] in HEAD_ROLES]

    if face:
        pts = np.vstack([m.verts for m in face])
    elif head_parts:
        pts = np.vstack([m.verts for m in head_parts])
    else:
        pts = np.vstack([m.verts for _, m in meshed])

    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    head_pivot = np.array([(x0 + x1) * 0.5, y0], dtype=np.float64)
    head_radius = float(max(x1 - x0, y1 - y0)) * 0.5

    body = [m for p, m in meshed if p["role"] in BODY_ROLES]
    if body:
        bpts = np.vstack([m.verts for m in body])
        body_pivot = np.array([(bpts[:, 0].min() + bpts[:, 0].max()) * 0.5,
                               bpts[:, 1].min()], dtype=np.float64)
    else:
        body_pivot = np.array([0.0, -ch / (2.0 * ppu)], dtype=np.float64)

    return RigContext(head_pivot=head_pivot, body_pivot=body_pivot,
                      head_radius=head_radius, unit=1.0 / ppu)


def _plan_deformation(role, side, mesh, ctx, pidx, sway_id=None):
    """Return (bound parameter indices, deform callable).

    The callable receives a tuple of key indices -- one per bound parameter, in
    the same order -- and returns absolute positions for that grid cell.
    """
    base = mesh.verts.astype(np.float64)
    bbox = _bbox_model(base)
    is_head = role in HEAD_ROLES
    is_neck = role in NECK_ROLES
    follow = ROLE_FOLLOW.get(role, 1.0)

    feature = _resolve_param(role, side)
    bound_ids: list[str] = []

    if is_head:
        bound_ids += ["ParamAngleX", "ParamAngleY", "ParamAngleZ"]
    elif is_neck:
        # Driven by both: the head drags it, the body carries it.
        #
        # Pitch (AngleY) is deliberately omitted. Keyform count is the PRODUCT
        # of bound key counts, so adding it takes the neck from 162 keyforms to
        # 486 -- 5.4 MB for a single mesh. Yaw and roll carry nearly all of the
        # visible neck motion; pitch mostly slides the head up and down, which
        # the throat barely registers.
        bound_ids += ["ParamAngleX", "ParamAngleZ",
                      "ParamBodyAngleX", "ParamBodyAngleZ"]
    else:
        bound_ids += ["ParamBodyAngleX", "ParamBodyAngleZ"]

    if not is_head and role in BREATH_ROLES:
        bound_ids.append("ParamBreath")

    if sway_id:
        bound_ids.append(sway_id)

    brow_form_id = mouth_x_bound = eye_smile_id = None
    if feature:
        bound_ids.append(feature)
        # An eyeball needs both look axes to be useful, and it must also hide
        # under the lid when the eye closes -- otherwise the lid collapses over
        # a stationary iris and the eye still reads as open.
        if feature == "ParamEyeBallX":
            bound_ids.append("ParamEyeBallY")
            bound_ids.append("ParamEyeROpen" if _is_right(side) else "ParamEyeLOpen")
        # A mouth needs both open and shape, or it can only gape.
        if feature == "ParamMouthOpenY":
            bound_ids.append("ParamMouthForm")
            bound_ids.append("ParamMouthX")
            mouth_x_bound = True
        if feature in ("ParamEyeLOpen", "ParamEyeROpen") and role in ("eyelid", "eyelash"):
            bound_ids.append("ParamEyeSmile")
            eye_smile_id = "ParamEyeSmile"
    if role == "eyebrow":
        brow_form_id = _brow_form_param(side)
        bound_ids.append(brow_form_id)

    bound = [pidx[b] for b in bound_ids]
    std_keys = dict((p[0], p[4]) for p in STANDARD_PARAMS)
    keys = [std_keys.get(b, [-1.0, 0.0, 1.0]) for b in bound_ids]

    def deform(cell):
        v = base.copy()
        vals = {bid: keys[i][cell[i]] for i, bid in enumerate(bound_ids)}

        if is_head:
            v = head_transform(v, ctx, follow,
                               vals.get("ParamAngleX", 0.0),
                               vals.get("ParamAngleY", 0.0),
                               vals.get("ParamAngleZ", 0.0))
        else:
            if is_neck:
                # Head drag first, in the rest frame, then the body carries the
                # result. Reversing this would rotate the head's contribution
                # by the body lean and double-count it.
                v = neck_transform(v, ctx, follow,
                                   vals.get("ParamAngleX", 0.0),
                                   vals.get("ParamAngleY", 0.0),
                                   vals.get("ParamAngleZ", 0.0))
            v = body_transform(v, ctx, 1.0,
                               vals.get("ParamBodyAngleX", 0.0),
                               vals.get("ParamBodyAngleZ", 0.0))
            if "ParamBreath" in vals:
                v = breath(v, ctx, vals["ParamBreath"], 1.0)

        if sway_id and vals.get(sway_id):
            v = sway(v, ctx, vals[sway_id], bbox, role)

        # Feature deformations act in the part's own local frame, so they are
        # applied to the already-posed vertices using the ORIGINAL bbox --
        # recomputing the bbox per cell would make the eyelid anchor drift.
        if feature == "ParamMouthOpenY":
            if vals.get("ParamMouthForm"):
                v = mouth_form(v, vals["ParamMouthForm"], bbox)
            v = mouth_open(v, vals[feature], bbox)
            if mouth_x_bound and vals.get("ParamMouthX"):
                v = mouth_x(v, vals["ParamMouthX"], bbox)
        elif feature in ("ParamEyeLOpen", "ParamEyeROpen"):
            v = blink(v, vals[feature], bbox, role)
            if eye_smile_id and vals.get(eye_smile_id):
                v = eye_smile(v, vals[eye_smile_id], bbox, role)
        elif feature == "ParamEyeBallX":
            v = eyeball_look(v, ctx, bbox, vals["ParamEyeBallX"],
                             vals.get("ParamEyeBallY", 0.0))
            lid = vals.get("ParamEyeROpen" if _is_right(side) else "ParamEyeLOpen")
            if lid is not None and lid < 1.0:
                v = blink(v, lid, bbox, role)
        elif feature in ("ParamBrowLY", "ParamBrowRY"):
            v = brow_raise(v, vals[feature], bbox)

        if brow_form_id and vals.get(brow_form_id):
            v = brow_form(v, vals[brow_form_id], bbox, _is_right(side))

        return v

    return bound, deform


def _safe_name(layer: str) -> str:
    out = layer.replace(" ", "_").replace(":", "")
    return out[:60] or "Part"
