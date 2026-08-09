"""Stage 3c -- the actual rig: parameters and the deformations they drive.

Cubism's convention is that a keyform stores absolute vertex positions for one
cell of the parameter grid, so "rigging" here means: for each mesh, for each
combination of parameter values, compute where its vertices should be.

The deformations are analytic rather than art-directed. A real rigger draws
each keyform by hand; we approximate the same intent with transforms whose
magnitude comes from the layer's role and its position relative to the head
pivot. That yields a rig that moves correctly and plausibly, and which a human
can refine -- not one that matches a hand rig pixel for pixel.

Parameter IDs are the standard Live2D ones. VTube Studio and the SDK bind by
ID, so renaming them breaks tracking integration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Standard parameter set. Keys are the values at which keyforms are authored --
# three-point (min, default, max) is what the Editor generates for angles.
STANDARD_PARAMS = [
    # (id,               min,   max,  default, keys)
    ("ParamAngleX",     -30.0,  30.0,  0.0, [-30.0, 0.0, 30.0]),
    ("ParamAngleY",     -30.0,  30.0,  0.0, [-30.0, 0.0, 30.0]),
    ("ParamAngleZ",     -30.0,  30.0,  0.0, [-30.0, 0.0, 30.0]),
    ("ParamEyeLOpen",     0.0,   1.0,  1.0, [0.0, 1.0]),
    ("ParamEyeROpen",     0.0,   1.0,  1.0, [0.0, 1.0]),
    ("ParamEyeBallX",    -1.0,   1.0,  0.0, [-1.0, 0.0, 1.0]),
    ("ParamEyeBallY",    -1.0,   1.0,  0.0, [-1.0, 0.0, 1.0]),
    ("ParamBrowLY",      -1.0,   1.0,  0.0, [-1.0, 0.0, 1.0]),
    ("ParamBrowRY",      -1.0,   1.0,  0.0, [-1.0, 0.0, 1.0]),
    ("ParamMouthOpenY",   0.0,   1.0,  0.0, [0.0, 1.0]),
    ("ParamMouthForm",   -1.0,   1.0,  0.0, [-1.0, 0.0, 1.0]),
    ("ParamBodyAngleX", -10.0,  10.0,  0.0, [-10.0, 0.0, 10.0]),
    ("ParamBodyAngleZ", -10.0,  10.0,  0.0, [-10.0, 0.0, 10.0]),
    ("ParamBreath",       0.0,   1.0,  0.0, [0.0, 1.0]),
]

# Roles that belong to the head and therefore follow head rotation.
#
# Every role classify.py can emit must appear in exactly one of these two sets.
# A role in neither silently falls through to the body group -- a blush would
# then rig to the torso and slide off the cheek on every head turn. Locked by
# test_role_vocabulary_total.
HEAD_ROLES = {
    "face", "eye_white", "eyeball", "eyelid", "eyelash", "eye_light",
    "eye_shadow", "eyebrow", "mouth", "mouth_inner", "nose", "ear",
    "hair_front", "hair_side", "hair_back", "hair", "hair_ahoge",
    "accessory", "ribbon", "blush",
}
# Roles that belong to the body and sway with it, not with the head.
BODY_ROLES = {"body", "neck", "arm", "hand", "leg", "skirt", "skirt_frill",
              "tail", "collar", "breast", "unknown"}

# Roles that sit on the boundary: anchored to the body but carried partly by the
# head. They get BOTH transforms, weighted by ROLE_FOLLOW. Without this the neck
# is rigid while the head turns, and the jaw visibly detaches from the throat.
NECK_ROLES = {"neck", "collar"}

# How strongly the head rotation carries each role. Hair overshoots slightly --
# it is further from the pivot and reads as following the head.
ROLE_FOLLOW = {
    "hair_front": 1.05, "hair_side": 1.05, "hair_back": 0.95,
    "hair": 1.0, "hair_ahoge": 1.15,
    "ear": 1.02, "accessory": 1.0, "ribbon": 1.0,
    "blush": 1.0,
    # Applied via NECK_ROLES, not HEAD_ROLES: a neck that followed the head at
    # full strength would tear away from the shoulders.
    "neck": 0.35, "collar": 0.20,
}


@dataclass
class RigContext:
    """Geometry the deformations are computed against, in model space."""
    head_pivot: np.ndarray      # (2,) rotation centre for head motion
    body_pivot: np.ndarray      # (2,) rotation centre for body motion
    head_radius: float          # scale for translation magnitudes
    unit: float                 # model units per original pixel


def _rotate(verts: np.ndarray, pivot: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0.0:
        return verts
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (verts - pivot) @ r.T + pivot


def _scale_about(verts: np.ndarray, pivot: np.ndarray,
                 sx: float, sy: float) -> np.ndarray:
    if sx == 1.0 and sy == 1.0:
        return verts
    out = verts - pivot
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out + pivot


def head_transform(verts: np.ndarray, ctx: RigContext, follow: float,
                   angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Approximate a head turn.

    Cubism heads are rigged as a 2.5D illusion: yaw and pitch translate and
    shear the face rather than rotating it in plane, because a flat drawing has
    no depth to rotate through. Z is a true in-plane roll.
    """
    v = verts.astype(np.float64).copy()
    p = ctx.head_pivot
    r = ctx.head_radius

    # Yaw: slide horizontally and squash slightly on the far side.
    if angle_x:
        f = angle_x / 30.0
        v[:, 0] += f * r * 0.30 * follow
        v = _scale_about(v, p, 1.0 - abs(f) * 0.04, 1.0)
        # subtle vertical arc so the turn does not read as a flat slide
        v[:, 1] -= abs(f) * r * 0.015

    # Pitch: slide vertically, and foreshorten the face height.
    if angle_y:
        f = angle_y / 30.0
        v[:, 1] += f * r * 0.22 * follow
        v = _scale_about(v, p, 1.0, 1.0 - abs(f) * 0.05)

    # Roll: real in-plane rotation about the neck pivot.
    if angle_z:
        v = _rotate(v, p, angle_z * follow)

    return v


def body_transform(verts: np.ndarray, ctx: RigContext, weight: float,
                   angle_x: float, angle_z: float) -> np.ndarray:
    """Body lean. weight scales with height above the body pivot, so the hips
    stay put and the shoulders move -- a rigid translation would detach the
    character from the ground."""
    v = verts.astype(np.float64).copy()
    p = ctx.body_pivot
    if angle_x:
        f = angle_x / 10.0
        h = np.clip((v[:, 1] - p[1]) / max(ctx.head_radius * 2.0, 1e-6), 0.0, 1.5)
        v[:, 0] += f * ctx.head_radius * 0.10 * weight * h
    if angle_z:
        v = _rotate(v, p, angle_z * 0.35 * weight)
    return v


# How each eye part behaves at openness=0, as (residual height, anchor height).
#   residual: fraction of original height kept when fully closed. Zero makes the
#             part vanish; a closed eye is a LINE, not an absence.
#   anchor:   position within the part's own bbox that stays put, 0=bottom.
#
# These differ per part because a real blink is not one uniform squash: the lash
# line sweeps down and stays visible, the sclera is occluded entirely, and the
# lid skin compresses toward the crease.
BLINK_PROFILE = {
    "eyelid":     (0.06, 0.30),   # lid skin -- stays as the closed lid line
    "eyelash":    (0.10, 0.28),   # lashes remain visible along the seam
    "eye_white":  (0.00, 0.34),   # sclera fully hidden under the lid
    "eyeball":    (0.00, 0.34),   # iris hidden with it
    "eye_light":  (0.00, 0.34),
    "eye_shadow": (0.05, 0.30),
}
_BLINK_DEFAULT = (0.0, 0.34)


def neck_transform(verts: np.ndarray, ctx: RigContext, follow: float,
                   angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Head motion partially carried into a body-anchored part.

    The neck is the one place where the head/body split is a lie: it belongs to
    the body but the head drags it. Applying head_transform at reduced `follow`
    is not enough on its own, because that moves the whole neck uniformly and
    detaches it from the shoulders. So the head's displacement is additionally
    ramped by height -- full at the top where it meets the jaw, zero at the
    bottom where it meets the torso.
    """
    moved = head_transform(verts, ctx, follow, angle_x, angle_y, angle_z)
    v = verts.astype(np.float64)
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    span = max(y1 - y0, 1e-6)
    w = np.clip((v[:, 1] - y0) / span, 0.0, 1.0)[:, None]
    return v + (moved - v) * w


def blink(verts: np.ndarray, openness: float, bbox: tuple[float, float, float, float],
          role: str) -> np.ndarray:
    """Close an eye part toward its lid line as openness goes to 0.

    Anchoring below centre (not at the middle) is what makes a blink read as an
    eyelid dropping rather than the eye shrinking symmetrically.

    The residual height matters as much as the anchor. Scaling every part to
    exactly zero -- which this did originally -- makes the closed eye read as a
    hole in the face, because the lash and lid geometry disappears along with
    the sclera. Keeping a few percent leaves a drawn line where the closed lid
    belongs. It is still an approximation: a hand rig draws a dedicated
    closed-lid keyform, which no analytic transform can invent from open-eye art.
    """
    if openness >= 1.0:
        return verts
    v = verts.astype(np.float64).copy()
    _, y0, _, y1 = bbox
    span = y1 - y0
    if span <= 0.0:
        return v

    residual, anchor_f = BLINK_PROFILE.get(role, _BLINK_DEFAULT)
    t = max(openness, 0.0)
    # Never scale below `residual`: at t=0 the part keeps that fraction of height.
    scale = residual + (1.0 - residual) * t
    anchor = y0 + span * anchor_f
    v[:, 1] = anchor + (v[:, 1] - anchor) * scale

    # A closing lid also rides downward slightly -- the lash line travels to
    # where the eye actually shuts rather than compressing in place.
    v[:, 1] -= (1.0 - t) * span * 0.10
    return v


def eyeball_look(verts: np.ndarray, ctx: RigContext, bbox: tuple,
                 bx: float, by: float) -> np.ndarray:
    """Shift the iris within the eye. Range is a fraction of the eye's own
    width so small eyes do not have the iris slide outside the sclera."""
    v = verts.astype(np.float64).copy()
    x0, y0, x1, y1 = bbox
    v[:, 0] += bx * (x1 - x0) * 0.22
    v[:, 1] += by * (y1 - y0) * 0.22
    return v


def brow_raise(verts: np.ndarray, amount: float, bbox: tuple) -> np.ndarray:
    v = verts.astype(np.float64).copy()
    _, y0, _, y1 = bbox
    v[:, 1] += amount * (y1 - y0) * 0.5
    return v


def mouth_open(verts: np.ndarray, amount: float, bbox: tuple) -> np.ndarray:
    """Open downward from the upper lip, which is where a jaw hinges."""
    if amount <= 0.0:
        return verts
    v = verts.astype(np.float64).copy()
    _, y0, _, y1 = bbox
    top = y1
    v[:, 1] = top - (top - v[:, 1]) * (1.0 + amount * 0.9)
    return v


def mouth_form(verts: np.ndarray, amount: float, bbox: tuple) -> np.ndarray:
    """Smile/frown: widen and lift the corners."""
    v = verts.astype(np.float64).copy()
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    half = max((x1 - x0) * 0.5, 1e-6)
    t = np.clip(np.abs(v[:, 0] - cx) / half, 0.0, 1.0)
    v[:, 0] = cx + (v[:, 0] - cx) * (1.0 + amount * 0.12)
    v[:, 1] += amount * (y1 - y0) * 0.30 * t
    return v


def breath(verts: np.ndarray, ctx: RigContext, amount: float, weight: float) -> np.ndarray:
    """Chest rise. Small, but its absence is what makes a model look dead."""
    if amount <= 0.0 or weight <= 0.0:
        return verts
    v = verts.astype(np.float64).copy()
    v[:, 1] += amount * ctx.head_radius * 0.012 * weight
    return v


# Peak swing angle in degrees at |amount|=1, per role. A pendulum's arc, not a
# shear: the strand rotates about where it attaches.
SWAY_ANGLE = {
    "hair_front": 8.0, "hair_side": 12.0, "hair_back": 12.0,
    "hair": 10.0, "hair_ahoge": 22.0,
    "ribbon": 18.0, "accessory": 8.0, "ear": 6.0,
    "tail": 20.0, "skirt": 7.0, "skirt_frill": 9.0,
    "breast": 4.0,
}
_SWAY_DEFAULT = 10.0


def sway(verts: np.ndarray, ctx: RigContext, amount: float,
         bbox: tuple[float, float, float, float], role: str) -> np.ndarray:
    """Physics-driven swing of a hanging part.

    Modelled as a rotation about the part's own attachment point (the centre of
    its top edge), with the angle ramped in by depth so the strand bends rather
    than swinging rigidly like a plank.

    This was originally a horizontal shear weighted by depth. A shear stretches
    the part sideways instead of swinging it, which is what made the tail read as
    offset to one side rather than swaying -- at full deflection its tip
    translated 0.43 model units horizontally while its length stayed constant,
    a motion no physical pendulum makes.

    This is the motion physics3.json re-drives with lag; on its own the
    parameter just poses the strand.
    """
    if amount == 0.0:
        return verts
    v = verts.astype(np.float64).copy()
    x0, y0, x1, y1 = bbox
    span = max(y1 - y0, 1e-6)

    # Pivot: centre of the top edge -- where hair meets scalp, a tail meets hip.
    pivot = np.array([(x0 + x1) * 0.5, y1], dtype=np.float64)

    # Depth below the anchor, 0 at the top and 1 at the free end. Squaring it
    # makes the bend accelerate toward the tip, which is how a flexible strand
    # actually deflects -- a linear ramp reads as a rigid hinge.
    t = np.clip((y1 - v[:, 1]) / span, 0.0, 1.0) ** 1.5

    theta = math.radians(SWAY_ANGLE.get(role, _SWAY_DEFAULT) * amount)
    # Per-vertex rotation angle: the strand curves progressively.
    ang = theta * t
    c, s = np.cos(ang), np.sin(ang)
    d = v - pivot
    v[:, 0] = pivot[0] + d[:, 0] * c - d[:, 1] * s
    v[:, 1] = pivot[1] + d[:, 0] * s + d[:, 1] * c
    return v
