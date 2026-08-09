"""Stage 2 — classify flat PSD layers into roles, sides, and a parent tree.

Produces rig_plan.json. This is the one fuzzy stage. It is written as scored
rules rather than an LLM call because artist layer names, while not
*standardized*, are highly regular within a file ("Eye L Eyelid2 Shadow",
"Front Hair R1 Light"). Rules get the common cases with a confidence score;
anything scoring low is surfaced for human review instead of being silently
mis-rigged.

An LLM can still be used here — override roles by hand-editing rig_plan.json,
or call `plan()` and patch its output. The file boundary is the point.

Ground truth for the role/parent conventions below is the hand-rigged Aka model
(samples/example-models/Aka_real.inx), whose tree is:
    Root > Skirt > {Tail, Ribbon, Frill}
    Root > Body > {Arm L/R > Hand, Neck > Face > {Eyes, Mouth, Hair, Brows}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Role rules: (role, pattern, parent_role, score).
# Order matters — first match wins, so specific patterns precede general ones.
# Patterns match against the lowercased layer name.
RULES: list[tuple[str, str, str | None, float]] = [
    # --- face interior: must precede 'hair'/'face' since names overlap -----
    ("eye_light",    r"\beye\b.*\blight\b",                 "eyeball",   0.9),
    ("eyeball",      r"\beye\b.*(eyeball|pupil|iris)",      "eye_white", 0.95),
    ("eye_white",    r"\beye\b.*white",                     "eyelid",    0.95),
    # 'eyeslash' is a typo in the real Midori PSD; artists misspell constantly.
    ("eyelash",      r"eyelash|eyeslash|eye\s*lash",        "eyelid",    0.9),
    ("eyelid",       r"eyelid",                             "face",      0.9),
    ("eye_shadow",   r"\beye\b.*shadow",                    "eyelid",    0.85),
    ("eyebrow",      r"eyebrow|\bbrow\b",                   "face",      0.95),
    ("mouth_inner",  r"mouth.*inner|\bmouth\b.*cavity",     "mouth",     0.9),
    ("mouth",        r"\bmouth\b|\blip\b|\bteeth\b|tongue", "face",      0.9),
    ("nose",         r"\bnose\b",                           "face",      0.9),
    ("blush",        r"blush|cheek",                        "face",      0.8),
    # --- hair: front/side/back drive different physics ---------------------
    # 'back side hair' must precede 'side hair' — it is back hair, not side.
    # Verified against the hand rig: 'Back Side:: Hair' parents under Face and
    # carries its own physics chain.
    ("hair_back",    r"back\s*side\s*hair|back\s*hair|\bhair\b.*\bback\b", "face", 0.9),
    ("hair_front",   r"front\s*hair|bangs",                 "face",      0.9),
    ("hair_side",    r"side\s*hair",                        "face",      0.9),
    ("hair_ahoge",   r"ahoge|antenna",                      "head",      0.85),
    ("hair",         r"\bhair\b",                           "head",      0.7),
    # --- head/face base ----------------------------------------------------
    ("ear",          r"\bear\b|\bears\b",                   "head",      0.85),
    ("face",         r"\bface\b|\bhead\b",                  "neck",      0.9),
    ("neck",         r"\bneck\b",                           "body",      0.9),
    # --- limbs -------------------------------------------------------------
    ("hand",         r"\bhand\b|\bfist\b|\bfinger",         "arm",       0.9),
    ("arm",          r"\barm\b|sleeve|shoulder",            "body",      0.9),
    ("leg",          r"\bleg\b|\bthigh\b|\bfoot\b|\bshoe",  "body",      0.9),
    # --- torso & clothing --------------------------------------------------
    ("tail",         r"\btail\b",                           "skirt",     0.85),
    ("ribbon",       r"ribbon|bow\b|tie\b",                 "skirt",     0.8),
    ("skirt_frill",  r"skirt.*frill|frill",                 "skirt",     0.85),
    ("skirt",        r"\bskirt\b|\bdress\b",                "body",      0.9),
    ("collar",       r"\bcollar\b|\bchoker\b",              "neck",      0.85),
    ("breast",       r"\bbreast\b|\bbust\b",                "body",      0.85),
    ("accessory",    r"\bleaf\b|\bhat\b|\bcap\b|clip|pin\b", "head",     0.6),
    ("body",         r"\bbody\b|\btorso\b|\bchest\b",       None,        0.9),
]

# Physics-driven roles: hair and dangling cloth sway. Matches the hand rig,
# which puts SimplePhysics on skirt, ribbon, sleeves and tail.
PHYSICS_ROLES = {"hair_back", "hair_side", "hair_ahoge", "tail", "ribbon",
                 "skirt", "skirt_frill", "breast"}

# Roles whose left/right variants are separate rig targets.
SIDED_ROLES = {"eyelid", "eyeball", "eye_white", "eye_light", "eyelash",
               "eye_shadow", "eyebrow", "arm", "hand", "leg", "ear",
               "hair_front", "hair_side", "hair_back", "breast"}


def normalize(name: str) -> str:
    """Lowercase and split trailing digits off words so \\b rules still match.

    'Collar2' -> 'collar 2', 'Eyelid2' -> 'eyelid 2'. Without this, every
    \\b-anchored rule silently fails on numbered layers — and numbered layers
    are everywhere in real PSDs ('Front Hair L1', 'Eye L Eyelid2').
    Separators are normalized too, so 'Eye:: Left' and 'eye_left' both work.
    """
    s = re.sub(r"([a-z])(\d)", r"\1 \2", name.lower())
    return re.sub(r"\s+", " ", re.sub(r"[:_\-/]+", " ", s)).strip()


def detect_side(name: str) -> str | None:
    """Extract left/right/center from a layer name.

    Token-based on purpose: a substring search for 'r' or 'l' would match
    inside almost every word. Handles 'Eye L', 'Arm R', 'Front Hair L1',
    'Eye:: Left', 'hair_l2', and 'Front Hair C' (center — a real position in
    the reference art, not a missing side).
    """
    n = normalize(name)
    if re.search(r"\bleft\b", n):
        return "left"
    if re.search(r"\bright\b", n):
        return "right"
    if re.search(r"\bcenter\b|\bcentre\b", n):
        return "center"
    # single-letter forms: ' L ', ' L1', '_l2', ' L' at end
    if re.search(r"(?:^|[\s_:])l(?:\d+)?(?:[\s_:]|$)", n):
        return "left"
    if re.search(r"(?:^|[\s_:])r(?:\d+)?(?:[\s_:]|$)", n):
        return "right"
    if re.search(r"(?:^|[\s_:])c(?:\d+)?(?:[\s_:]|$)", n):
        return "center"
    return None


def variant_index(name: str) -> int | None:
    """Strand/variant number attached to a side token: 'Front Hair L2' -> 2.

    Distinguishes sibling parts that share role and side. Only digits bound to
    a side letter or trailing the name count — '2' in 'Eyelid2' is a variant,
    but a bare year or size in a name is not treated as one.
    """
    n = normalize(name)
    m = re.search(r"(?:^|\s)[lrc]\s*(\d+)(?:\s|$)", n)
    if m:
        return int(m.group(1))
    m = re.search(r"[a-z]\s+(\d+)\s*$", n)
    return int(m.group(1)) if m else None


def is_shadow(name: str) -> bool:
    return bool(re.search(r"\bshadow\b|\bshade\b", normalize(name)))


def is_highlight(name: str) -> bool:
    # 'Ligth' is a typo present in the real reference PSD — match it too.
    return bool(re.search(r"\blight\b|\bligth\b|\bhighlight\b", normalize(name)))


def owner_name(name: str) -> str | None:
    """For a Light/Shadow sublayer, the base layer it belongs to.

    'Front Hair R2 Light' -> 'Front Hair R2'. In the hand rig these parent to
    their owner part, not to the face, so they inherit its deformation.
    Returns None when the name is not a sublayer.
    """
    m = re.match(r"^(.*?)[\s_:]+(shadow|shade|light|ligth|highlight)\s*$",
                 name.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else None


def classify_one(layer: dict, canvas: list[int]) -> dict:
    """Assign role/side/parent to a single layer with a confidence score."""
    name = layer["name"]
    low = normalize(name)

    role, parent_role, score = None, None, 0.0
    for r, pattern, parent, sc in RULES:
        if re.search(pattern, low):
            role, parent_role, score = r, parent, sc
            break

    if role is None:
        # Geometry fallback: unknown layers in the upper third of a canvas are
        # far more likely head accessories than body parts.
        cy = layer["center"][1] / canvas[1]
        if cy < 0.33:
            role, parent_role, score = "accessory", "head", 0.3
        else:
            role, parent_role, score = "unknown", "body", 0.2

    side = detect_side(name) if role in SIDED_ROLES else None
    if role in SIDED_ROLES and side is None:
        # A sided role with no side token is genuinely ambiguous — use x
        # position, but say so with a lower score.
        cx = layer["center"][0] / canvas[0]
        if abs(cx - 0.5) > 0.04:
            # Off-center with no side token: geometry is a real but weaker
            # signal than a name, so flag it for review.
            side = "left" if cx < 0.5 else "right"
            score = min(score, 0.55)
        else:
            # Centered and unsided is not ambiguity — it is a center part
            # ('Back Side Hair', 'Front Hair C'). Keep the role's confidence.
            side = "center"

    shadow, highlight = is_shadow(name), is_highlight(name)
    sub = shadow or highlight
    if sub:
        # Shadows/highlights follow their owner; never independent rig targets.
        score = min(score, 0.8)

    return {
        "layer": layer["name"],
        "file": layer["file"],
        "index": layer["index"],
        "role": role,
        "side": side,
        # Strand/variant index: 'Front Hair L2' -> 2. Distinct parts that share
        # a role AND a side; without it L1/L2/L3 collapse into one name.
        "variant": variant_index(name),
        "parent_role": parent_role,
        "parent_layer": owner_name(name) if sub else None,  # resolved in plan()
        "is_shadow": shadow,
        "is_highlight": highlight,
        "physics": role in PHYSICS_ROLES and not sub,
        "confidence": round(score, 2),
    }


def build_params(parts: list[dict]) -> list[dict]:
    """Standard VTuber parameter set, emitted only for roles actually present.

    Names and 2D-vs-1D choices mirror the hand-rigged reference model
    ('Head:: Yaw-Pitch' is one 2D param, not two 1D sliders).
    """
    roles = {p["role"] for p in parts}
    sides = {(p["role"], p["side"]) for p in parts}
    params: list[dict] = []

    def add(name, kind, drives, note):
        params.append({"name": name, "type": kind, "drives": drives, "note": note})

    if {"face", "neck"} & roles:
        add("Head:: Yaw-Pitch", "2d", ["face"], "head turn; carries eyes/mouth/hair")
        add("Head:: Roll", "1d", ["face"], "head tilt")
    for side in ("left", "right"):
        if any(r == "eyelid" and s == side for r, s in sides):
            add(f"Eye:: {side.title()}:: Blink", "1d", [f"eyelid:{side}"], "0=closed 1=open")
        if any(r == "eyeball" and s == side for r, s in sides):
            add(f"Eye:: {side.title()}:: Move", "2d", [f"eyeball:{side}"], "pupil tracking")
    if "eyebrow" in roles:
        add("Brow:: Emotion", "2d", ["eyebrow"], "up/down + angry/sad form")
    for side in ("left", "right"):
        if any(r == "eyelid" and s == side for r, s in sides):
            add(f"Eye:: {side.title()}:: Smile", "1d", [f"eyelid:{side}"], "0=neutral 1=smiling arc")
    if roles & {"mouth", "mouth_inner"}:
        add("Mouth:: Shape", "2d", ["mouth"], "open/close + form + x-shift")
    if "body" in roles:
        add("Body:: Yaw-Pitch", "2d", ["body"], "torso lean")
        add("Body:: Roll", "1d", ["body"], "torso tilt")
        add("Breath", "1d", ["body"], "idle breathing loop")
    for side in ("left", "right"):
        if any(r == "arm" and s == side for r, s in sides):
            add(f"Arm:: {side.title()}:: Move", "1d", [f"arm:{side}"], "arm swing")
    for p in parts:
        if p["physics"]:
            key = f"{p['role']}:{p['side']}" if p["side"] else p["role"]
            add(f"{p['layer']}:: Physics", "1d", [key], "driven by SimplePhysics")
    return params


def plan(manifest: dict | str | Path) -> dict:
    """layers.json (dict or path) -> rig_plan dict."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())

    canvas = manifest["canvas"]
    parts = [classify_one(l, canvas) for l in manifest["layers"]]

    # Resolve sublayer -> owner. A 'Front Hair R2 Light' whose 'Front Hair R2'
    # is absent must fall back to role-parenting, or stage 3 gets a dangling
    # parent reference.
    names = {p["layer"] for p in parts}
    for p in parts:
        if p["parent_layer"] and p["parent_layer"] not in names:
            p["parent_layer"] = None

    low = [p for p in parts if p["confidence"] < 0.6]

    return {
        "source": manifest.get("source"),
        "canvas": canvas,
        "root": "body",
        "parts": parts,
        "params": build_params(parts),
        "review": {
            "low_confidence_count": len(low),
            "layers": [
                {"layer": p["layer"], "role": p["role"], "confidence": p["confidence"]}
                for p in low
            ],
        },
    }


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Stage 2: layers.json -> rig_plan.json")
    ap.add_argument("manifest", help="path to layers.json")
    ap.add_argument("-o", "--out", default=None, help="output path (default: alongside manifest)")
    a = ap.parse_args(argv)

    src = Path(a.manifest)
    p = plan(src)
    out = Path(a.out) if a.out else src.parent / "rig_plan.json"
    out.write_text(json.dumps(p, indent=2, ensure_ascii=False))

    roles: dict[str, int] = {}
    for part in p["parts"]:
        roles[part["role"]] = roles.get(part["role"], 0) + 1
    print(f"{len(p['parts'])} parts, {len(p['params'])} params -> {out}")
    print("roles:", ", ".join(f"{k}={v}" for k, v in sorted(roles.items())))
    if p["review"]["low_confidence_count"]:
        print(f"needs review: {p['review']['low_confidence_count']} layers")
        for r in p["review"]["layers"]:
            print(f"  {r['confidence']:.2f}  {r['layer']}  -> {r['role']}")


if __name__ == "__main__":
    main()
