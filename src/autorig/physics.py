"""Stage 5 -- physics3.json: inertia for hair, accessories and cloth.

physics3.json is an open, fully documented format (Live2D/CubismSpecs), so
unlike the rig itself this part needs no reverse engineering. Each setting is a
pendulum chain: head/body angle parameters feed in as Input, a chain of
Vertices simulates, and the result is written back out to a parameter as Angle.

The catch that shapes this module: Output can only drive parameters that
already exist and already deform something. Physics does not create motion --
it re-drives existing motion with lag. So sway parameters are created by the
rig stage, and this module wires them up.

Tuning constants are exposed rather than baked: pendulum feel is empirical and
depends on the art's proportions.
"""
from __future__ import annotations

from dataclasses import dataclass

# Per-role physical character. Longer/looser things lag more and swing wider.
#   segments: chain length (more = more whip)
#   mobility: how freely it swings
#   delay:    reaction lag -- the "inertia" knob
#   acceleration: how hard input drives it
#   radius:   pendulum arm length in physics units
@dataclass(frozen=True)
class PhysicsProfile:
    segments: int
    mobility: float
    delay: float
    acceleration: float
    radius: float
    output_scale: float


PROFILES = {
    "hair_front":  PhysicsProfile(2, 0.95, 0.75, 1.2, 12.0, 1.0),
    "hair_side":   PhysicsProfile(3, 0.95, 0.80, 1.3, 16.0, 1.1),
    "hair_back":   PhysicsProfile(3, 0.90, 0.85, 1.4, 20.0, 1.2),
    "ribbon":      PhysicsProfile(3, 1.00, 0.70, 1.5, 14.0, 1.3),
    "accessory":   PhysicsProfile(2, 0.90, 0.70, 1.1, 10.0, 1.0),
    "ear":         PhysicsProfile(2, 0.85, 0.60, 1.0, 8.0, 0.8),
    "tail":        PhysicsProfile(4, 0.95, 0.90, 1.5, 24.0, 1.3),
    "skirt":       PhysicsProfile(2, 0.80, 0.85, 1.0, 18.0, 0.9),
    "skirt_frill": PhysicsProfile(2, 0.85, 0.80, 1.1, 14.0, 1.0),
}

# What drives each kind of chain. Head-mounted things follow head angles;
# body-mounted things follow the body.
HEAD_DRIVEN = {"hair_front", "hair_side", "hair_back", "ribbon", "accessory", "ear"}


def _inputs(head_driven: bool, reflect: bool = False) -> list[dict]:
    """Inputs that drive a chain.

    `reflect` mirrors the response horizontally. A right-side strand hangs on
    the opposite side of the pivot from its left-side twin, so the same head
    turn should push it the other way; without this both sides swing in
    lockstep and the hair reads as one rigid helmet.

    Head-driven chains also take pitch, at low weight. Nodding does swing hair,
    just far less than turning does -- omitting it entirely makes a nod look
    like the head is moving inside static hair.
    """
    if head_driven:
        return [
            {"Source": {"Target": "Parameter", "Id": "ParamAngleX"},
             "Weight": 60.0, "Type": "X", "Reflect": reflect},
            {"Source": {"Target": "Parameter", "Id": "ParamAngleZ"},
             "Weight": 40.0, "Type": "Angle", "Reflect": reflect},
            {"Source": {"Target": "Parameter", "Id": "ParamAngleY"},
             "Weight": 15.0, "Type": "Y", "Reflect": False},
        ]
    return [
        {"Source": {"Target": "Parameter", "Id": "ParamBodyAngleX"},
         "Weight": 60.0, "Type": "X", "Reflect": reflect},
        {"Source": {"Target": "Parameter", "Id": "ParamBodyAngleZ"},
         "Weight": 40.0, "Type": "Angle", "Reflect": reflect},
    ]


def _vertices(profile: PhysicsProfile) -> list[dict]:
    """A chain hanging downward from the attachment point.

    The first vertex is the anchor: it has zero mobility and no delay because
    it is bolted to the head. Later vertices get progressively more delay, which
    is what produces the travelling-wave look of real hair.
    """
    out = []
    for i in range(profile.segments + 1):
        if i == 0:
            out.append({"Position": {"X": 0.0, "Y": 0.0}, "Mobility": 1.0,
                        "Delay": 1.0, "Acceleration": 1.0, "Radius": 0.0})
            continue
        t = i / profile.segments
        out.append({
            "Position": {"X": 0.0, "Y": round(profile.radius * i, 3)},
            "Mobility": round(profile.mobility, 3),
            "Delay": round(profile.delay * (0.85 + 0.3 * t), 3),
            "Acceleration": round(profile.acceleration, 3),
            "Radius": round(profile.radius, 3),
        })
    return out


def _outputs(param_id: str, profile: PhysicsProfile, n_vertices: int) -> list[dict]:
    # Drive from the last vertex: it has the most accumulated lag.
    return [{
        "Destination": {"Target": "Parameter", "Id": param_id},
        "VertexIndex": n_vertices - 1,
        "Scale": round(profile.output_scale, 3),
        "Weight": 100.0,
        "Type": "Angle",
        "Reflect": False,
    }]


def _normalization() -> dict:
    return {
        "Position": {"Minimum": -10.0, "Default": 0.0, "Maximum": 10.0},
        "Angle": {"Minimum": -10.0, "Default": 0.0, "Maximum": 10.0},
    }


def build_physics(chains: list[tuple[str, str, str, str | None]],
                  fps: int = 30) -> dict:
    """Build physics3.json.

    chains: list of (human_name, sway_parameter_id, role, side) tuples -- see
    physics_chains() which derives them from the rig plan.
    """
    settings = []
    dictionary = []
    total_in = total_out = total_v = 0

    for idx, chain in enumerate(chains):
        name, param_id, role = chain[0], chain[1], chain[2]
        side = chain[3] if len(chain) > 3 else None
        profile = PROFILES.get(role, PROFILES["accessory"])
        sid = f"PhysicsSetting{idx + 1}"
        verts = _vertices(profile)
        ins = _inputs(role in HEAD_DRIVEN,
                      reflect=(side or "").lower().startswith("r"))
        outs = _outputs(param_id, profile, len(verts))

        settings.append({
            "Id": sid,
            "Input": ins,
            "Output": outs,
            "Vertices": verts,
            "Normalization": _normalization(),
        })
        dictionary.append({"Id": sid, "Name": name})
        total_in += len(ins)
        total_out += len(outs)
        total_v += len(verts)

    return {
        "Version": 3,
        "Meta": {
            "PhysicsSettingCount": len(settings),
            "TotalInputCount": total_in,
            "TotalOutputCount": total_out,
            "VertexCount": total_v,
            "Fps": fps,
            "EffectiveForces": {
                "Gravity": {"X": 0.0, "Y": -1.0},
                "Wind": {"X": 0.0, "Y": 0.0},
            },
            "PhysicsDictionary": dictionary,
        },
        "PhysicsSettings": settings,
    }


def physics_chains(sway_params: list[tuple]) -> list[tuple]:
    """Order chains so heavier, slower parts settle behind lighter ones.

    Kept as a seam so chain selection can grow smarter (grouping strands,
    merging symmetric pairs) without touching the emitter.

    Sorting by delay is not cosmetic: physics settings are evaluated in file
    order, and a long chain reading a short chain's already-updated value in the
    same frame is what makes fine strands appear to lead the mass they hang
    from. Heavy first means light strands trail.
    """
    def weight(chain: tuple) -> float:
        role = chain[2] if len(chain) > 2 else ""
        return -PROFILES.get(role, PROFILES["accessory"]).delay

    return sorted(sway_params, key=weight)
