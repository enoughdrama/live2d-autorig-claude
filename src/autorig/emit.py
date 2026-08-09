"""Stage 4 -- write the runtime model directory.

VTube Studio (and every Cubism SDK) loads a *directory*, not a bare moc3:

    Model/
      Model.moc3
      Model.model3.json          manifest: moc, textures, physics, groups
      Model.physics3.json        inertia
      Model.cdi3.json            display names for the parameter panel
      Model.4096/texture_00.png  texture atlas

The atlas here is the whole PSD canvas rendered once, because stage 3a assigns
UVs in canvas space. That trades atlas efficiency for exactness -- no repacking
means no chance of a UV/atlas mismatch, which is invisible in validation and
obvious the moment a model renders.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

# Power-of-two atlas sizes. 4096 is the practical ceiling: VTube Studio and a
# lot of GPUs baulk at 8192, and a 5500px-tall PSD would otherwise demand it.
ATLAS_SIZES = [1024, 2048, 4096]
MAX_ATLAS = ATLAS_SIZES[-1]


def _atlas_size(w: int, h: int) -> int:
    """Smallest power-of-two atlas that holds the canvas, capped at MAX_ATLAS.

    When the canvas exceeds the cap the atlas does not grow -- render_atlas
    downscales the artwork to fit instead. UVs are normalised, so they stay
    correct either way.
    """
    need = max(w, h)
    for s in ATLAS_SIZES:
        if s >= need:
            return s
    return MAX_ATLAS


def render_atlas(build_dir: Path, layers: dict, out_png: Path) -> int:
    """Composite every layer back onto a square power-of-two canvas.

    UVs from stage 3a are canvas-relative, so the atlas must place each layer at
    its original canvas position and the canvas must sit at the atlas origin
    with Y flipped -- UV (0,0) is bottom-left, PSD (0,0) is top-left.
    """
    cw, ch = layers["canvas"]
    size = _atlas_size(cw, ch)
    atlas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Canvas bigger than the atlas cap: shrink the artwork to fit. UVs are
    # normalised against the scaled canvas, so geometry is unaffected.
    scale = min(1.0, size / max(cw, ch))
    sw, sh = int(round(cw * scale)), int(round(ch * scale))

    # bottom-left anchored, so v = 1 - y/ch maps onto the canvas region
    y_base = size - sh

    for rec in layers["layers"]:
        png = build_dir / rec["file"]
        if not png.exists():
            continue
        img = Image.open(png).convert("RGBA")
        x0, y0, _, _ = rec["bbox"]
        if scale < 1.0:
            nw = max(1, int(round(img.width * scale)))
            nh = max(1, int(round(img.height * scale)))
            img = img.resize((nw, nh), Image.LANCZOS)
        atlas.alpha_composite(img, (int(round(x0 * scale)),
                                    int(round(y_base + y0 * scale))))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_png)
    return size


def _uv_rescale(cw: int, ch: int, size: int) -> tuple[float, float, float, float]:
    """Scale/offset to map canvas-space UVs into atlas-space UVs.

    Must mirror render_atlas exactly -- the canvas occupies the bottom-left
    (cw*scale x ch*scale) region of a size x size atlas. A mismatch here is
    invisible to every validator and shows up as textures sliding off the mesh.

    Returns (sx, sy, ox, oy) with u' = u*sx + ox, v' = v*sy + oy.
    """
    scale = min(1.0, size / max(cw, ch))
    return (cw * scale) / size, (ch * scale) / size, 0.0, 0.0


def rescale_uvs(builder, cw: int, ch: int, size: int) -> None:
    """Rewrite mesh UVs from canvas space into atlas space, in place."""
    sx, sy, ox, oy = _uv_rescale(cw, ch, size)
    for spec in builder.meshes:
        uv = spec.mesh.uvs
        uv[:, 0] = uv[:, 0] * sx + ox
        uv[:, 1] = uv[:, 1] * sy + oy


def model3(name: str, texture_files: list[str], has_physics: bool,
           has_display_info: bool, groups: list[dict]) -> dict:
    refs: dict = {
        "Moc": f"{name}.moc3",
        "Textures": texture_files,
    }
    if has_physics:
        refs["Physics"] = f"{name}.physics3.json"
    if has_display_info:
        refs["DisplayInfo"] = f"{name}.cdi3.json"
    return {
        "Version": 3,
        "FileReferences": refs,
        "Groups": groups,
        "HitAreas": [],
    }


def eye_blink_group(param_ids: list[str]) -> dict:
    ids = [p for p in param_ids if p in ("ParamEyeLOpen", "ParamEyeROpen")]
    return {"Target": "Parameter", "Name": "EyeBlink", "Ids": ids}


def lipsync_group(param_ids: list[str]) -> dict:
    ids = [p for p in param_ids if p == "ParamMouthOpenY"]
    return {"Target": "Parameter", "Name": "LipSync", "Ids": ids}


def cdi3(params, parts) -> dict:
    """Display names for the Editor/VTS parameter panel.

    Without this, VTube Studio shows raw IDs. Grouping matters for usability
    once a model has more than a handful of parameters.
    """
    groups = [
        {"Id": "GroupHead", "GroupName": "Head", "Ids": [
            p.pid for p in params if p.pid.startswith("ParamAngle")]},
        {"Id": "GroupFace", "GroupName": "Face", "Ids": [
            p.pid for p in params
            if p.pid.startswith(("ParamEye", "ParamBrow", "ParamMouth"))]},
        {"Id": "GroupBody", "GroupName": "Body", "Ids": [
            p.pid for p in params
            if p.pid.startswith("ParamBody") or p.pid == "ParamBreath"]},
    ]
    return {
        "Version": 3,
        "Parameters": [
            {"Id": p.pid, "GroupId": _group_of(p.pid), "Name": _pretty(p.pid)}
            for p in params
        ],
        "ParameterGroups": [
            {"Id": g["Id"], "GroupId": "", "Name": g["GroupName"]}
            for g in groups if g["Ids"]
        ],
        "Parts": [{"Id": p.name, "Name": p.name.replace("_", " ")} for p in parts],
    }


def _group_of(pid: str) -> str:
    if pid.startswith("ParamAngle"):
        return "GroupHead"
    if pid.startswith(("ParamEye", "ParamBrow", "ParamMouth")):
        return "GroupFace"
    return "GroupBody"


_PRETTY = {
    "ParamAngleX": "Angle X", "ParamAngleY": "Angle Y", "ParamAngleZ": "Angle Z",
    "ParamEyeLOpen": "Eye Open L", "ParamEyeROpen": "Eye Open R",
    "ParamEyeBallX": "Eyeball X", "ParamEyeBallY": "Eyeball Y",
    "ParamBrowLY": "Brow L", "ParamBrowRY": "Brow R",
    "ParamBrowLForm": "Brow L Form", "ParamBrowRForm": "Brow R Form",
    "ParamMouthOpenY": "Mouth Open", "ParamMouthForm": "Mouth Form",
    "ParamMouthX": "Mouth X",
    "ParamEyeSmile": "Eye Smile",
    "ParamBodyAngleX": "Body Angle X", "ParamBodyAngleZ": "Body Angle Z",
    "ParamBreath": "Breath",
}


def _pretty(pid: str) -> str:
    return _PRETTY.get(pid, pid.replace("Param", ""))


def emit_model(builder, moc, build_dir: str | Path, out_dir: str | Path,
               name: str, physics: dict | None = None) -> Path:
    """Write the full runtime directory and return its path."""
    build_dir = Path(build_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = json.loads((build_dir / "layers.json").read_text())
    cw, ch = layers["canvas"]

    size = _atlas_size(cw, ch)
    tex_dir = f"{name}.{size}"
    tex_rel = f"{tex_dir}/texture_00.png"
    render_atlas(build_dir, layers, out_dir / tex_rel)

    (out_dir / f"{name}.moc3").write_bytes(moc.to_bytes())

    param_ids = [p.pid for p in builder.params]
    groups = [g for g in (eye_blink_group(param_ids), lipsync_group(param_ids))
              if g["Ids"]]

    manifest = model3(name, [tex_rel], physics is not None, True, groups)
    (out_dir / f"{name}.model3.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / f"{name}.cdi3.json").write_text(
        json.dumps(cdi3(builder.params, builder.parts), indent=2))
    if physics is not None:
        (out_dir / f"{name}.physics3.json").write_text(json.dumps(physics, indent=2))

    return out_dir
