"""Software-render a model the way a runtime would, and save a PNG.

This is the check that catches everything structural validation cannot: whether
the rig actually *looks* like the character. It walks drawables in draw order,
samples the atlas through each triangle's UVs, and composites with the runtime's
opacity -- the same steps a GPU renderer performs.

Vertex positions come from Cubism Core via ctypes, so what gets drawn is the
runtime's own deformation output at whatever parameter values you ask for, not
a re-implementation of it.

Usage:
    python3 tools/render_model.py out/Aka --out render.png
    python3 tools/render_model.py out/Aka --param ParamAngleX=30 ParamEyeLOpen=0
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DYLIB = ROOT / "reference" / "core" / "libLive2DCubismCore.dylib"


class Vec2(C.Structure):
    _fields_ = [("X", C.c_float), ("Y", C.c_float)]


def load_core() -> C.CDLL:
    if not DYLIB.exists():
        raise SystemExit(
            f"missing {DYLIB}\n"
            "Copy libLive2DCubismCore.dylib from the Cubism SDK "
            "(Core/dll/macos/) into reference/core/.")
    lib = C.CDLL(str(DYLIB))
    lib.csmGetVersion.restype = C.c_uint
    lib.csmGetMocVersion.restype = C.c_uint
    lib.csmGetMocVersion.argtypes = [C.c_void_p, C.c_uint]
    lib.csmHasMocConsistency.restype = C.c_int
    lib.csmHasMocConsistency.argtypes = [C.c_void_p, C.c_uint]
    lib.csmReviveMocInPlace.restype = C.c_void_p
    lib.csmReviveMocInPlace.argtypes = [C.c_void_p, C.c_uint]
    lib.csmGetSizeofModel.restype = C.c_uint
    lib.csmGetSizeofModel.argtypes = [C.c_void_p]
    lib.csmInitializeModelInPlace.restype = C.c_void_p
    lib.csmInitializeModelInPlace.argtypes = [C.c_void_p, C.c_void_p, C.c_uint]
    lib.csmUpdateModel.argtypes = [C.c_void_p]
    for fn in ("csmGetDrawableCount", "csmGetParameterCount"):
        getattr(lib, fn).restype = C.c_int
        getattr(lib, fn).argtypes = [C.c_void_p]
    for fn in ("csmGetDrawableVertexCounts", "csmGetDrawableIndexCounts",
               "csmGetDrawableTextureIndices", "csmGetDrawableRenderOrders",
               "csmGetDrawableDrawOrders"):
        getattr(lib, fn).restype = C.POINTER(C.c_int)
        getattr(lib, fn).argtypes = [C.c_void_p]
    for fn in ("csmGetDrawableVertexPositions", "csmGetDrawableVertexUvs"):
        getattr(lib, fn).restype = C.POINTER(C.POINTER(Vec2))
        getattr(lib, fn).argtypes = [C.c_void_p]
    lib.csmGetDrawableIndices.restype = C.POINTER(C.POINTER(C.c_ushort))
    lib.csmGetDrawableIndices.argtypes = [C.c_void_p]
    lib.csmGetDrawableOpacities.restype = C.POINTER(C.c_float)
    lib.csmGetDrawableOpacities.argtypes = [C.c_void_p]
    for fn in ("csmGetParameterValues", "csmGetParameterDefaultValues"):
        getattr(lib, fn).restype = C.POINTER(C.c_float)
        getattr(lib, fn).argtypes = [C.c_void_p]
    lib.csmGetParameterIds.restype = C.POINTER(C.c_char_p)
    lib.csmGetParameterIds.argtypes = [C.c_void_p]
    lib.csmGetDrawableIds.restype = C.POINTER(C.c_char_p)
    lib.csmGetDrawableIds.argtypes = [C.c_void_p]
    lib.csmReadCanvasInfo.argtypes = [C.c_void_p, C.POINTER(Vec2),
                                     C.POINTER(Vec2), C.POINTER(C.c_float)]
    return lib


class Model:
    def __init__(self, lib: C.CDLL, moc_path: Path):
        self.lib = lib
        raw = moc_path.read_bytes()
        # csmReviveMocInPlace needs the moc at a 64-byte boundary and writes
        # back into that memory, so the buffer must stay alive and aligned for
        # the model's whole lifetime. numpy arrays give a stable base address;
        # ctypes string buffers are only guaranteed 8-byte aligned, which
        # crashed with SIGBUS on roughly half of all runs depending on where
        # the allocator happened to land.
        self._buf = np.zeros(len(raw) + 64, dtype=np.uint8)
        base = self._buf.ctypes.data
        self._moc_off = (-base) % 64
        self._buf[self._moc_off:self._moc_off + len(raw)] = np.frombuffer(raw, np.uint8)
        moc_ptr = base + self._moc_off

        if not lib.csmHasMocConsistency(moc_ptr, len(raw)):
            raise SystemExit("moc3 failed csmHasMocConsistency")
        self.moc_version = lib.csmGetMocVersion(moc_ptr, len(raw))
        self.moc = lib.csmReviveMocInPlace(moc_ptr, len(raw))
        if not self.moc:
            raise SystemExit("csmReviveMocInPlace failed")

        # The model block needs 256-byte alignment AND `size` usable bytes past
        # that aligned address. Allocating exactly size+256 and then advancing
        # to the aligned offset leaves fewer than `size` bytes at the tail, so
        # Core writes past the end -- SIGBUS, with no Python traceback. Reserve
        # a full extra alignment window.
        size = lib.csmGetSizeofModel(self.moc)
        self._mbuf = np.zeros(size + 256, dtype=np.uint8)
        maddr = self._mbuf.ctypes.data
        self.model = lib.csmInitializeModelInPlace(
            self.moc, maddr + ((-maddr) % 256), size)
        if not self.model:
            raise SystemExit("csmInitializeModelInPlace failed")
        self.update()

    def update(self) -> None:
        self.lib.csmUpdateModel(self.model)

    @property
    def param_ids(self) -> list[str]:
        ids = self.lib.csmGetParameterIds(self.model)
        n = self.lib.csmGetParameterCount(self.model)
        return [ids[i].decode() for i in range(n)]

    def set_param(self, name: str, value: float) -> bool:
        ids = self.param_ids
        if name not in ids:
            return False
        self.lib.csmGetParameterValues(self.model)[ids.index(name)] = value
        return True

    def canvas(self) -> tuple[float, float, float, float, float]:
        size, origin, ppu = Vec2(), Vec2(), C.c_float()
        self.lib.csmReadCanvasInfo(self.model, C.byref(size), C.byref(origin),
                                  C.byref(ppu))
        return size.X, size.Y, origin.X, origin.Y, ppu.value

    def drawables(self):
        lib, m = self.lib, self.model
        n = lib.csmGetDrawableCount(m)
        vc = lib.csmGetDrawableVertexCounts(m)
        ic = lib.csmGetDrawableIndexCounts(m)
        vp = lib.csmGetDrawableVertexPositions(m)
        uv = lib.csmGetDrawableVertexUvs(m)
        ix = lib.csmGetDrawableIndices(m)
        op = lib.csmGetDrawableOpacities(m)
        order = lib.csmGetDrawableRenderOrders(m)
        ids = lib.csmGetDrawableIds(m)
        for i in range(n):
            nv, ni = vc[i], ic[i]
            yield {
                "index": i,
                "id": ids[i].decode(),
                "order": order[i],
                "opacity": op[i],
                "verts": np.array([(vp[i][k].X, vp[i][k].Y) for k in range(nv)]),
                "uvs": np.array([(uv[i][k].X, uv[i][k].Y) for k in range(nv)]),
                "indices": np.array([ix[i][k] for k in range(ni)], dtype=int),
            }


def rasterize(model: Model, atlas: Image.Image, width: int = 800) -> Image.Image:
    """Composite every drawable, back to front, sampling the atlas per pixel."""
    cw, ch, ox, oy, ppu = model.canvas()
    aspect = ch / cw if cw else 1.0
    height = int(round(width * aspect))

    tex = np.asarray(atlas.convert("RGBA"), dtype=np.float32) / 255.0
    th, tw = tex.shape[:2]
    canvas = np.zeros((height, width, 4), dtype=np.float32)

    draws = sorted(model.drawables(), key=lambda d: d["order"])
    for d in draws:
        if d["opacity"] <= 0.0 or len(d["indices"]) == 0:
            continue
        # model space -> canvas pixels. Core reports positions in model units
        # with the origin at the canvas centre and Y up.
        v = d["verts"].copy()
        px = (v[:, 0] + ox) / cw * width
        py = (1.0 - (v[:, 1] + oy) / ch) * height
        pts = np.stack([px, py], axis=1)

        for t in d["indices"].reshape(-1, 3):
            _fill_triangle(canvas, pts[t], d["uvs"][t], tex, tw, th, d["opacity"])

    out = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _fill_triangle(canvas, tri, uvtri, tex, tw, th, opacity):
    """Barycentric scanline fill with nearest-neighbour texture sampling."""
    h, w = canvas.shape[:2]
    x0 = max(int(np.floor(tri[:, 0].min())), 0)
    x1 = min(int(np.ceil(tri[:, 0].max())) + 1, w)
    y0 = max(int(np.floor(tri[:, 1].min())), 0)
    y1 = min(int(np.ceil(tri[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return

    ax, ay = tri[0]
    bx, by = tri[1]
    cx, cy = tri[2]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-12:
        return

    ys, xs = np.mgrid[y0:y1, x0:x1]
    xs_f = xs + 0.5
    ys_f = ys + 0.5
    l1 = ((by - cy) * (xs_f - cx) + (cx - bx) * (ys_f - cy)) / det
    l2 = ((cy - ay) * (xs_f - cx) + (ax - cx) * (ys_f - cy)) / det
    l3 = 1.0 - l1 - l2
    inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
    if not inside.any():
        return

    u = l1 * uvtri[0, 0] + l2 * uvtri[1, 0] + l3 * uvtri[2, 0]
    # UV origin is bottom-left; image rows run top-down.
    vv = l1 * uvtri[0, 1] + l2 * uvtri[1, 1] + l3 * uvtri[2, 1]
    sx = np.clip((u * tw).astype(int), 0, tw - 1)
    sy = np.clip(((1.0 - vv) * th).astype(int), 0, th - 1)

    src = tex[sy, sx]
    a = src[..., 3] * opacity * inside
    dst = canvas[y0:y1, x0:x1]
    ia = 1.0 - a
    dst[..., :3] = src[..., :3] * a[..., None] + dst[..., :3] * ia[..., None]
    dst[..., 3] = a + dst[..., 3] * ia


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--param", nargs="*", default=[],
                    help="ParamName=value overrides")
    args = ap.parse_args()

    d = Path(args.model_dir)
    manifest = json.loads(next(d.glob("*.model3.json")).read_text())
    refs = manifest["FileReferences"]
    lib = load_core()
    print(f"Cubism Core {lib.csmGetVersion() >> 24}."
          f"{(lib.csmGetVersion() >> 16) & 0xFF}."
          f"{lib.csmGetVersion() & 0xFFFF}")

    model = Model(lib, d / refs["Moc"])
    print(f"moc version {model.moc_version}, "
          f"{lib.csmGetDrawableCount(model.model)} drawables, "
          f"{lib.csmGetParameterCount(model.model)} parameters")

    for spec in args.param:
        name, _, val = spec.partition("=")
        if model.set_param(name, float(val)):
            print(f"  {name} = {val}")
        else:
            print(f"  WARNING: no such parameter {name}")
    model.update()

    atlas = Image.open(d / refs["Textures"][0])
    img = rasterize(model, atlas, args.width)

    out = Path(args.out) if args.out else d / "render.png"
    img.save(out)
    filled = (np.asarray(img)[..., 3] > 0).mean()
    print(f"wrote {out} ({img.width}x{img.height}, {filled*100:.1f}% covered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
