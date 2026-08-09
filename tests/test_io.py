"""Container-level tests: the writer must be provably correct before any rig
logic is built on top of it, so that a broken rig is never misdiagnosed as a
broken serializer.

Run: python3 tests/test_io.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from moc3.io import ALIGN, BODY_OFFSET, MAGIC, Moc3  # noqa: E402

SAMPLES = ROOT / "samples"
MODELS = ["Hiyori", "Mark", "Haru"]   # v4.0, v4.0, v3.0


def test_roundtrip():
    """Byte-identical round-trip. Catches every offset/alignment/count bug."""
    found = 0
    for name in MODELS:
        p = SAMPLES / f"{name}.moc3"
        if not p.exists():
            continue
        found += 1
        raw = p.read_bytes()
        out = Moc3.from_bytes(raw).to_bytes()
        assert len(out) == len(raw), f"{name}: size {len(out)} != {len(raw)}"
        if out != raw:
            i = next(i for i, (a, b) in enumerate(zip(raw, out)) if a != b)
            raise AssertionError(f"{name}: first byte differs at {hex(i)}")
        print(f"ok: {name} round-trips byte-for-byte ({len(raw)} bytes)")
    if not found:
        print("skip: no sample models in samples/ -- see README")


def test_alignment_rule():
    """Sections are 64-byte aligned, except a non-runtime section directly
    after a RUNTIME_PTR one, which is packed flush. Getting this backwards
    still produces a loadable-looking file with every offset shifted."""
    p = SAMPLES / "Mark.moc3"
    if not p.exists():
        print("skip: alignment rule (no Mark.moc3)")
        return
    m = Moc3.from_bytes(p.read_bytes())
    out = m.to_bytes()
    import struct
    sot = struct.unpack_from("<480I", out, 0x40)
    lay = m.layout()
    prev_rt = False
    for i, (name, et, _cf, _v) in enumerate(lay):
        if not (prev_rt and et != "RUNTIME_PTR"):
            assert sot[i] % ALIGN == 0, f"{name} not 64-aligned"
        prev_rt = et == "RUNTIME_PTR"
    print("ok: alignment rule holds across all sections")


def test_header():
    p = SAMPLES / "Mark.moc3"
    if not p.exists():
        print("skip: header (no Mark.moc3)")
        return
    raw = p.read_bytes()
    assert raw[:4] == MAGIC
    assert raw[5] == 0, "big-endian file"
    m = Moc3.from_bytes(raw)
    assert m.version == 3, f"expected moc 4.0 (version byte 3), got {m.version}"
    assert m.get_count("art_meshes") > 0
    assert len(m["art_mesh_src.id"]) == m.get_count("art_meshes")
    assert m.to_bytes()[:BODY_OFFSET].count(b"\0") > 0
    print(f"ok: header + counts ({m.get_count('art_meshes')} art meshes, "
          f"{m.get_count('parameters')} params)")


if __name__ == "__main__":
    test_header()
    test_alignment_rule()
    test_roundtrip()
    print("\nall checks passed")
