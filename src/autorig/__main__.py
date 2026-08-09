"""One-shot pipeline: layered PSD -> a VTube Studio ready model directory.

    python3 -m src.autorig model.psd -o out/MyModel
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .autorig import build_rig
from .emit import _atlas_size, emit_model, rescale_uvs
from .physics import build_physics, physics_chains


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autorig", description=__doc__)
    ap.add_argument("psd", help="layered PSD")
    ap.add_argument("-o", "--out", required=True, help="output model directory")
    ap.add_argument("--name", help="model name (default: output dir name)")
    ap.add_argument("--work", help="intermediate dir (default: <out>/.work)")
    ap.add_argument("--ppu", type=float, default=1000.0,
                    help="pixels per unit (default 1000)")
    ap.add_argument("--spacing", type=float, default=90.0,
                    help="mesh density in px (default 90; lower = denser)")
    ap.add_argument("--no-physics", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    name = args.name or out.name
    work = Path(args.work) if args.work else out / ".work"
    work.mkdir(parents=True, exist_ok=True)

    # stage 1-2 run as modules so they stay independently rerunnable
    print(f"[1/5] parsing {args.psd}")
    subprocess.run([sys.executable, "-m", "src.autorig.psd", args.psd,
                    "-o", str(work)], check=True)
    print("[2/5] classifying layers")
    subprocess.run([sys.executable, "-m", "src.autorig.classify",
                    str(work / "layers.json")], check=True)

    print("[3/5] meshing + rigging")
    rb = build_rig(work, pixels_per_unit=args.ppu, spacing=args.spacing)

    layers = json.loads((work / "layers.json").read_text())
    cw, ch = layers["canvas"]
    rescale_uvs(rb, cw, ch, _atlas_size(cw, ch))
    moc = rb.build()

    print("[4/5] physics")
    physics = None
    if not args.no_physics and getattr(rb, "sway_params", None):
        physics = build_physics(physics_chains(rb.sway_params))

    print("[5/5] writing model")
    emit_model(rb, moc, work, out, name, physics)

    n_sway = len(getattr(rb, "sway_params", []))
    print(f"\n{out}/")
    print(f"  {len(rb.meshes)} art meshes, {len(rb.params)} parameters, "
          f"{n_sway} physics chains")
    print(f"  {moc.get_count('art_mesh_keyforms')} keyforms")
    print(f"\nValidate:  ./build/validate {out}/{name}.moc3")
    print(f"Exercise:  ./build/exercise {out}/{name}.moc3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
