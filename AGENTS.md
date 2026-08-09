# Instructions for agents working on this repo

Read this before running anything. It records facts that were verified the hard
way, so they do not get rediscovered or re-litigated.

## What this is

A code-only Live2D autorigger: layered PSD in, `.moc3` model directory out,
loads in VTube Studio. No Cubism Editor anywhere in the pipeline.

The format spec lives in `skills/live2d-autorig/SKILL.md`. **Read it before
touching any rigging or container code.**

## Non-negotiable facts (verified, do not re-litigate)

1. **Live2D CAN be autorigged.** Writing `.moc3` from scratch works and is
   proven here against official Cubism Core 5.1.0. Widely-repeated claims to the
   contrary (including in the sibling inochi2d-autorig repo) conflate `.moc3`
   with `.cmo3`.
2. **`.cmo3` is a dead end.** Proprietary encrypted container, magic `CAFF`,
   7.6-7.95 bits/byte. Not zip, not JSON, no zlib. Do not attempt it.
3. **The Editor cannot be automated.** External Application Integration API
   (port 22033, 15 methods) only reads/writes values of existing parameters.
   No export trigger, no headless mode, no CLI.
4. **`src/moc3/_layout.py` is generated. Never hand-edit it.** Regenerate with
   `python3 tools/gen_layout.py reference/purism/src/moc3.h > src/moc3/_layout.py`.
   Hand-maintaining this table is exactly how upstream py-moc3 ended up with
   swapped art_mesh labels and a lost 256-byte tail.
5. **Section counts and offsets are in FLOATS, not vertex pairs.** Keyform
   position blocks are additionally padded to a multiple of 16 floats.
6. **All integers little-endian.** Sections are 64-byte aligned except a
   non-runtime section directly after a RUNTIME_PTR one.

## How to work here

**Run the tests before and after every change.**

```bash
python3 tests/test_io.py         # container: byte-identical round-trip
python3 tests/test_pipeline.py   # UV/atlas, manifest, physics consistency
```

`test_io.py` first, always. It isolates container bugs from rig bugs.

**When a generated moc3 is rejected, use `build/diagnose` (Purism), not
`build/validate` (official Core).** Purism names the exact field:

```
[PSM] E: art_mesh[75]: UV [3247, +13*2) oob (max 3260)
```

The official Core only returns 0/1.

**A file that writes without a Python exception is not a rig.** Run
`./build/exercise` — it drives every parameter and reports which drawables
respond. A parameter that drives nothing is a bug, and consistency checks
cannot see it.

**Never trust a section name you did not check.** `Moc3.__setitem__` rejects
unknown names because a typo is otherwise completely silent: the section is
never written, stays zero-filled, and surfaces much later as an out-of-bounds
read. `uv_src.uv` vs the real `uv_src.xy` cost real debugging time.

**Keep the LLM confined to stage 2.** It classifies layers and lays out
hierarchy — a small declarative plan. It must never emit vertex coordinates or
keyform grids; stage 3 computes those deterministically.

## Setup

```bash
python3 -m pip install --user psd-tools numpy pillow triangle scikit-image
```

`reference/purism/` and `reference/core/` are gitignored and must be fetched:
PurismCore from https://github.com/SakuraMotion/PurismCore (build with `make`),
and `Live2DCubismCore.h` + `libLive2DCubismCore.a` from the Cubism SDK for
Native. Then:

```bash
./tools/build_oracle.sh
cc -O2 -o build/exercise tools/exercise.c -Ireference/core \
   reference/core/libLive2DCubismCore.a -lm
cc -O1 -o build/diagnose tools/diagnose.c -Ireference/purism/include \
   -Ireference/purism/src reference/purism/build/libPurismCore.a
```

Sample models (`samples/*.moc3`, `samples/*.psd`) are gitignored too — the
round-trip test skips without them.

## Scope honesty

Autorigging produces a starting rig, not a finished commercial one. Deformations
are analytic approximations. Say this rather than overselling the output.

## Git

Do not commit: `reference/`, `samples/*.moc3`, `samples/*.psd`, `build/`, `out/`.
