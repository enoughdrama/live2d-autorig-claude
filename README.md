# live2d-autorig-claude

Rig a Live2D Cubism character from a layered PSD entirely in code — no Cubism
Editor. Output is a `.moc3` model directory that loads in VTube Studio.

```bash
python3 -m src.autorig character.psd -o out/MyModel
```

```
out/MyModel/
  MyModel.moc3              76 art meshes, 51 parameters
  MyModel.model3.json       manifest with EyeBlink + LipSync groups
  MyModel.physics3.json     37 inertia chains
  MyModel.cdi3.json         display names for the parameter panel
  MyModel.4096/texture_00.png
```

## "Live2D can't be autorigged" is wrong

- **`.cmo3`** (the Editor project file) is a proprietary encrypted container —
  magic `CAFF`, 7.6–7.95 bits/byte entropy, no zip/JSON/zlib inside. Not
  writable, and not a useful target anyway.
- **Cubism Editor has no scripting API.** The External Application Integration
  API (WebSocket, port 22033, 15 methods) only reads and writes *values of
  parameters that already exist*. It cannot create geometry or trigger exports.

But **`.moc3` — the runtime format VTube Studio and every SDK actually load —
is writable.** We can construct from scratch,
and this repo does exactly that:

```
$ ./build/validate out/Aka/Aka.moc3
out/Aka/Aka.moc3   size=1534592 mocVersion=3 consistency=1
Live2D Cubism SDK Core Version 5.1.0
 -> LOADED drawables=76 params=51 parts=79

$ ./build/exercise out/Aka/Aka.moc3
51/51 parameters drive at least one drawable
```

That is the official Cubism Core 5.1.0 loading a model built from a PSD with no
Editor anywhere in the process.

## Pipeline

```
character.psd
  ├─[1] parse    → layers.json + layers/*.png    psd.py
  ├─[2] classify → rig_plan.json                 classify.py   (role, side, parent)
  ├─[3] mesh+rig → meshes, params, keyforms      mesh.py rig.py autorig.py
  ├─[4] pack     → moc3 + model3 + atlas         emit.py
  └─[5] physics  → physics3.json                 physics.py
```

Stages 1–2 are shared with the sibling `inochi2d-autorig` project and are
output-format agnostic.

**What gets rigged:** 2.5D head yaw/pitch/roll, lower-lid-anchored blink, iris
look, brow raise, jaw-hinged mouth open + form, body lean weighted by height
above the hips, breath, and a physics sway chain per hair strand / accessory /
skirt piece.

## Setup

```bash
python3 -m pip install --user psd-tools numpy pillow triangle scikit-image
```

Two native dependencies, both fetched rather than vendored (see `.gitignore`):

- **PurismCore** (`reference/purism/`) — open-source Live2D-compatible runtime.
  Its `src/moc3.h` is the authoritative section table and the source
  `tools/gen_layout.py` generates `src/moc3/_layout.py` from. Its validator
  names the exact field that failed, which the official Core does not.
- **Cubism SDK for Native** (`reference/core/`) — `Live2DCubismCore.h` plus
  `libLive2DCubismCore.a`, for the authoritative `csmHasMocConsistency` oracle.

```bash
./tools/build_oracle.sh      # -> build/validate
cc -O2 -o build/exercise tools/exercise.c -Ireference/core \
   reference/core/libLive2DCubismCore.a -lm
```

## Verification

Four levels, in the order they should be run:

```bash
python3 tests/test_io.py         # byte-identical round-trip of real models
python3 tests/test_pipeline.py   # UV/atlas agreement, manifest, physics counts
./build/validate model.moc3      # official Core accepts + initializes
./build/exercise model.moc3      # every parameter actually moves geometry
```

`test_io.py` runs first on purpose: it isolates container bugs from rig bugs.
Debugging a broken rig on top of a broken writer wastes hours.

`exercise` exists because **a rig that loads but does not move** is the failure
mode `consistency=1` hides. It caught right-side parts resolving to left-side
parameters, and a declared parameter that was never bound.

Rendering caught two more that even `exercise` cannot see, because both produce
motion -- just wrong motion:

- the keyform grid is **column-major** (first bound axis varies fastest), so a
  row-major write transposes every multi-parameter mesh
- `self_group_idx` must be `-1`; writing `0` collapses every drawable to render
  order 0 and layers composite arbitrarily

```bash
python3 tools/render_model.py out/Aka --out render.png
python3 tools/render_model.py out/Aka --param ParamAngleX=30 ParamEyeLOpen=0
```

That tool drives Cubism Core through ctypes and rasterises the runtime's own
deformation output, so what you see is what a real renderer would draw.

The last check is a human loading it in VTube Studio. Nothing automated can
tell you the motion looks *right*.

## Format notes

The container facts that cost the most time, all verified by bisecting against
Hiyori/Mark/Haru until they round-tripped byte-for-byte:

- SOT is **480 u32 slots**; body starts at **0x7C0**
- `count_info` is **128 bytes** on disk (32 i32), though Purism's struct lists 39
- sections are 64-byte aligned **except** a non-runtime section directly after a
  `RUNTIME_PTR` one, which packs flush
- section **counts** are in floats; **offsets** are in floats too, but keyform
  position blocks are padded up to a multiple of **16 floats** (SIMD)
- upstream py-moc3 has `art_mesh` slots 41/44 labelled backwards (41 is vertex
  count, 44 is index count) and loses a 256-byte tail on v3.03+ files

Full spec in [`skills/live2d-autorig/SKILL.md`](skills/live2d-autorig/SKILL.md).

## Scope honesty

Autorigging produces a **starting** rig, not a finished commercial one. The
deformations are analytic approximations of what a rigger would draw by hand.
A flat PSD cannot reveal occluded geometry — what is behind the hair, how an arm
connects at the shoulder — so expect to refine the result.

Physics tuning constants (`Mobility`, `Delay`, `Acceleration`, `Radius`) are
exposed in `physics.py` rather than baked in: pendulum feel is empirical and
depends on the art's proportions.

## License

MIT for this code. Sample models and the Cubism Core are not redistributed —
they are fetched at setup time under their own licenses.
