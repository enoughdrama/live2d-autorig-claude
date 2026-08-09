---
name: live2d-autorig
description: Use when rigging a Live2D Cubism model in code from a layered PSD — writing .moc3 directly (art meshes, deformers, parameters, keyforms), or generating physics3.json / motion3.json / model3.json on top of a rig. Covers the verified moc3 binary format, the keyform data model, and validation against official Cubism Core. Triggers: "авториг", "autorig", "live2d rig", "generate moc3", "physics3", "VTube Studio model", "рига live2d".
---

# Live2D Autorig

Build a rigged Cubism model from a layered PSD entirely in code. No Cubism
Editor. Output loads in VTube Studio.

## Critical context: what is actually possible

The widely repeated claim "Live2D cannot be autorigged" is **wrong**, and this
project disproves it. What is true:

- `.cmo3` (Editor project) is a proprietary encrypted container — magic `CAFF`,
  entropy 7.6–7.95 bits/byte, no zip/JSON/zlib inside. Writing it is not viable.
- Cubism Editor has no scripting API. The External Application Integration API
  (WebSocket, port 22033, 15 methods) only reads/writes *values of parameters
  that already exist*. It cannot create geometry or trigger an export.

But `.moc3` — the runtime format VTube Studio and every SDK actually load — is
reverse-engineered well enough to **write from scratch**. Verified: a moc3 built
by this repo loads in official Cubism Core 5.1.0 with `consistency=1` and
deforms under parameter control. That is the whole basis of this project.

Physics, motion, expressions and the manifest are all open JSON formats
(`CubismSpecs`) and were never the hard part.

## The oracle — use it constantly

`csmHasMocConsistency` from the official Core walks every section table and
rejects anything the runtime would choke on, naming the offending field.

```bash
./tools/build_oracle.sh                  # -> build/validate
./build/validate model.moc3              # structural check + load
./build/validate model.moc3 0 30         # also drive param 0 to 30.0
```

A file that writes without a Python exception means nothing. A file that passes
the oracle is structurally valid. Whether it *looks* right is a third question
only a human answers in VTube Studio.

## Container format (verified byte-for-byte)

Established by bisecting a naive rewrite against Hiyori/Mark/Haru until all
three round-tripped byte-identically. Every constant here was wrong in some
earlier iteration:

```
0x000  "MOC3"  u8 version  u8 endian(0=LE)   padding to 0x40
0x040  section offset table: 480 × u32
0x7C0  body — sections in layout order, each at its SOT offset
```

- **Version byte**: 1=3.0, 2=3.3, **3=4.0**, 4=4.2, 5=5.0, 6=5.3. Emit 3.
- **`count_info` is 128 bytes on disk** (32 i32), even though Purism's struct
  lists 39 fields — the later ones only exist in newer mocs.
- **Alignment**: every section starts 64-byte aligned, **except** a non-runtime
  section immediately following a `RUNTIME_PTR` section, which packs flush.
  Get this backwards and every offset past the first runtime table shifts.
- `RUNTIME_PTR` sections are 8 bytes/element of zeroes on disk — pointer slots
  the runtime fills in. They still occupy a SOT slot.
- All integers little-endian.

### The section table is generated, not hand-written

`tools/gen_layout.py` parses the `PSM__SECTIONS_V**` macros out of PurismCore's
`moc3.h` and emits `src/moc3/_layout.py` — 167 sections across v3.0…v5.3, in
exact on-disk order.

**Do not hand-edit `_layout.py`.** Upstream py-moc3 hand-maintains this table
and gets it wrong in ways that pass a smoke test and fail on real models: a
read-to-EOF heuristic for `quad_transforms` that drops a 256-byte tail, and
`art_mesh` slot 41/44 labelled backwards (41 is *vertex* count, 44 is *index*
count — verified on Hiyori: slot41 sums to 2822 = len(uvs)/2, slot44 to 10278 =
total indices).

Regenerate after updating Purism:
```bash
python3 tools/gen_layout.py reference/purism/src/moc3.h > src/moc3/_layout.py
```

## The rig data model

This is the part that matters for generating a rig. Verified against Mark.

### Parameters → key tables → keys

```
param_src.key_table_off/len   ->  key_table_src[]
key_table_src.keys_off/len    ->  keys_src.key[]   (float breakpoints)
```

A parameter owns one key table; the table lists the parameter values at which
keyforms are authored. `ParamAngleX [-30..30]` has keys `[-30, 0, 30]` — three
breakpoints, so anything bound to it needs three keyforms.

### Bindings — the N-dimensional grid

```
binding_src.key_table_idx_off/len  ->  key_table_idx_src.idx[]  ->  key_table indices
```

A binding lists *which parameters* a deformable is driven by. The number of
keyforms it must supply is the **product of those tables' key counts**:

| binding | tables | key counts | keyforms required |
|---------|--------|-----------|-------------------|
| 0 (empty) | — | — | 1 (rest pose only) |
| 1 | [8] | [3] | 3 |
| 4 | [5, 6] | [3, 3] | **9** |

Verified: `EyeL4` binds tables [5,6] and has `key_len=9`. `Hair2` binds nothing
and has `key_len=1`. Getting this product wrong is the single most common way to
produce a file the oracle rejects.

**Grid order is column-major: the FIRST bound table varies FASTEST.** This is
the opposite of `itertools.product`'s natural order and of what "row-major"
suggests. Getting it backwards is completely silent -- the model loads,
validates, and deforms, it just moves wrong, because every multi-parameter mesh
reads a transposed grid. Locked by `test_keyform_grid_order`, which encodes cell
indices into vertex positions and reads them back through the runtime.

### Deformables

Three kinds, each with `binding_idx`, `keyform_off`, `key_len`:

Draw order lives in `draw_group_obj_src`, and `self_group_idx` must be **-1**
("not a nested group"). Writing 0 points every item at the group that contains
it; the runtime then reports render order 0 for every drawable and layers
composite in arbitrary order. Locked by `test_render_order_distinct`.

- **art_mesh** — a drawable. `vertex_count`, `uv_off`, `idx_off/len`,
  `texture_no`, `parent_part_idx`, `parent_deformer_idx`.
  Keyforms live in `art_mesh_key_src` (opacity, draw_order, `key_pos_off`).
- **warp** (`warp_src`) — a grid deformer. `row`/`col` control points,
  `vertex_count = row*col`.
- **rotation** (`rotation_src`) — a pivot deformer with `base_angle`.

`deformer_src` is the shared parent list (`type` selects warp vs rotation,
`local_idx` indexes into the type-specific array).

### Keyform positions — where deformation actually lives

`keyform_pos_src.value` is one flat float array of interleaved x,y for **every
keyform of every deformable**. A mesh's keyform *k* starts at
`art_mesh_key_src.key_pos_off[keyform_off + k]`.

There are no "offsets from a base pose" — each keyform stores absolute
positions. The rest pose is just the keyform at the parameter's default value,
and the runtime interpolates between them. Mark: 18880 floats total.

## Pipeline

```
model.psd
  ├─[1] parse    → layers.json + layers/*.png     src/autorig/psd.py
  ├─[2] classify → rig_plan.json                  src/autorig/classify.py
  ├─[3] build    → mesh, hierarchy, keyforms      src/autorig/build.py
  ├─[4] pack     → model.moc3 + model3.json       src/autorig/emit.py
  └─[5] physics  → physics3.json                  src/autorig/physics.py
```

Stages 1–2 are ported from the sibling inochi2d-autorig project and are
output-format agnostic. Both official sample PSDs are **completely flat — zero
groups**; names and geometry carry the structure, so do not build hierarchy
logic that depends on PSD groups.

**Keep the LLM confined to stage 2.** It classifies layers and lays out the
hierarchy — a small declarative plan. It must never emit vertex coordinates or
keyform grids; stage 3 computes those deterministically. Models are unreliable
at generating thousands of numbers.

## Ambiguous rig values — where "it loads" hides "it moves wrong"

Every bug in this section passed `csmHasMocConsistency`, deformed geometry, and
reported all parameters as driving something. Structural validators cannot see
any of them; only measuring the runtime's own output at specific poses can.

**The role vocabulary must be total.** `classify.py` emits roles by regex;
`rig.py` sorts them into `HEAD_ROLES`/`BODY_ROLES`. A role in neither falls
through to the body group *silently* — `blush`, `breast`, `collar`, `hair`, and
`hair_ahoge` all did, so a blush rigged to the torso and slid off the cheek on
every head turn. Locked by `test_role_vocabulary_total`. The same class of bug
had `classify.PHYSICS_ROLES` and `autorig.PHYSICS_ROLES` as two literals that
drifted apart in both directions; the rig now derives its set from the
classifier's.

**Dead config reads as working config.** `ROLE_FOLLOW` declared `neck: 0.35`,
but `neck` is in `BODY_ROLES`, so `head_transform` was never called on it — the
constant had no effect and the neck stayed perfectly rigid (displacement
`0.0000`) through a 30° head turn. Roles on the head/body boundary need
`NECK_ROLES`: both transforms, with the head's contribution ramped by height so
the jaw follows and the shoulders stay put.

**A closed eye is a line, not a hole.** Scaling every eye part to exactly zero
height makes the eye *vanish* rather than close, because the lash and lid
geometry disappears along with the sclera. `BLINK_PROFILE` gives each part a
residual height (lids/lashes keep 6–10%, sclera and iris go to 0) and a separate
anchor. The remaining gap is the honest ceiling: a fully closed lid is *artwork*,
and no analytic transform can invent it from open-eye art.

**Hanging parts rotate; they do not shear.** `sway()` was a depth-weighted
horizontal shear, so at full deflection the tail tip translated 0.43 model units
while the strand's length stayed constant — a motion no pendulum makes, and the
real reason the tail read as "offset to one side". Rotating about the centre of
the part's own top edge, with the angle ramped by depth**^1.5**, preserves
length and pins the anchor.

**Keyform count is a product, so binding is a budget.** Giving the neck all
three head angles plus both body angles plus breath costs 3·3·3·3·3·2 = 486
keyforms ≈ 5.4 MB *per mesh*. Dropping pitch — which the throat barely registers
— brings it to 162. Always price a binding before adding it.

**Physics needs sides.** A right-side strand hangs on the opposite side of the
pivot from its left-side twin, so the same head turn must push it the other way;
without `Reflect` on the mirrored inputs both sides swing in lockstep and the
hair reads as a rigid helmet.

## Validation order

1. `python3 tests/test_io.py` — byte-identical round-trip of real models. Run
   this first, always: it isolates container bugs from rig bugs.
2. `./build/validate out.moc3` — official Core accepts and initializes it.
3. `./build/validate out.moc3 <i> <v>` — geometry actually moves when driven.
   A rig that loads but never deforms is the most common silent failure.
4. `python3 tests/test_pipeline.py` — measures the runtime's deformation output
   at specific poses. This is the only layer that catches *wrong* motion as
   opposed to *absent* motion; every bug in the section above got through 1–3.
5. `python3 tools/render_model.py out/Aka --param ParamEyeLOpen=0` — look at it.
6. Load in VTube Studio. Only a human can judge whether motion looks right.

When adding a deformation, add the pose measurement with it. The pattern is
always the same: snapshot vertices at rest, set the parameter, snapshot again,
and assert on the *difference* — magnitude, which vertices moved, and whether
the anchor stayed put. Asserting only that something changed is what let the
transposed keyform grid and the shear-instead-of-swing survive.

## Scope honesty

Autorigging produces a *starting* rig, not a finished commercial one. A flat PSD
cannot reveal occluded geometry — what is behind the hair, how an arm connects
at the shoulder. Say this rather than overselling the output.
