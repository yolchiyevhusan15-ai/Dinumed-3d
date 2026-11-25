# DINUMED — offline 3D avatar asset package

Rigged male and female medical avatars with separate clothing and accessory GLBs.
Everything shares one 55-bone skeleton, ships with a local copy of three.js r187,
and runs completely offline.

## Contents

| Path | What |
|---|---|
| `assets/characters/male.glb`, `female.glb` | Rigged full-body characters (body, eyes, hair + eyebrows, underwear) in **A-pose**, with embedded `Idle` animation |
| `assets/clothing/<gender>/*.glb` | Separate skinned clothing: `scrubs_top`, `medical_trousers`, `white_coat`, `medical_shoes`, `medical_cap`, `stethoscope_over_coat`, `stethoscope_over_scrubs`, `glasses`, `id_card_over_coat`, `id_card_over_scrubs`, `watch` |
| `assets/animations/idle.glb` | 8 s looping idle clip (rotation-only; drives both skeletons) |
| `assets/textures/*.png` | All textures (also embedded in the GLBs that use them) |
| `assets/manifest.json` | Asset manifest: every file, byte size, SHA-256, vertex/triangle counts, meshes, materials, skeleton joint order |
| `source/generator/` | Complete source that generates every asset (Python) |
| `vendor/three/` | Unmodified npm package `three@0.187.0` (core + `GLTFLoader`, `OrbitControls`, …) |
| `demo/` | Offline Three.js demo (`index.html`, `main.js`, reusable `avatar.js`) |
| `tools/serve.mjs` | Zero-dependency local web server |
| `tools/validate.mjs` | Khronos glTF-Validator + manifest/hash check |
| `tests/` | Three.js test suite (`threejs-test.mjs`) and its reports (`REPORT.md`, `VALIDATOR.md`, `report.json`) |
| `LICENSE.md` | Licenses (assets CC0-1.0, three.js MIT) |

## Run the demo (offline)

Browsers block ES modules on `file://`, so serve the folder locally. No internet is needed.

```bash
node tools/serve.mjs          # -> http://localhost:8080/demo/
# or: python3 -m http.server 8080   then open http://localhost:8080/demo/
```

The import map in `demo/index.html` points `three` and `three/addons/` at `vendor/three/`,
so the core and addons always come from the same version (0.187.0). No CDN URLs are used anywhere.

URL options: `?gender=female`, `?items=white_coat,scrubs_top,…`, `?cam=front|side|back|face|torso|hand|feet`,
`?anim=0` (no animation), `?t=2.5` (freeze the idle at 2.5 s).

## Using the assets in your own three.js app

```js
import { Avatar, loadManifest } from './demo/avatar.js';
const manifest = await loadManifest('./');
const avatar = new Avatar(scene, manifest, './');
await avatar.setGender('female');
await avatar.setItems(['scrubs_top', 'medical_trousers', 'white_coat', 'stethoscope', 'id_card']);
avatar.play();                       // idle loop; call avatar.update(dt) each frame
```

How clothing is attached (`attachClothing` in `demo/avatar.js`): every clothing GLB contains its own
copy of the character's skeleton with the same bone names, order and inverse bind matrices. On load,
each clothing `SkinnedMesh` is re-bound to the character's bones by name:
`mesh.bind(new THREE.Skeleton(characterBonesByName, mesh.skeleton.boneInverses), identity)`.

Rules:
- Clothing is fitted per gender: use `clothing/male/*` with `male.glb` and `clothing/female/*` with `female.glb`.
- `stethoscope` and `id_card` each have two fits: `*_over_coat` (rests on the white coat) and `*_over_scrubs`
  (rests on the scrubs top). Use `over_coat` whenever `white_coat` is worn (the `Avatar` class does this).
- Shoes have soles: raise the character by `shoeGroundOffset` from the manifest when `medical_shoes` is worn
  (the `Avatar` class does this).

## Technical specification

- glTF 2.0 binary, metres, +Y up, characters face +Z, character left = +X.
- Bind pose: A-pose (arms 45° below horizontal). All bone rest rotations are identity.
- Skeleton: 55 joints, Mixamo-style names without prefix (`Hips, Spine, Spine1, Spine2, Neck, Head,
  LeftShoulder, LeftArm, LeftForeArm, LeftHand, LeftHandThumb1-3 … LeftHandPinky1-3, LeftUpLeg, LeftLeg,
  LeftFoot, LeftToeBase …`). Identical names and joint order in every GLB; see `manifest.skeleton.jointOrder`.
- Skinning: ≤ 4 influences per vertex, normalised weights. Body weights come from the anatomical
  primitives that generated the surface, limited to neighbouring bones and smoothed. Clothing weights
  are transferred from the body. Rigid items (glasses, cap, watch, ID card) are bound to a single bone.
- Materials: glTF metallic-roughness PBR. Skin uses vertex colours (lips, cheeks); eyes, ID card and
  watch dial use PNG base-colour textures; metal parts are metallic = 1.

## Rebuilding the assets

```bash
pip install numpy scipy scikit-image fast-simplification pillow
python3 source/generator/build.py        # ~3-4 min on 4 cores; writes assets/
python3 source/generator/manifest.py     # writes assets/manifest.json
npm install                              # dev tool: gltf-validator
npm run validate                         # glTF-Validator + manifest hashes
node tests/threejs-test.mjs              # three.js tests (needs Playwright + Chromium)
```

The generator is deterministic: each body and garment is a signed distance field (smooth unions of
anatomical primitives). Clothing surfaces are offsets of the body, clipped exactly at hems, cuffs and
necklines. Surfaces are meshed with marching cubes, decimated, then projected back onto the exact
surface. Because garments are offset surfaces of the same body, they cannot intersect it in the
bind pose. The test suite checks this, and the idle animation, in three.js.

## Verification status

Latest results (reports are generated into `tests/`):

- `tests/VALIDATOR.md`: Khronos glTF-Validator 2.0.0-dev.3.10. **25/25 GLB files, 0 errors, 0 warnings.**
  Manifest sizes and SHA-256 hashes match the files on disk.
- `tests/REPORT.md`: three.js r187 in headless Chromium with every non-local request blocked.
  **111/111 checks pass**:
  - both characters and all 22 clothing GLBs load with GLTFLoader;
  - identical bone names and order; inverse bind matrices identical to the character's (max difference 0);
  - skin weights normalised, joint indices valid, no NaN;
  - idle clip binds to existing bones;
  - zero vertices penetrating more than 1 mm across 27 layer pairs (body, underwear, scrubs, trousers,
    coat, shoes, cap, hair, glasses, watch, stethoscope, ID card), in the bind pose and at 8 idle frames;
  - accessories within 2.5 mm of their support (not floating);
  - no network request leaves the machine; the demo renders both genders with the full outfit.

## Known limitations

- The look is stylised and procedural (built from smooth anatomical primitives), not photoscanned or
  hand-sculpted. Faces are simplified: no facial rig or blend shapes; eyes and jaw are not animated.
- Fingers are rigged (3 bones each) but the idle clip does not animate them.
- Clothing is skinned (linear blend skinning), not cloth-simulated. Clipping is verified for the bind
  pose and the supplied idle animation. Other animations with extreme poses (deep crouches, arms raised
  overhead) can still cause clothing to intersect, as with any skinned garment.
- Clothing is fitted per gender and per body. It is not intended for differently proportioned bodies.
- The stethoscope and ID card come in two fits (over the coat / over the scrubs) because a rigid
  accessory cannot rest on both layers.
