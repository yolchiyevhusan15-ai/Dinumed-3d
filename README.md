# DINUMED — offline 3D avatar va garderob

Multfilm uslubidagi erkak va ayol shifokor avatarlari, alohida kiyim va aksessuar GLB fayllari,
"Garderob" do'koni (DC tangalari bilan sotib olish) va three.js r187 ning mahalliy nusxasi.
Hammasi internetsiz ishlaydi.

*Stylised male and female doctor avatars with separate clothing/accessory GLBs, a wardrobe shop
demo, and a vendored copy of three.js r187. Everything runs offline.*

## Ishga tushirish (Run the demo)

Brauzerlar `file://` orqali ES modullarni bloklaydi, shuning uchun papkani mahalliy server orqali oching.
Internet kerak emas.

```bash
node tools/serve.mjs          # -> http://localhost:8080/demo/
# yoki: python3 -m http.server 8080   keyin http://localhost:8080/demo/
```

Garderob sahifasida:
- **Garderob** — kiyimlarni sotib olish (DC), kiyish va yechish. Har bir buyumning rasmi GLB fayldan shu zahoti chiziladi.
- **Ko'rinish** — erkak/ayol, teri va soch rangi, harakat (idle) va A-poza.
- **DC va daraja** — balans, daraja va sinov uchun DC qo'shish. Holat faqat shu brauzerda (`localStorage`) saqlanadi.

URL parametrlari (testlar uchun): `?gender=female`, `?wear=scrub_teal,coat_basic,…` (katalog id lari),
`?cam=three|front|back|side|face|torso|feet`, `?anim=0`, `?t=2.5`, `?fresh=1` (saqlangan holatni e'tiborsiz qoldirish).

## Contents

| Path | What |
|---|---|
| `assets/characters/male.glb`, `female.glb` | Rigged full-body characters (body, eyes, hair + brows, mouth, teeth, underwear) in **A-pose**, with embedded `Idle` animation. Each under 3 MB |
| `assets/clothing/<gender>/*.glb` | Separate skinned items: `scrubs_top`, `medical_trousers`, `white_coat`, `medical_shoes` (clogs), `sneakers`, `medical_cap`, `medical_mask`, `glasses`, `watch`, and three items with two fits each: `stethoscope_*`, `id_card_*`, `backpack_*` (`over_coat` / `over_scrubs`) |
| `assets/animations/idle.glb` | 8 s looping idle clip (rotation-only; drives both skeletons) |
| `assets/textures/*.png` | All textures (also embedded in the GLBs that use them) |
| `assets/manifest.json` | Asset manifest: every file, byte size, SHA-256, vertex/triangle counts, meshes, materials, skeleton joint order, ground offsets |
| `assets/catalog.json` | Shop catalogue: items, categories, prices, colour variants, levels |
| `source/generator/` | Complete source that generates every asset (Python) |
| `vendor/three/` | Unmodified npm package `three@0.187.0` (core + `GLTFLoader`, `OrbitControls`, …) |
| `demo/` | Offline Garderob demo (`index.html`, `main.js`, reusable `avatar.js`) |
| `tools/serve.mjs` | Zero-dependency local web server |
| `tools/validate.mjs` | Khronos glTF-Validator + manifest/hash/catalogue check |
| `tests/` | Three.js test suite (`threejs-test.mjs`) and its reports (`REPORT.md`, `VALIDATOR.md`, `report.json`) |
| `LICENSE.md` | Licenses (assets CC0-1.0, three.js MIT) |

The import map in `demo/index.html` points `three` and `three/addons/` at `vendor/three/`, so the core and
addons always come from the same version (0.187.0). No CDN URLs are used anywhere.

## Using the assets in your own three.js app

```js
import { Avatar, loadManifest } from './demo/avatar.js';
const manifest = await loadManifest('./');
const avatar = new Avatar(scene, manifest, './');
await avatar.setGender('female');
await avatar.setOutfit([
  { key: 'top', asset: 'scrubs_top', colors: { ScrubsTop: '#1c9aa6' } },   // recolour by mesh name
  { key: 'coat', asset: 'white_coat', colors: {} },
  { key: 'steth', asset: 'stethoscope', colors: {} },
]);
avatar.play();                       // idle loop; call avatar.update(dt) each frame
```

How clothing is attached (`attachClothing` in `demo/avatar.js`): every clothing GLB contains its own
copy of the character's skeleton with the same bone names, order and inverse bind matrices. On load,
each clothing `SkinnedMesh` is re-bound to the character's bones by name.

Rules:
- Clothing is fitted per gender: use `clothing/male/*` with `male.glb` and `clothing/female/*` with `female.glb`.
- `stethoscope`, `id_card` and `backpack` each have two fits: `*_over_coat` (rests on the white coat) and
  `*_over_scrubs` (rests on the scrubs top). Use `over_coat` whenever `white_coat` is worn (`Avatar` does this).
- Footwear has soles: raise the character by `groundOffset[<shoe id>]` from the manifest (`Avatar` does this).
- Colour variants (scrub colours, purple stethoscope, …) are material colour overrides listed in
  `assets/catalog.json`; they reuse the same GLB.

## Technical specification

- glTF 2.0 binary, metres, +Y up, characters face +Z, character left = +X.
- Bind pose: A-pose (arms 55° below horizontal). All bone rest rotations are identity.
- Skeleton: 55 joints, Mixamo-style names without prefix (`Hips, Spine, Spine1, Spine2, Neck, Head,
  LeftShoulder, LeftArm, LeftForeArm, LeftHand, LeftHandThumb1-3 … LeftHandPinky1-3, LeftUpLeg, LeftLeg,
  LeftFoot, LeftToeBase …`). Identical names and joint order in every GLB; see `manifest.skeleton.jointOrder`.
- Skinning: ≤ 4 influences per vertex, normalised weights. Body weights come from the anatomical
  primitives that generated the surface, limited to neighbouring bones and smoothed. Clothing weights
  are transferred from the body. Rigid items (glasses, cap, mask, watch, ID card, backpack body) are bound to a single bone.
- Materials: glTF metallic-roughness PBR. Skin uses vertex colours; eyes, ID card and watch dial use PNG
  base-colour textures; metal parts are metallic = 1.

## Rebuilding the assets

```bash
pip install numpy scipy scikit-image fast-simplification pillow
python3 source/generator/build.py        # ~10 min on 4 cores; writes assets/
python3 source/generator/build.py --items glasses,sneakers   # rebuild single items (reuses cached body)
python3 source/generator/manifest.py     # writes assets/manifest.json
npm install                              # dev tool: gltf-validator
npm run validate                         # glTF-Validator + manifest hashes + catalogue
node tests/threejs-test.mjs              # three.js tests (needs Playwright + Chromium)
```

The generator is deterministic: each body and garment is a signed distance field (smooth unions of
anatomical primitives). Clothing surfaces are offsets of the body, clipped exactly at hems, cuffs and
necklines. Surfaces are meshed with marching cubes, decimated (with extra density on the head), then
projected back onto the exact surface. Because garments are offset surfaces of the same body, they
cannot intersect it in the bind pose. The test suite checks this, and the idle animation, in three.js.

## Verification status

Latest results (reports are generated into `tests/`):

- `tests/VALIDATOR.md`: Khronos glTF-Validator 2.0.0-dev.3.10. **33/33 GLB files, 0 errors, 0 warnings.**
  Manifest sizes and SHA-256 hashes match the files on disk; the shop catalogue only references existing assets and meshes.
- `tests/REPORT.md`: three.js r187 in headless Chromium with every non-local request blocked.
  **147/147 checks pass**:
  - both characters and all 30 clothing GLBs load with GLTFLoader;
  - identical bone names and order; inverse bind matrices identical to the character's (max difference 0);
  - skin weights normalised, joint indices valid, no NaN;
  - idle clip binds to existing bones;
  - zero vertices penetrating more than 1 mm across 50 layer pairs, in the bind pose and at 8 idle frames;
  - accessories within 2.5 mm of their support (not floating);
  - no network request leaves the machine; the Garderob page renders both genders, a purchase
    deducts DC and wears the item, and every shop card gets a rendered thumbnail.

## Known limitations

- The look is stylised and procedural (built from smooth anatomical primitives in code), not hand-sculpted.
  It follows a cartoon style but does not reach the polish of an artist-made character.
  There is no facial rig or blend shapes; eyes, mouth and fingers are not animated.
- Clothing is skinned (linear blend skinning), not cloth-simulated. Clipping is verified for the bind
  pose and the supplied idle animation. Other animations with extreme poses can still cause clothing
  to intersect, as with any skinned garment.
- Clothing is fitted per gender and per body. It is not intended for differently proportioned bodies.
- The stethoscope, ID card and backpack come in two fits (over the coat / over the scrubs) because a
  rigid accessory cannot rest on both layers. Worn with neither layer they sit about 1 cm off the body.
- Shop balance, purchases and level are a demo stored in the browser's `localStorage`; there is no server.
