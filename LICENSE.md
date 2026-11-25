# Licenses

## 1. DINUMED avatar assets and generator — CC0 1.0 Universal

Covers everything in `assets/` (GLB models, animations, PNG textures, manifest),
`source/generator/`, `demo/`, `tools/` and `tests/`.

All meshes, rigs, skin weights, animations and textures in this package were
generated from scratch by the procedural generator in `source/generator/`
(no third-party models, scans, textures or mocap data were used). They are
dedicated to the public domain under
**CC0 1.0 Universal** — https://creativecommons.org/publicdomain/zero/1.0/
— and may be used, modified and redistributed for any purpose, including
commercially, without attribution.

The project owner may replace this dedication with a different license for
their own distribution; nothing in the assets depends on CC0.

Text in the ID-card and watch-face textures was rasterised with the DejaVu
Sans font (Bitstream Vera / DejaVu license, which permits embedding rendered
glyphs). No font files are distributed in this package.

## 2. three.js r187 (`vendor/three/`) — MIT

`vendor/three/` is an unmodified copy of the npm package `three@0.187.0`
(including `GLTFLoader`, `OrbitControls` and `RoomEnvironment` from
`examples/jsm/`). Copyright © 2010-2026 three.js authors. MIT License — see
`vendor/three/LICENSE`.

## 3. Development-only tools (not shipped, not vendored)

- `gltf-validator` (Apache-2.0, The Khronos Group) — installed by `npm install`,
  used only by `tools/validate.mjs`.
- Python packages used by the generator: numpy (BSD-3), scipy (BSD-3),
  scikit-image (BSD-3), fast-simplification (MIT), Pillow (MIT-CMU).
- Playwright (Apache-2.0), used only by `tests/threejs-test.mjs`.
