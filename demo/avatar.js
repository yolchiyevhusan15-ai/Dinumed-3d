// DINUMED avatar runtime: loads a character GLB, attaches clothing GLBs to the
// character's skeleton (by bone name), recolours items and plays the idle animation.
// Depends only on the locally vendored three.js r187 (see ../vendor/three).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();

export function loadGLTF(url) {
  return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
}

export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' not found');
  return r.json();
}

export const loadManifest = (base = '../') => loadJSON(base + 'assets/manifest.json');
export const loadCatalog = (base = '../') => loadJSON(base + 'assets/catalog.json');

/** Pick the right file for a clothing asset given the other worn assets.
 *  Items that rest on clothing (stethoscope, ID card, backpack) have two fits. */
export function resolveItemFile(manifest, gender, id, wornAssets) {
  const item = manifest.clothing.find((c) => c.id === id);
  if (!item) throw new Error('unknown item ' + id);
  if (item.variants) {
    const v = wornAssets.has('white_coat') ? 'over_coat' : 'over_scrubs';
    return item.variants[v].files[gender];
  }
  return item.files[gender];
}

/** Rebind every SkinnedMesh of a clothing glTF onto the character skeleton. */
export function attachClothing(characterRoot, characterBones, clothingScene) {
  const meshes = [];
  clothingScene.traverse((o) => { if (o.isSkinnedMesh) meshes.push(o); });
  for (const m of meshes) {
    const bones = m.skeleton.bones.map((b) => {
      const t = characterBones.get(b.name);
      if (!t) throw new Error(`bone ${b.name} missing on character`);
      return t;
    });
    const sk = new THREE.Skeleton(bones, m.skeleton.boneInverses.map((x) => x.clone()));
    m.removeFromParent();
    m.bind(sk, new THREE.Matrix4()); // assets are authored with identity mesh transforms
    m.frustumCulled = false;
    characterRoot.add(m);
  }
  return meshes;
}

/** Give each mesh its own material and apply {meshName: '#rrggbb'} overrides. */
export function applyColors(meshes, colors = {}) {
  for (const m of meshes) {
    m.material = m.material.clone();
    if (colors[m.name]) m.material.color.set(colors[m.name]);
  }
}

export class Avatar {
  constructor(scene, manifest, base = '../') {
    this.scene = scene;
    this.manifest = manifest;
    this.base = base;
    this.gender = null;
    this.root = null;
    this.bones = new Map();
    this.parts = {};           // character meshes by name (Body, Hair, ...)
    this.items = new Map();    // key -> {file, meshes, sig}
    this.outfit = [];
    this.mixer = null;
    this.action = null;
    this.idleClip = null;
    this.cache = new Map();
    this.skin = null;
    this.hair = null;
  }

  gltf(file) {
    if (!this.cache.has(file)) this.cache.set(file, loadGLTF(this.base + file));
    return this.cache.get(file);
  }

  async setGender(gender) {
    if (this.gender === gender) return;
    const outfit = this.outfit;
    this.clear();
    const ch = this.manifest.characters.find((c) => c.id === gender);
    const gltf = await loadGLTF(this.base + ch.file);
    this.gender = gender;
    this.root = gltf.scene;
    this.root.name = 'Avatar_' + gender;
    this.root.traverse((o) => {
      if (o.isBone) this.bones.set(o.name, o);
      if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; this.parts[o.name] = o; o.material = o.material.clone(); }
    });
    this.hairBase = this.parts.Hair ? this.parts.Hair.material.color.clone() : null;
    this.scene.add(this.root);
    this.mixer = new THREE.AnimationMixer(this.root);
    if (!this.idleClip) {
      const a = await loadGLTF(this.base + this.manifest.animations[0].file);
      this.idleClip = a.animations[0];
    }
    this.action = this.mixer.clipAction(this.idleClip);
    this.setSkin(this.skin);
    this.setHair(this.hair);
    await this.setOutfit(outfit);
  }

  clear() {
    if (this.root) this.scene.remove(this.root);
    this.items.clear();
    this.bones.clear();
    this.parts = {};
    this.root = null;
    this.gender = null;
  }

  /** entries: [{key, asset, colors}] - `asset` is a manifest clothing id. */
  async setOutfit(entries) {
    this.outfit = entries;
    const assets = new Set(entries.map((e) => e.asset));
    const want = new Map(entries.map((e) => [e.key, {
      ...e, file: resolveItemFile(this.manifest, this.gender, e.asset, assets),
    }]));
    for (const [key, it] of [...this.items]) {
      const w = want.get(key);
      if (!w || w.file !== it.file) {
        it.meshes.forEach((m) => m.removeFromParent());
        this.items.delete(key);
      }
    }
    for (const [key, w] of want) {
      if (this.items.has(key)) continue;
      const gltf = await this.gltf(w.file);
      if (this.outfit !== entries) return;          // superseded by a newer call
      const meshes = attachClothing(this.root, this.bones, gltf.scene.clone(true));
      applyColors(meshes, w.colors);
      meshes.forEach((m) => { m.castShadow = true; });
      this.items.set(key, { file: w.file, meshes });
    }
    // stand on the soles of whatever footwear is worn
    const ch = this.manifest.characters.find((c) => c.id === this.gender);
    const go = ch.groundOffset || {};
    let lift = 0;
    for (const a of assets) if (go[a]) lift = Math.max(lift, go[a]);
    this.root.position.y = lift;
  }

  /** Simple helper: wear manifest clothing ids with their default colours. */
  setItems(ids) { return this.setOutfit(ids.map((id) => ({ key: id, asset: id, colors: {} }))); }

  setSkin(hex) {
    this.skin = hex || null;
    if (this.parts.Body) this.parts.Body.material.color.set(hex || '#ffffff');
  }

  setHair(hex) {
    this.hair = hex || null;
    if (this.parts.Hair) { if (hex) this.parts.Hair.material.color.set(hex); else this.parts.Hair.material.color.copy(this.hairBase); }
  }

  play(on = true) {
    if (!this.action) return;
    if (on) this.action.reset().play(); else this.action.stop();
  }

  setTime(t) {
    if (!this.action) return;
    this.action.play();
    this.action.paused = false;
    this.action.time = t % this.idleClip.duration;
    this.mixer.update(0);
    this.action.paused = true;
  }

  bindPose() {
    if (this.action) this.action.stop();
    this.bones.forEach((b) => b.quaternion.identity());
  }

  update(dt) { if (this.mixer) this.mixer.update(dt); }
}
