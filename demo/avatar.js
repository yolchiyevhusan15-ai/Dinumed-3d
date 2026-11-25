// DINUMED avatar runtime: loads a character GLB, attaches clothing GLBs to the
// character's skeleton (by bone name) and plays the idle animation.
// Depends only on the locally vendored three.js r187 (see ../vendor/three).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();

export function loadGLTF(url) {
  return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
}

export async function loadManifest(base = '../') {
  const r = await fetch(base + 'assets/manifest.json');
  if (!r.ok) throw new Error('manifest.json not found');
  return r.json();
}

/** Pick the right file for a clothing item given what else is worn. */
export function resolveItemFile(manifest, gender, id, worn) {
  const item = manifest.clothing.find((c) => c.id === id);
  if (!item) throw new Error('unknown item ' + id);
  if (item.variants) {
    const v = worn.has('white_coat') ? 'over_coat' : 'over_scrubs';
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

export class Avatar {
  constructor(scene, manifest, base = '../') {
    this.scene = scene;
    this.manifest = manifest;
    this.base = base;
    this.gender = null;
    this.root = null;
    this.bones = new Map();
    this.items = new Map(); // id -> {file, meshes}
    this.mixer = null;
    this.action = null;
    this.idleClip = null;
    this.cache = new Map();
  }

  async _gltf(file) {
    if (!this.cache.has(file)) this.cache.set(file, loadGLTF(this.base + file));
    return this.cache.get(file);
  }

  async setGender(gender) {
    if (this.gender === gender) return;
    const worn = [...this.items.keys()];
    this.clear();
    const ch = this.manifest.characters.find((c) => c.id === gender);
    const gltf = await loadGLTF(this.base + ch.file);
    this.gender = gender;
    this.root = gltf.scene;
    this.root.name = 'Avatar_' + gender;
    this.root.traverse((o) => {
      if (o.isBone) this.bones.set(o.name, o);
      if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; }
    });
    this.embeddedClips = gltf.animations;
    this.scene.add(this.root);
    this.mixer = new THREE.AnimationMixer(this.root);
    if (!this.idleClip) {
      const anim = this.manifest.animations[0];
      const a = await loadGLTF(this.base + anim.file);
      this.idleClip = a.animations[0];
    }
    this.action = this.mixer.clipAction(this.idleClip);
    await this.setItems(worn);
  }

  clear() {
    if (this.root) this.scene.remove(this.root);
    this.items.clear();
    this.bones.clear();
    this.root = null;
    this.gender = null;
  }

  async setItems(ids) {
    const want = new Set(ids);
    // decide the file for every wanted item (variants depend on the full set)
    const files = new Map([...want].map((id) => [id, resolveItemFile(this.manifest, this.gender, id, want)]));
    for (const [id, it] of [...this.items]) {
      if (!want.has(id) || files.get(id) !== it.file) {
        it.meshes.forEach((m) => m.removeFromParent());
        this.items.delete(id);
      }
    }
    for (const id of want) {
      if (this.items.has(id)) continue;
      const file = files.get(id);
      const gltf = await this._gltf(file);
      const scene = gltf.scene.clone(true); // clone so the cache stays reusable
      // SkinnedMesh clones share skeleton; rebuild from names
      const meshes = attachClothing(this.root, this.bones, scene);
      meshes.forEach((m) => { m.castShadow = true; });
      this.items.set(id, { file, meshes });
    }
    // standing on shoe soles
    const fit = this.manifest.characters.find((c) => c.id === this.gender).shoeGroundOffset || 0;
    this.root.position.y = want.has('medical_shoes') ? fit : 0;
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

  update(dt) { if (this.mixer) this.mixer.update(dt); }
}
