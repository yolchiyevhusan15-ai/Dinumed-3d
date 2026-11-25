// DINUMED Garderob: avatar viewer + wardrobe shop (offline, three.js r187 from ../vendor).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import { Avatar, loadManifest, loadCatalog, loadGLTF, resolveItemFile, applyColors } from './avatar.js';

const params = new URLSearchParams(location.search);
const $ = (s) => document.querySelector(s);
const STORE = 'dinumed.garderob.v2';

// ------------------------------------------------------------------ scene
const stage = $('#stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;
const key = new THREE.DirectionalLight(0xfff4e8, 1.7);
key.position.set(1.6, 3.2, 2.6);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 2.3, bottom: -0.2, near: 0.5, far: 8 });
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.02;
scene.add(key);
const rim = new THREE.DirectionalLight(0x9fe8ff, 0.7);
rim.position.set(-2, 2, -1.5);
scene.add(rim);
scene.add(new THREE.HemisphereLight(0xffffff, 0x2c6e78, 0.55));
const ground = new THREE.Mesh(new THREE.CircleGeometry(1.5, 64), new THREE.ShadowMaterial({ opacity: 0.28 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const disc = new THREE.Mesh(new THREE.CircleGeometry(0.75, 64),
  new THREE.MeshBasicMaterial({ color: 0x7ef0c8, transparent: true, opacity: 0.10 }));
disc.rotation.x = -Math.PI / 2;
disc.position.y = -0.001;
scene.add(disc);

const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 0.5;
controls.maxDistance = 7;

const CAMS = {
  three: [[1.3, 1.15, 4.3], [0, 0.97, 0]], front: [[0, 1.05, 4.6], [0, 0.97, 0]],
  back: [[-1.2, 1.2, -4.3], [0, 0.97, 0]], side: [[4.5, 1.1, 0], [0, 0.97, 0]],
  face: [[0.22, 0.02, 0.95], [0, 0, 0]], headside: [[0.9, 0.03, 0.3], [0, 0, 0]],
  torso: [[0.5, 0.1, 1.7], [0, 0, 0]], feet: [[0.6, 0.35, 1.1], [0, 0.08, 0]],
  hand: [[1.2, 1.1, 0.9], [0.5, 1.0, 0]],
};
const ANCHOR = { face: ['Head', [0, 0.10, 0]], headside: ['Head', [0, 0.10, 0]], torso: ['Spine2', [0, -0.02, 0]] };
let camName = 'three';
function setCam(name) {
  let [p, t] = CAMS[name] || CAMS.three;
  const a = ANCHOR[name];
  if (a && avatar && avatar.bones.get(a[0])) {
    const w = avatar.bones.get(a[0]).getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(...a[1]));
    p = [w.x + p[0], w.y + p[1], w.z + p[2]];
    t = [w.x, w.y, w.z];
  }
  if (!a && name !== 'feet' && name !== 'hand') {
    // full-body views: back off just enough to fit the avatar in the current aspect ratio
    const tv = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const d = Math.max(1.02 / tv, 0.72 / (tv * camera.aspect));
    const dir = new THREE.Vector3(p[0] - t[0], p[1] - t[1], p[2] - t[2]).normalize();
    p = [t[0] + dir.x * d, t[1] + dir.y * d, t[2] + dir.z * d];
  }
  camName = name;
  camera.position.set(...p);
  controls.target.set(...t);
  controls.update();
  document.querySelectorAll('#views button').forEach((b) => b.classList.toggle('on', b.dataset.cam === name));
}
function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (avatar) setCam(camName);
}
addEventListener('resize', resize);

// ------------------------------------------------------------------ state
let manifest, catalog, avatar;
let playing = params.get('anim') !== '0';
const mem = { get() { try { return JSON.parse(localStorage.getItem(STORE)); } catch { return null; } },
              set(v) { try { localStorage.setItem(STORE, JSON.stringify(v)); } catch { /* private mode */ } } };
let state;
function freshState() {
  return { balance: catalog.startBalance, xp: 0, owned: [...catalog.defaultOwned], worn: [...catalog.defaultWorn],
           gender: 'male', skin: 'base', hair: 'brown' };
}
const save = () => { if (!params.has('fresh')) mem.set(state); };
const item = (id) => catalog.items.find((i) => i.id === id);
const toast = (msg) => { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove('show'), 1800); };

function levelInfo() {
  const L = catalog.levels;
  let i = 0;
  while (i + 1 < L.length && state.xp >= L[i + 1].xp) i++;
  const next = L[i + 1];
  return { cur: L[i], next, pct: next ? (state.xp - L[i].xp) / (next.xp - L[i].xp) * 100 : 100 };
}

async function applyOutfit() {
  const entries = state.worn.map(item).filter(Boolean).map((i) => ({ key: i.id, asset: i.asset, colors: i.colors }));
  await avatar.setOutfit(entries);
}

function wear(id) {
  const it = item(id);
  if (state.worn.includes(id)) state.worn = state.worn.filter((w) => w !== id);
  else state.worn = state.worn.filter((w) => item(w).slot !== it.slot).concat(id);
  save(); renderShop(); return applyOutfit();
}

function buy(id) {
  const it = item(id);
  if (state.balance < it.price) { toast(`DC yetarli emas: yana ${it.price - state.balance} DC kerak`); return; }
  state.balance -= it.price;
  state.xp += Math.round(it.price / 4);
  state.owned.push(id);
  toast(`Sotib olindi: ${it.title}`);
  renderHeader();
  return wear(id);
}

// ------------------------------------------------------------------ UI
function renderHeader() {
  $('#balance').textContent = state.balance;
  const li = levelInfo();
  $('#levelChip').textContent = `${li.cur.level}-daraja · ${li.cur.title}`;
  $('#levelTitle').textContent = `${li.cur.level}-daraja · ${li.cur.title}`;
  $('#xpText').textContent = li.next ? `${state.xp} / ${li.next.xp} XP · keyingi daraja: ${li.next.title}` : `${state.xp} XP · eng yuqori daraja`;
  $('#xpBar').style.width = li.pct.toFixed(0) + '%';
}

const thumbs = new Map();     // `${gender}:${id}` -> dataURL
function renderShop() {
  const root = $('#tab-shop');
  root.textContent = '';
  for (const cat of catalog.categories) {
    const h = document.createElement('h3');
    h.textContent = cat.title;
    const grid = document.createElement('div');
    grid.className = 'grid';
    for (const it of catalog.items.filter((i) => i.category === cat.id)) {
      const owned = state.owned.includes(it.id), worn = state.worn.includes(it.id);
      const card = document.createElement('div');
      card.className = 'card' + (worn ? ' worn' : '');
      card.dataset.id = it.id;
      const th = document.createElement('div');
      th.className = 'thumb';
      const src = thumbs.get(`${state.gender}:${it.id}`);
      if (src) { const img = new Image(); img.src = src; img.alt = it.title; th.appendChild(img); }
      if (it.tag) { const tg = document.createElement('span'); tg.className = 'tag'; tg.textContent = it.tag; th.appendChild(tg); }
      const nm = document.createElement('div');
      nm.className = 'name';
      nm.textContent = it.title;
      const b = document.createElement('button');
      b.className = 'buy' + (worn ? ' on' : owned ? ' owned' : (state.balance < it.price ? ' poor' : ''));
      if (worn) b.textContent = 'Yechish';
      else if (owned) b.textContent = 'Kiyish';
      else { b.innerHTML = `<span>Sotib olish</span><span class="pr"><i class="coin"></i>${it.price}</span>`; }
      b.onclick = () => (owned ? wear(it.id) : buy(it.id));
      card.append(th, nm, b);
      grid.appendChild(card);
    }
    root.append(h, grid);
  }
}

function renderLook() {
  for (const [kind, el] of [['skin', $('#skin')], ['hair', $('#hair')]]) {
    el.textContent = '';
    for (const o of catalog.appearance[kind]) {
      const b = document.createElement('button');
      b.className = 'sw' + (state[kind] === o.id ? ' on' : '');
      b.title = o.title;
      b.setAttribute('aria-label', o.title);
      b.style.background = kind === 'skin'
        ? ({ light: '#f7d3bc', base: '#eebf9f', tan: '#c99673', dark: '#8a5f45' })[o.id] : (o.color || '#7a4a2c');
      b.onclick = () => { state[kind] = o.id; save(); applyLook(); renderLook(); };
      el.appendChild(b);
    }
  }
  document.querySelectorAll('#gender button').forEach((b) => b.classList.toggle('on', b.dataset.g === state.gender));
}
function applyLook() {
  avatar.setSkin(catalog.appearance.skin.find((s) => s.id === state.skin)?.color);
  avatar.setHair(catalog.appearance.hair.find((s) => s.id === state.hair)?.color);
}
function renderEarn() {
  const el = $('#earn');
  el.textContent = '';
  for (const e of catalog.earn) {
    const d = document.createElement('div');
    d.className = 'earn';
    d.innerHTML = `<span></span><b style="white-space:nowrap">+${e.dc} DC</b>`;
    d.firstChild.textContent = e.title;
    el.appendChild(d);
  }
}

// ------------------------------------------------------- item thumbnails
class ThumbStudio {
  constructor() {
    this.r = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.r.setSize(256, 256, false);
    this.r.toneMapping = THREE.ACESFilmicToneMapping;
    this.scene = new THREE.Scene();
    this.scene.environment = new THREE.PMREMGenerator(this.r).fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.8;
    const l = new THREE.DirectionalLight(0xffffff, 1.6);
    l.position.set(1, 2, 2);
    this.scene.add(l, new THREE.HemisphereLight(0xffffff, 0x4a8a90, 0.7));
    this.cam = new THREE.PerspectiveCamera(24, 1, 0.01, 20);
  }
  async render(it, gender) {
    const file = resolveItemFile(manifest, gender, it.asset, new Set());
    const gltf = await avatar.gltf(file);
    const obj = cloneSkinned(gltf.scene);
    const meshes = [];
    obj.traverse((o) => { if (o.isSkinnedMesh) { meshes.push(o); o.frustumCulled = false; } });
    applyColors(meshes, it.colors);
    this.scene.add(obj);
    obj.updateMatrixWorld(true);
    const box = new THREE.Box3();
    for (const m of meshes) { m.geometry.computeBoundingBox(); box.union(m.geometry.boundingBox); }
    const c = box.getCenter(new THREE.Vector3());
    const rad = box.getSize(new THREE.Vector3()).length() / 2;
    const dir = ({ back: [-0.55, 0.25, -1], wrist: [0.9, 0.9, 0.5], feet: [0.9, 0.55, 1], head: [0.6, 0.3, 1], face: [0.35, 0.1, 1] })[it.slot] || [0.45, 0.2, 1];
    const d = rad / Math.sin(THREE.MathUtils.degToRad(12)) * 1.02;
    this.cam.position.copy(c).add(new THREE.Vector3(...dir).normalize().multiplyScalar(d));
    this.cam.lookAt(c);
    this.r.render(this.scene, this.cam);
    const url = this.r.domElement.toDataURL('image/png');
    this.scene.remove(obj);
    return url;
  }
}
let studio, thumbJob = 0;
async function makeThumbs() {
  const job = ++thumbJob, gender = state.gender;
  studio ||= new ThumbStudio();
  for (const it of catalog.items) {
    const k = `${gender}:${it.id}`;
    if (thumbs.has(k)) continue;
    try { thumbs.set(k, await studio.render(it, gender)); } catch (e) { console.warn('thumb', it.id, e); }
    if (job !== thumbJob) return;
    const th = document.querySelector(`.card[data-id="${it.id}"] .thumb`);
    if (th && !th.querySelector('img')) { const img = new Image(); img.src = thumbs.get(k); img.alt = it.title; th.prepend(img); }
  }
  window.__thumbsReady = true;
}

// ------------------------------------------------------------------ boot
async function setGender(g) {
  state.gender = g;
  save();
  $('#loading').hidden = false;
  await avatar.setGender(g);
  applyLook();
  await applyOutfit();
  avatar.play(playing);
  $('#loading').hidden = true;
  renderLook(); renderShop();
  makeThumbs();
}

async function main() {
  [manifest, catalog] = await Promise.all([loadManifest('../'), loadCatalog('../')]);
  state = (!params.has('fresh') && mem.get()) || freshState();
  // drop anything the catalogue no longer has
  state.owned = state.owned.filter(item); state.worn = state.worn.filter(item);
  if (params.has('gender')) state.gender = params.get('gender');
  if (params.has('wear')) {
    state.worn = params.get('wear').split(',').filter(item);
    state.owned = [...new Set([...state.owned, ...state.worn])];
  }
  avatar = new Avatar(scene, manifest, '../');
  renderHeader(); renderEarn(); renderLook(); renderShop();
  resize();
  setCam('three');
  await setGender(state.gender);
  if (params.has('t')) { playing = false; avatar.setTime(Number(params.get('t'))); }
  setCam(params.get('cam') || 'three');
  window.__state = state;
  window.__ready = true;
}

$('#gender').addEventListener('click', (e) => { if (e.target.dataset.g && e.target.dataset.g !== state.gender) setGender(e.target.dataset.g); });
$('#views').addEventListener('click', (e) => { if (e.target.dataset.cam) setCam(e.target.dataset.cam); });
$('#tabs').addEventListener('click', (e) => {
  const t = e.target.dataset.tab;
  if (!t) return;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('on', b === e.target));
  for (const k of ['shop', 'look', 'dc']) $('#tab-' + k).hidden = k !== t;
});
$('#play').onclick = () => { playing = !playing; avatar.play(playing); $('#play').textContent = playing ? "Harakatni to'xtatish" : 'Harakatni yoqish'; };
$('#bind').onclick = () => { playing = false; avatar.bindPose(); $('#play').textContent = 'Harakatni yoqish'; };
$('#undress').onclick = () => { state.worn = []; save(); renderShop(); applyOutfit(); };
$('#topup').onclick = () => { state.balance += 100; state.xp += 150; save(); renderHeader(); renderShop(); toast('+100 DC'); };
$('#reset').onclick = async () => { state = freshState(); window.__state = state; save(); renderHeader(); await setGender('male'); toast('Qaytadan boshlandi'); };

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  if (avatar && playing) avatar.update(dt);
  controls.update();
  renderer.render(scene, camera);
});

main().catch((e) => { console.error(e); $('#loading').hidden = false; $('#loading').textContent = 'Xato: ' + e.message; });
