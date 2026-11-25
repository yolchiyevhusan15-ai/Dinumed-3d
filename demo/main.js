import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Avatar, loadManifest } from './avatar.js';

const params = new URLSearchParams(location.search);
const $ = (s) => document.querySelector(s);
$('#rev').textContent = THREE.REVISION;

const view = $('#view');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x1a2228 : 0xe4eaee);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(1.5, 3, 2.5);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 2.2, bottom: -0.2, near: 0.5, far: 8 });
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.02;
scene.add(key);
const fill = new THREE.DirectionalLight(0xdfe8ff, 0.5);
fill.position.set(-2, 1.5, 1);
scene.add(fill);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9aa5, 0.5));

const ground = new THREE.Mesh(new THREE.CircleGeometry(1.6, 64),
  new THREE.MeshStandardMaterial({ color: dark ? 0x2a343b : 0xcfd8dd, roughness: 0.95 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const CAMS = {
  front: [[0, 1.0, 4.2], [0, 0.92, 0]],
  back: [[0, 1.0, -4.2], [0, 0.92, 0]],
  side: [[4.2, 1.0, 0], [0, 0.92, 0]],
  three: [[2.6, 1.4, 3.2], [0, 0.92, 0]],
  face: [[0.25, 1.68, 0.75], [0, 1.64, 0]],
  torso: [[0.5, 1.35, 1.6], [0, 1.25, 0]],
  hand: [[1.2, 1.1, 0.9], [0.5, 1.0, 0]],
  feet: [[0.6, 0.35, 1.1], [0, 0.08, 0]],
  headside: [[0.62, 1.70, 0.25], [0, 1.66, 0]],
  sleeve: [[0.75, 1.35, 0.6], [0.3, 1.25, 0]],
  skirt: [[0.45, 0.8, 0.7], [0.1, 0.75, 0]],
  neck: [[0.35, 1.62, 0.55], [0, 1.45, 0]],
};
function setCam(name) {
  let [p, t] = CAMS[name] || CAMS.front;
  // face / torso views follow the current character's proportions
  const anchor = { face: ['Head', [0, 0.075, 0]], headside: ['Head', [0, 0.07, 0]], neck: ['Neck', [0, -0.01, 0]], torso: ['Spine2', [0, -0.02, 0]] }[name];
  if (anchor && avatar && avatar.bones.get(anchor[0])) {
    const w = avatar.bones.get(anchor[0]).getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(...anchor[1]));
    p = [w.x + p[0] - t[0], w.y + p[1] - t[1], w.z + p[2] - t[2]];
    t = [w.x, w.y, w.z];
  }
  camera.position.set(...p);
  controls.target.set(...t);
  controls.update();
}

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

const status = (msg, err = false) => { const s = $('#status'); s.textContent = msg; s.className = err ? 'err' : ''; };

let manifest, avatar;
let playing = params.get('anim') !== '0';
const worn = new Set();

async function apply() {
  status('Loading…');
  const t0 = performance.now();
  await avatar.setItems([...worn]);
  status(`${avatar.gender} · ${worn.size} item(s) · ${(performance.now() - t0).toFixed(0)} ms\n` +
    [...avatar.items.values()].map((i) => '✓ ' + i.file.split('/').pop()).join('\n'));
  document.querySelectorAll('#items input').forEach((cb) => { cb.checked = worn.has(cb.value); });
}

async function main() {
  manifest = await loadManifest('../');
  avatar = new Avatar(scene, manifest, '../');
  const items = $('#items');
  for (const c of manifest.clothing) {
    const l = document.createElement('label');
    l.className = 'item';
    l.innerHTML = `<input type="checkbox" value="${c.id}"> ${c.title}`;
    l.querySelector('input').addEventListener('change', async (e) => {
      e.target.checked ? worn.add(c.id) : worn.delete(c.id);
      await apply();
    });
    items.appendChild(l);
  }
  const g = params.get('gender') || 'male';
  const initial = params.has('items') ? params.get('items').split(',').filter(Boolean)
    : manifest.clothing.map((c) => c.id);
  initial.forEach((i) => worn.add(i));
  document.querySelectorAll('#gender button').forEach((b) => b.classList.toggle('on', b.dataset.g === g));
  await avatar.setGender(g);
  await apply();
  if (params.has('t')) { playing = false; avatar.setTime(Number(params.get('t'))); }
  else avatar.play(playing);
  $('#play').textContent = playing ? 'Pause idle' : 'Play idle';
  setCam(params.get('cam') || 'three');
  resize();
  window.__ready = true;
}

$('#gender').addEventListener('click', async (e) => {
  const g = e.target.dataset.g;
  if (!g) return;
  document.querySelectorAll('#gender button').forEach((b) => b.classList.toggle('on', b === e.target));
  status('Loading…');
  await avatar.setGender(g);
  await apply();
  avatar.play(playing);
});
$('#all').onclick = async () => { manifest.clothing.forEach((c) => worn.add(c.id)); await apply(); };
$('#none').onclick = async () => { worn.clear(); await apply(); };
$('#play').onclick = () => {
  playing = !playing;
  avatar.play(playing);
  $('#play').textContent = playing ? 'Pause idle' : 'Play idle';
};
$('#bind').onclick = () => {
  playing = false;
  avatar.play(false);
  avatar.root.traverse((o) => { if (o.isBone) { o.quaternion.identity(); } });
  $('#play').textContent = 'Play idle';
};
document.querySelectorAll('[data-cam]').forEach((b) => (b.onclick = () => setCam(b.dataset.cam)));

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  if (avatar && playing) avatar.update(dt);
  controls.update();
  renderer.render(scene, camera);
});

main().catch((e) => { console.error(e); status('Error: ' + e.message, true); });
