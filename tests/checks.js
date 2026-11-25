// In-browser verification of every DINUMED asset with the vendored three.js.
// Loaded by tests/threejs-test.html, driven by tests/threejs-test.mjs.
import * as THREE from 'three';
import { loadGLTF, loadManifest, attachClothing } from '../demo/avatar.js';

const TOL = 0.001;          // 1 mm penetration tolerance
const CONTACT = 0.0025;     // accessories must be within 2.5 mm of their support
const results = { checks: [], clipping: [], contact: [], info: {} };
const ok = (name, pass, detail = '') => { results.checks.push({ name, pass: !!pass, detail }); return pass; };

// ---------------------------------------------------------------- geometry
function skinnedSnapshot(mesh) {
  // world-space skinned positions + normals of a SkinnedMesh at the current pose
  const g = mesh.geometry;
  const n = g.attributes.position.count;
  const P = new Float32Array(n * 3), N = new Float32Array(n * 3);
  const v = new THREE.Vector3(), nn = new THREE.Vector3();
  const sk = mesh.skeleton, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
  const m = new THREE.Matrix4(), acc = new THREE.Matrix4();
  for (let i = 0; i < n; i++) {
    mesh.getVertexPosition(i, v);       // applies skinning (bind matrix + bones)
    v.applyMatrix4(mesh.matrixWorld);
    P.set([v.x, v.y, v.z], i * 3);
    acc.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (let k = 0; k < 4; k++) {
      const w = sw.getComponent(i, k);
      if (!w) continue;
      const b = si.getComponent(i, k);
      m.multiplyMatrices(sk.bones[b].matrixWorld, sk.boneInverses[b]);
      for (let e = 0; e < 16; e++) acc.elements[e] += m.elements[e] * w;
    }
    nn.fromBufferAttribute(g.attributes.normal, i).transformDirection(acc);
    N.set([nn.x, nn.y, nn.z], i * 3);
  }
  return { P, N, F: g.index.array, n };
}

function merge(snaps) {
  let nv = 0, nf = 0;
  snaps.forEach((s) => { nv += s.n; nf += s.F.length; });
  const P = new Float32Array(nv * 3), N = new Float32Array(nv * 3), F = new Uint32Array(nf);
  let ov = 0, of = 0;
  for (const s of snaps) {
    P.set(s.P, ov * 3); N.set(s.N, ov * 3);
    for (let i = 0; i < s.F.length; i++) F[of + i] = s.F[i] + ov;
    ov += s.n; of += s.F.length;
  }
  return { P, N, F, n: nv };
}

class TriGrid {
  constructor(S, cell = 0.02) {
    this.S = S; this.cell = cell; this.map = new Map();
    const { P, F } = S;
    // open-surface boundary (edges used by one triangle): the sign of the
    // distance is undefined beyond a sheet's edge, so those hits never count
    const ec = new Map();
    const ek = (a, b) => (a < b ? a * 4194304 + b : b * 4194304 + a);
    for (let t = 0; t < F.length / 3; t++) for (let k = 0; k < 3; k++) {
      const key = ek(F[t * 3 + k], F[t * 3 + (k + 1) % 3]);
      ec.set(key, (ec.get(key) || 0) + 1);
    }
    this.bEdge = new Set(); this.bVert = new Set();
    for (const [key, c] of ec) if (c === 1) {
      this.bEdge.add(key);
      this.bVert.add(Math.floor(key / 4194304)); this.bVert.add(key % 4194304);
    }
    this.ek = ek;
    for (let t = 0; t < F.length / 3; t++) {
      let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
      for (let k = 0; k < 3; k++) {
        const i = F[t * 3 + k];
        for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], P[i * 3 + a]); hi[a] = Math.max(hi[a], P[i * 3 + a]); }
      }
      const c0 = lo.map((x) => Math.floor(x / cell)), c1 = hi.map((x) => Math.floor(x / cell));
      for (let x = c0[0]; x <= c1[0]; x++) for (let y = c0[1]; y <= c1[1]; y++) for (let z = c0[2]; z <= c1[2]; z++) {
        const key = x + ',' + y + ',' + z;
        if (!this.map.has(key)) this.map.set(key, []);
        this.map.get(key).push(t);
      }
    }
  }
  // signed distance of p to the surface (sign from interpolated vertex normals), within radius
  query(p, r = 0.02) {
    const { P, N, F } = this.S;
    const c = this.cell;
    const x0 = Math.floor((p[0] - r) / c), x1 = Math.floor((p[0] + r) / c);
    const y0 = Math.floor((p[1] - r) / c), y1 = Math.floor((p[1] + r) / c);
    const z0 = Math.floor((p[2] - r) / c), z1 = Math.floor((p[2] + r) / c);
    let best = Infinity, bestSigned = null;
    const seen = new Set();
    for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) {
      const list = this.map.get(x + ',' + y + ',' + z);
      if (!list) continue;
      for (const t of list) {
        if (seen.has(t)) continue;
        seen.add(t);
        const ia = F[t * 3], ib = F[t * 3 + 1], ic = F[t * 3 + 2];
        const res = closestOnTri(p, P, ia, ib, ic);
        if (res.d2 < best) {
          best = res.d2;
          const f = res.feat, ids = [ia, ib, ic];
          this.lastBoundary = f.length === 1 ? this.bVert.has(ids[f[0]])
            : f.length === 2 ? this.bEdge.has(this.ek(ids[f[0]], ids[f[1]])) : false;
          const nx = N[ia * 3] * res.u + N[ib * 3] * res.v + N[ic * 3] * res.w;
          const ny = N[ia * 3 + 1] * res.u + N[ib * 3 + 1] * res.v + N[ic * 3 + 1] * res.w;
          const nz = N[ia * 3 + 2] * res.u + N[ib * 3 + 2] * res.v + N[ic * 3 + 2] * res.w;
          const dot = (p[0] - res.q[0]) * nx + (p[1] - res.q[1]) * ny + (p[2] - res.q[2]) * nz;
          bestSigned = Math.sqrt(best) * (dot < 0 ? -1 : 1);
        }
      }
    }
    this.boundary = this.lastBoundary;
    if (best > r * r) return null;      // nothing within the search radius
    return bestSigned;
  }
}

function closestOnTri(p, P, ia, ib, ic) {
  // Ericson, Real-Time Collision Detection 5.1.5
  const a = [P[ia * 3], P[ia * 3 + 1], P[ia * 3 + 2]], b = [P[ib * 3], P[ib * 3 + 1], P[ib * 3 + 2]], c = [P[ic * 3], P[ic * 3 + 1], P[ic * 3 + 2]];
  const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  const out = (u, v, w, feat = [0, 1, 2]) => {
    const q = [a[0] * u + b[0] * v + c[0] * w, a[1] * u + b[1] * v + c[1] * w, a[2] * u + b[2] * v + c[2] * w];
    const d = sub(p, q);
    return { q, u, v, w, d2: dot(d, d), feat };
  };
  if (d1 <= 0 && d2 <= 0) return out(1, 0, 0, [0]);
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return out(0, 1, 0, [1]);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); return out(1 - v, v, 0, [0, 1]); }
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return out(0, 0, 1, [2]);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); return out(1 - w, 0, w, [0, 2]); }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return out(0, 1 - w, w, [1, 2]); }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return out(1 - v - w, v, w);
}

/** penetration of `outer` vertices into `inner` surface. */
function penetration(outer, innerGrid, radius = 0.02) {
  let count = 0, maxDepth = 0, tested = 0, minAbs = Infinity, worst = null;
  const p = [0, 0, 0];
  for (let i = 0; i < outer.n; i++) {
    p[0] = outer.P[i * 3]; p[1] = outer.P[i * 3 + 1]; p[2] = outer.P[i * 3 + 2];
    const s = innerGrid.query(p, radius);
    if (s === null) continue;
    tested++;
    minAbs = Math.min(minAbs, Math.abs(s));
    if (innerGrid.boundary) continue;     // closest point on an open edge: no "inside"
    if (s < -TOL) { count++; if (-s > maxDepth) { maxDepth = -s; worst = [p[0], p[1], p[2]].map((x) => +x.toFixed(4)); } }
  }
  return { count, maxDepth, tested, minDistance: minAbs, worst };
}

// ----------------------------------------------------------------- tests
async function run() {
  const manifest = await loadManifest('../');
  const names = manifest.skeleton.jointOrder;
  results.info.three = THREE.REVISION;
  ok('three.js revision is 187 (vendored 0.187.0)', THREE.REVISION === '187', THREE.REVISION);
  const idle = await loadGLTF('../' + manifest.animations[0].file);
  ok('idle.glb has one clip "Idle"', idle.animations.length === 1 && idle.animations[0].name === 'Idle');
  const clip = idle.animations[0];

  for (const ch of manifest.characters) {
    const g = ch.id;
    const gltf = await loadGLTF('../' + ch.file);
    const root = gltf.scene;
    const scene = new THREE.Scene();
    scene.add(root);
    const bones = new Map();
    const parts = {};
    root.traverse((o) => { if (o.isBone) bones.set(o.name, o); if (o.isSkinnedMesh) parts[o.name] = o; });
    ok(`${g}: character loads with SkinnedMeshes Body/Eyes/Hair/Underwear`,
      ['Body', 'Eyes', 'Hair', 'Underwear'].every((k) => parts[k]), Object.keys(parts).join(','));
    const charSk = parts.Body.skeleton;
    ok(`${g}: skeleton has ${names.length} bones in canonical order`,
      charSk.bones.map((b) => b.name).join() === names.join(), charSk.bones.length);
    ok(`${g}: file size < 3 MB`, ch.bytes < 3 * 1024 * 1024, `${(ch.bytes / 1048576).toFixed(2)} MB`);
    ok(`${g}: embedded Idle animation`, gltf.animations.some((a) => a.name === 'Idle'));
    const trackBones = clip.tracks.map((t) => t.name.split('.')[0]);
    ok(`${g}: every Idle track targets an existing bone`, trackBones.every((b) => bones.has(b)), trackBones.length + ' tracks');
    // bind pose = identity local rotations
    ok(`${g}: bind pose has identity bone rotations (A-pose rest)`,
      [...bones.values()].every((b) => b.quaternion.angleTo(new THREE.Quaternion()) < 1e-6));
    checkWeights(`${g}/character`, Object.values(parts), names.length);

    // ----- load every clothing file and attach
    const items = {};
    for (const c of manifest.clothing) {
      const variants = c.variants ? Object.entries(c.variants).map(([v, d]) => [c.id + '_' + v, d.files[g]]) : [[c.id, c.files[g]]];
      for (const [id, file] of variants) {
        const cg = await loadGLTF('../' + file);
        const meshes = [];
        cg.scene.traverse((o) => { if (o.isSkinnedMesh) meshes.push(o); });
        const sk = meshes[0].skeleton;
        ok(`${g}/${id}: loads, ${meshes.length} skinned mesh(es)`, meshes.length > 0);
        ok(`${g}/${id}: identical bone names/order`, sk.bones.map((b) => b.name).join() === names.join());
        let maxDiff = 0;
        sk.boneInverses.forEach((m, i) => m.elements.forEach((e, k) => { maxDiff = Math.max(maxDiff, Math.abs(e - charSk.boneInverses[i].elements[k])); }));
        ok(`${g}/${id}: bind matrices equal character's (max diff ${maxDiff.toExponential(1)})`, maxDiff < 1e-5);
        checkWeights(`${g}/${id}`, meshes, names.length);
        const attached = attachClothing(root, bones, cg.scene);
        attached.forEach((m) => { m.visible = false; });
        items[id] = attached;
      }
    }
    // ----- clipping + contact in bind pose and over the idle animation
    const mixer = new THREE.AnimationMixer(root);
    const action = mixer.clipAction(clip);
    const poses = [{ label: 'bind', t: null }];
    for (let t = 0; t < clip.duration; t += 1.0) poses.push({ label: `idle t=${t.toFixed(1)}s`, t });
    const pairs = [
      ['Underwear', 'Body'], ['Hair', 'Body'],
      ['scrubs_top', 'Body'], ['scrubs_top', 'Underwear'],
      ['medical_trousers', 'Body'], ['medical_trousers', 'Underwear'],
      ['white_coat', 'Body'], ['white_coat', 'scrubs_top'], ['white_coat', 'medical_trousers'],
      ['scrubs_top', 'medical_trousers'],
      ['medical_shoes', 'Body'], ['medical_trousers', 'medical_shoes'],
      ['medical_cap', 'Body'], ['medical_cap', 'Hair'],
      ['glasses', 'Body'], ['glasses', 'Hair'], ['glasses', 'medical_cap'],
      ['watch', 'Body'], ['white_coat', 'watch'],
      ['stethoscope_over_coat', 'white_coat'], ['stethoscope_over_coat', 'Body'],
      ['id_card_over_coat', 'white_coat'], ['stethoscope_over_coat', 'id_card_over_coat'],
      ['stethoscope_over_scrubs', 'scrubs_top'], ['stethoscope_over_scrubs', 'Body'],
      ['id_card_over_scrubs', 'scrubs_top'], ['stethoscope_over_scrubs', 'id_card_over_scrubs'],
    ];
    const contacts = [
      ['glasses', ['Body']], ['watch', ['Body']], ['medical_cap', ['Body', 'Hair']], ['medical_shoes', ['Body']],
      ['stethoscope_over_coat', ['white_coat', 'Body', 'scrubs_top']], ['id_card_over_coat', ['white_coat']],
      ['stethoscope_over_scrubs', ['scrubs_top', 'Body']], ['id_card_over_scrubs', ['scrubs_top']],
    ];
    const meshesOf = (k) => (parts[k] ? [parts[k]] : items[k]);
    for (const pose of poses) {
      if (pose.t === null) { action.stop(); bones.forEach((b) => b.quaternion.identity()); }
      else { action.play(); action.time = pose.t; mixer.update(0); }
      root.updateMatrixWorld(true);
      const snap = {};
      const get = (k) => (snap[k] ||= merge(meshesOf(k).map((m) => { m.skeleton.update(); return skinnedSnapshot(m); })));
      const grids = {};
      const grid = (k) => (grids[k] ||= new TriGrid(get(k)));
      for (const [outer, inner] of pairs) {
        const r = penetration(get(outer), grid(inner));
        results.clipping.push({ gender: g, pose: pose.label, outer, inner, ...r });
      }
      if (pose.t === null || pose.t === 4) {
        for (const [acc, supports] of contacts) {
          let minD = Infinity;
          for (const s of supports) minD = Math.min(minD, penetration(get(acc), grid(s), 0.03).minDistance);
          results.contact.push({ gender: g, pose: pose.label, accessory: acc, supports, minDistance: minD, pass: minD <= CONTACT });
        }
      }
    }
  }
  // summary checks
  const bind = results.clipping.filter((c) => c.pose === 'bind');
  const anim = results.clipping.filter((c) => c.pose !== 'bind');
  ok('bind pose: no vertex penetrates more than 1 mm in any tested pair', bind.every((c) => c.count === 0),
    bind.filter((c) => c.count).map((c) => `${c.gender} ${c.outer}->${c.inner}: ${c.count}`).join('; '));
  const animBad = anim.filter((c) => c.count > 0);
  results.info.animationPenetrations = animBad.length;
  ok('idle animation: no vertex penetrates more than 1 mm in any tested pair/frame', animBad.length === 0,
    animBad.map((c) => `${c.gender} ${c.pose} ${c.outer}->${c.inner}: ${c.count} (max ${(c.maxDepth * 1000).toFixed(1)} mm)`).join('; '));
  ok('accessories are in contact with their support (<= 2.5 mm)', results.contact.every((c) => c.pass),
    results.contact.filter((c) => !c.pass).map((c) => `${c.gender} ${c.pose} ${c.accessory}: ${(c.minDistance * 1000).toFixed(1)} mm`).join('; '));
  results.passed = results.checks.every((c) => c.pass);
  return results;
}

function checkWeights(label, meshes, nb) {
  let bad = 0, nan = 0, idx = 0, total = 0;
  for (const m of meshes) {
    const g = m.geometry, w = g.attributes.skinWeight, j = g.attributes.skinIndex, p = g.attributes.position;
    for (let i = 0; i < w.count; i++) {
      const s = w.getX(i) + w.getY(i) + w.getZ(i) + w.getW(i);
      if (Math.abs(s - 1) > 1e-3) bad++;
      for (let k = 0; k < 4; k++) if (j.getComponent(i, k) >= nb) idx++;
      if (!Number.isFinite(p.getX(i) + p.getY(i) + p.getZ(i))) nan++;
      total++;
    }
  }
  ok(`${label}: skin weights normalised, joint indices valid, no NaN (${total} verts)`, bad === 0 && idx === 0 && nan === 0,
    `bad=${bad} idx=${idx} nan=${nan}`);
}

window.__results = run().catch((e) => ({ error: String(e && e.stack || e), passed: false }));
