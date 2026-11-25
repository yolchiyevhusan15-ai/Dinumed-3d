// Validates every GLB with the official Khronos glTF-Validator and checks the
// manifest (existence, byte sizes, SHA-256, size budgets).
// Usage: npm install && node tools/validate.mjs
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import validator from 'gltf-validator';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const man = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets/manifest.json'), 'utf8'));
const rows = [];
let failures = 0;

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]);
}

const listed = new Map();
const add = (file, bytes, sha) => listed.set(file, { bytes, sha });
man.characters.forEach((c) => add(c.file, c.bytes, c.sha256));
man.animations.forEach((a) => add(a.file, a.bytes, a.sha256));
man.textures.forEach((t) => add(t.file, t.bytes, t.sha256));
for (const c of man.clothing) {
  if (c.variants) for (const v of Object.values(c.variants)) for (const g of ['male', 'female']) add(v.files[g], v.details[g].bytes, v.details[g].sha256);
  else for (const g of ['male', 'female']) add(c.files[g], c.details[g].bytes, c.details[g].sha256);
}

// every GLB / PNG on disk must be in the manifest and vice versa
const onDisk = walk(path.join(ROOT, 'assets')).map((f) => path.relative(ROOT, f).split(path.sep).join('/'))
  .filter((f) => /\.(glb|png)$/.test(f));
for (const f of onDisk) if (!listed.has(f)) { console.log('NOT IN MANIFEST:', f); failures++; }
for (const [f, { bytes, sha }] of listed) {
  const p = path.join(ROOT, f);
  if (!fs.existsSync(p)) { console.log('MISSING FILE:', f); failures++; continue; }
  const data = fs.readFileSync(p);
  if (data.length !== bytes) { console.log('SIZE MISMATCH:', f); failures++; }
  if (crypto.createHash('sha256').update(data).digest('hex') !== sha) { console.log('SHA256 MISMATCH:', f); failures++; }
}

// shop catalogue must only reference existing assets and mesh names
const cat = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets/catalog.json'), 'utf8'));
for (const it of cat.items) {
  const c = man.clothing.find((x) => x.id === it.asset);
  if (!c) { console.log('CATALOG: unknown asset', it.id, it.asset); failures++; continue; }
  const det = c.variants ? Object.values(c.variants)[0].details.male : c.details.male;
  for (const mesh of Object.keys(it.colors || {})) {
    if (!det.meshes.some((m) => m.name === mesh)) { console.log('CATALOG: unknown mesh', it.id, mesh); failures++; }
  }
}
for (const id of [...cat.defaultOwned, ...cat.defaultWorn]) if (!cat.items.some((i) => i.id === id)) { console.log('CATALOG: unknown default', id); failures++; }

for (const f of [...listed.keys()].filter((f) => f.endsWith('.glb'))) {
  const data = new Uint8Array(fs.readFileSync(path.join(ROOT, f)));
  const r = await validator.validateBytes(data, { uri: f, maxIssues: 50 });
  const { numErrors, numWarnings, numInfos } = r.issues;
  const size = data.length;
  const isChar = f.includes('/characters/');
  const sizeOk = !isChar || size < 3 * 1024 * 1024;
  if (numErrors || !sizeOk) failures++;
  rows.push({ file: f, kb: (size / 1024).toFixed(1), errors: numErrors, warnings: numWarnings, infos: numInfos,
    messages: r.issues.messages.filter((m) => m.severity <= 1).map((m) => `${m.code}: ${m.message}`).slice(0, 5) });
}
let md = '# glTF-Validator report\n\nKhronos glTF-Validator ' + validator.version() + '\n\n| File | KB | Errors | Warnings | Infos |\n|---|---|---|---|---|\n';
md += rows.map((r) => `| ${r.file} | ${r.kb} | ${r.errors} | ${r.warnings} | ${r.infos} |`).join('\n') + '\n';
const warn = rows.filter((r) => r.messages.length);
if (warn.length) md += '\n## Messages\n\n' + warn.map((r) => `- ${r.file}: ${r.messages.join('; ')}`).join('\n') + '\n';
fs.writeFileSync(path.join(ROOT, 'tests', 'VALIDATOR.md'), md);
for (const r of rows) console.log(`${r.errors ? 'FAIL' : 'ok  '} ${r.file.padEnd(52)} ${r.kb.padStart(8)} KB  E${r.errors} W${r.warnings} I${r.infos}`);
console.log(failures ? `\n${failures} problem(s)` : `\nAll ${rows.length} GLB files valid; manifest sizes and SHA-256 match.`);
process.exit(failures ? 1 : 0);
