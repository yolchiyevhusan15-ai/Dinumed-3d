"""Minimal, dependency-free glTF 2.0 binary (GLB) writer with skins,
PBR materials, embedded PNG textures and animations."""
import json
import struct
import numpy as np

FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


class GLB:
    def __init__(self, generator="Dinumed procedural asset generator 1.0", copyright=None):
        self.j = {"asset": {"version": "2.0", "generator": generator},
                  "scene": 0, "scenes": [{"name": "Scene", "nodes": []}],
                  "nodes": [], "meshes": [], "materials": [], "accessors": [],
                  "bufferViews": [], "buffers": []}
        if copyright:
            self.j["asset"]["copyright"] = copyright
        self.bin = bytearray()
        self._img_cache = {}

    # ----------------------------------------------------------- low level
    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.j["bufferViews"].append(v)
        return len(self.j["bufferViews"]) - 1

    def accessor(self, arr, ctype, atype, target=None, normalized=False, minmax=False):
        arr = np.ascontiguousarray(arr)
        dt = {FLOAT: np.float32, UBYTE: np.uint8, USHORT: np.uint16, UINT: np.uint32}[ctype]
        arr = arr.astype(dt)
        v = self._view(arr.tobytes(), target)
        count = arr.shape[0]
        a = {"bufferView": v, "componentType": ctype, "count": int(count), "type": atype}
        if normalized:
            a["normalized"] = True
        if minmax:
            flat = arr.reshape(count, -1)
            a["min"] = [float(x) for x in flat.min(0)]
            a["max"] = [float(x) for x in flat.max(0)]
        self.j["accessors"].append(a)
        return len(self.j["accessors"]) - 1

    # ----------------------------------------------------------- high level
    def image(self, png_bytes, name):
        if name in self._img_cache:
            return self._img_cache[name]
        self.j.setdefault("images", [])
        self.j.setdefault("textures", [])
        self.j.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987,
                                         "wrapS": 33071, "wrapT": 33071}])
        v = self._view(png_bytes)
        self.j["images"].append({"bufferView": v, "mimeType": "image/png", "name": name})
        self.j["textures"].append({"sampler": 0, "source": len(self.j["images"]) - 1, "name": name})
        t = len(self.j["textures"]) - 1
        self._img_cache[name] = t
        return t

    def material(self, name, color, metallic=0.0, roughness=0.6, texture=None,
                 double_sided=False, alpha=None, emissive=None):
        pbr = {"baseColorFactor": [float(c) for c in color],
               "metallicFactor": float(metallic), "roughnessFactor": float(roughness)}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        m = {"name": name, "pbrMetallicRoughness": pbr}
        if double_sided:
            m["doubleSided"] = True
        if alpha == "BLEND":
            m["alphaMode"] = "BLEND"
        if emissive is not None:
            m["emissiveFactor"] = [float(x) for x in emissive]
        self.j["materials"].append(m)
        return len(self.j["materials"]) - 1

    def node(self, name, translation=None, rotation=None, children=None, **kw):
        n = {"name": name}
        if translation is not None and np.any(np.abs(translation) > 0):
            n["translation"] = [float(x) for x in translation]
        if rotation is not None:
            n["rotation"] = [float(x) for x in rotation]
        if children:
            n["children"] = list(children)
        n.update(kw)
        self.j["nodes"].append(n)
        return len(self.j["nodes"]) - 1

    def skeleton(self, names, parents, world_pos, root_name="Armature"):
        """Creates bone nodes (identity rotations) + armature root.
        Returns (armature_node, {bone_name: node_index})."""
        idx = {}
        # create nodes in order (children filled later)
        for n in names:
            p = parents[n]
            t = world_pos[n] - (world_pos[p] if p else 0)
            idx[n] = self.node(n, translation=t)
        for n in names:
            p = parents[n]
            if p:
                self.j["nodes"][idx[p]].setdefault("children", []).append(idx[n])
        roots = [idx[n] for n in names if not parents[n]]
        arm = self.node(root_name, children=roots)
        self.j["scenes"][0]["nodes"].append(arm)
        return arm, idx

    def skin(self, names, idx, world_pos, name="Skin"):
        ibm = np.zeros((len(names), 4, 4), np.float32)
        for i, n in enumerate(names):
            m = np.eye(4)
            m[:3, 3] = -world_pos[n]
            ibm[i] = m.T  # column-major
        acc = self.accessor(ibm.reshape(len(names), 16), FLOAT, "MAT4")
        self.j.setdefault("skins", []).append({
            "name": name, "inverseBindMatrices": acc,
            "joints": [idx[n] for n in names], "skeleton": idx[names[0]]})
        return len(self.j["skins"]) - 1

    def skinned_mesh(self, name, mesh, joints, weights, material, skin):
        P = mesh.P.astype(np.float32)
        attrs = {"POSITION": self.accessor(P, FLOAT, "VEC3", ARRAY_BUFFER, minmax=True),
                 "NORMAL": self.accessor(mesh.N.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER)}
        if mesh.UV is not None:
            attrs["TEXCOORD_0"] = self.accessor(mesh.UV.astype(np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
        if mesh.C is not None:
            c = np.clip(np.round(mesh.C * 255), 0, 255).astype(np.uint8)
            attrs["COLOR_0"] = self.accessor(c, UBYTE, "VEC4", ARRAY_BUFFER, normalized=True)
        attrs["JOINTS_0"] = self.accessor(joints, UBYTE, "VEC4", ARRAY_BUFFER)
        attrs["WEIGHTS_0"] = self.accessor(weights, FLOAT, "VEC4", ARRAY_BUFFER)
        it = USHORT if len(P) < 65535 else UINT
        ind = self.accessor(mesh.F.reshape(-1), it, "SCALAR", ELEMENT_ARRAY_BUFFER)
        self.j["meshes"].append({"name": name, "primitives": [
            {"attributes": attrs, "indices": ind, "material": material, "mode": 4}]})
        mi = len(self.j["meshes"]) - 1
        n = self.node(name, mesh=mi, skin=skin)
        self.j["scenes"][0]["nodes"].append(n)
        return n

    def animation(self, name, times, tracks, idx):
        """tracks: {bone_name: (T,4) quaternions xyzw}"""
        tin = self.accessor(np.asarray(times, np.float32), FLOAT, "SCALAR", minmax=True)
        samplers, channels = [], []
        for bone, q in tracks.items():
            out = self.accessor(np.asarray(q, np.float32), FLOAT, "VEC4")
            samplers.append({"input": tin, "output": out, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": idx[bone], "path": "rotation"}})
        self.j.setdefault("animations", []).append(
            {"name": name, "samplers": samplers, "channels": channels})

    def write(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("meshes", "materials", "accessors"):
            if not self.j[k]:
                del self.j[k]
        js = json.dumps(self.j, separators=(",", ":")).encode()
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942)); f.write(bytes(self.bin))
        return total
