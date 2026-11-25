"""Writes assets/manifest.json by reading back every generated GLB."""
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(__file__))
from body import bone_names

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A = os.path.join(REPO, "assets")

ITEMS = [
    ("scrubs_top", "Surgical scrubs (top)", "torso-mid"),
    ("medical_trousers", "Medical trousers (scrub pants)", "legs"),
    ("white_coat", "White medical coat", "torso-outer"),
    ("medical_shoes", "Medical clogs", "feet"),
    ("sneakers", "Comfortable sneakers", "feet"),
    ("medical_mask", "Surgical face mask", "mouth"),
    ("backpack", "Backpack", "back"),
    ("medical_cap", "Medical (scrub) cap", "head"),
    ("stethoscope", "Stethoscope", "neck-accessory"),
    ("glasses", "Glasses", "face"),
    ("id_card", "ID card (clip-on badge)", "chest-accessory"),
    ("watch", "Wrist watch (left wrist)", "wrist"),
]


def glb_info(path):
    data = open(path, "rb").read()
    magic, ver, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and ver == 2 and total == len(data)
    jlen, _ = struct.unpack_from("<II", data, 12)
    j = json.loads(data[20:20 + jlen])
    tris = verts = 0
    meshes = []
    for m in j.get("meshes", []):
        for p in m["primitives"]:
            v = j["accessors"][p["attributes"]["POSITION"]]["count"]
            t = j["accessors"][p["indices"]]["count"] // 3
            verts += v; tris += t
            meshes.append({"name": m["name"], "vertices": v, "triangles": t,
                           "material": j["materials"][p["material"]]["name"]})
    return {
        "bytes": len(data),
        "sizeKB": round(len(data) / 1024, 1),
        "sha256": hashlib.sha256(data).hexdigest(),
        "vertices": verts, "triangles": tris, "meshes": meshes,
        "textures": [im.get("name") for im in j.get("images", [])],
        "animations": [a["name"] for a in j.get("animations", [])],
        "joints": len(j["skins"][0]["joints"]) if j.get("skins") else 0,
    }


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, "/")


def main():
    man = {
        "name": "DINUMED 3D avatar asset package",
        "version": "1.0.0",
        "generator": "source/generator/build.py (procedural, deterministic)",
        "license": "CC0-1.0 (assets) - see LICENSE.md; three.js MIT - see vendor/three/LICENSE",
        "threeVersion": "0.187.0",
        "units": "metres",
        "coordinateSystem": "glTF 2.0: +Y up, characters face +Z, character left = +X",
        "bindPose": "A-pose (arms 55 degrees below horizontal); all bone rest rotations are identity",
        "skeleton": {"boneCount": len(bone_names()), "jointOrder": bone_names(),
                     "naming": "Mixamo-style names without prefix; identical joint order and names in every GLB"},
        "characters": [], "clothing": [], "animations": [], "textures": [],
    }
    for g in ("male", "female"):
        p = os.path.join(A, "characters", f"{g}.glb")
        fit = json.load(open(os.path.join(A, "clothing", g, "_fit.json")))
        e = {"id": g, "file": rel(p), **glb_info(p), "shoeGroundOffset": fit["shoeGroundOffset"],
             "groundOffset": fit["groundOffset"], "heightMetres": fit["heightMetres"]}
        man["characters"].append(e)
    for iid, title, slot in ITEMS:
        e = {"id": iid, "title": title, "slot": slot}
        variants = None
        if iid in ("stethoscope", "id_card", "backpack"):
            variants = {}
            for v, desc in (("over_coat", "fitted to rest on the white coat"),
                            ("over_scrubs", "fitted to rest on the scrubs top (no coat)")):
                files, info = {}, {}
                for g in ("male", "female"):
                    p = os.path.join(A, "clothing", g, f"{iid}_{v}.glb")
                    files[g] = rel(p)
                    info[g] = glb_info(p)
                variants[v] = {"description": desc, "files": files, "details": info}
            e["variants"] = variants
            e["variantRule"] = "use over_coat when white_coat is worn, otherwise over_scrubs"
        else:
            files, info = {}, {}
            for g in ("male", "female"):
                p = os.path.join(A, "clothing", g, f"{iid}.glb")
                files[g] = rel(p)
                info[g] = glb_info(p)
            e["files"], e["details"] = files, info
        man["clothing"].append(e)
    p = os.path.join(A, "animations", "idle.glb")
    man["animations"].append({"id": "idle", "file": rel(p), **glb_info(p),
                              "durationSeconds": 8.0, "loop": True,
                              "note": "rotation-only tracks; drives both male and female skeletons; also embedded in each character GLB"})
    for f in sorted(os.listdir(os.path.join(A, "textures"))):
        p = os.path.join(A, "textures", f)
        data = open(p, "rb").read()
        man["textures"].append({"file": rel(p), "bytes": len(data), "sizeKB": round(len(data) / 1024, 1),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "note": "embedded in the GLB files that use it; provided separately for editing"})
    total = 0
    for root, _, fs in os.walk(A):
        for f in fs:
            if f == "manifest.json":
                continue
            total += os.path.getsize(os.path.join(root, f))
    man["totalAssetBytes"] = total
    json.dump(man, open(os.path.join(A, "manifest.json"), "w"), indent=2)
    print("manifest written:", rel(os.path.join(A, "manifest.json")))


if __name__ == "__main__":
    main()
