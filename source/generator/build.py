"""DINUMED asset build entry point.

Usage:  python3 build.py [--only male|female] [--fast]
Writes GLB files, textures and the manifest into <repo>/assets.
"""
import argparse
import json
import os
import sys
import time
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from body import Body, bone_names, bone_parent
from sdf import (Union, Intersect, Offset, Complement, HalfSpace, Func, Ellipsoid,
                 Sphere, RoundCone, Displace, project, rot_from_axes)
from mesh import extract, uv_sphere, merge, Mesh, DensityWarp, tube
from skinning import body_weights, transfer, dense, rigid, NAMES
from gltf_writer import GLB
import textures as tx
import animation
import garments

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "assets")
COPYRIGHT = "DINUMED avatar asset package - procedurally generated, CC0-1.0"


def lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgba(srgb, a=1.0):
    return list(lin(srgb)) + [a]


SKIN = {"male": (232, 182, 148), "female": (238, 192, 166)}
LIPS = {"male": (176, 92, 88), "female": (196, 92, 104)}
HAIR = {"male": (92, 58, 36), "female": (74, 44, 28)}
IRIS = {"male": (0.40, 0.25, 0.13), "female": (0.30, 0.42, 0.26)}


def log(*a):
    print(*a, flush=True)


class Context:
    """Everything a garment needs to know about a character."""
    pass


def body_colors(b, P, gender):
    C = np.tile(np.append(lin(SKIN[gender]), 1.0), (len(P), 1))
    # lips
    dl = b.lip_prims[0](P)                      # soft lower lip tint
    t = np.clip(1 - (dl + 0.0010) / 0.0040, 0, 1)[:, None] * 0.55
    C[:, :3] = C[:, :3] * (1 - t) + lin(LIPS[gender]) * t
    # rosy cartoon cheeks
    for s in (1, -1):
        c = b.Hpt(0.047 * s, -0.030, 0.062)
        d = np.linalg.norm(P - c, axis=1)
        t = np.clip(1 - d / (0.026 * b.hs), 0, 1)[:, None] ** 1.5 * (0.30 if gender == "female" else 0.20)
        C[:, :3] = C[:, :3] * (1 - t) + lin((200, 110, 105)) * t
    # slightly darker soles / palms are lighter: palms
    return C


def build_character(gender, fast=False, only=None, reuse_body=False):
    log(f"== character: {gender}")
    b = Body(gender)
    h = 0.0032 if fast else 0.0021
    head_only = bool(os.environ.get("DINUMED_HEAD_ONLY"))
    Yn = b.joints["Neck"][1]
    warp = DensityWarp(Yn + 0.02, Yn + 0.10, 2.6)
    cache = os.path.join(REPO, ".cache", f"body_{gender}.npz")
    reuse = (bool(only) or reuse_body) and os.path.exists(cache)
    skip_char = bool(only) and not reuse_body
    if reuse:
        z = np.load(cache)
        bodyM = Mesh(z["P"], z["F"], z["N"])
    elif head_only:     # quick preview of the head (developer option)
        bodyM = extract(b.field, (-0.3, Yn - 0.08, -0.2), (0.3, b.height + 0.01, 0.27), 0.0021, 30000, log=log)
    else:
        bodyM = extract(b.field, (-0.76, -0.01, -0.2), (0.76, b.height + 0.01, 0.27), h,
                        22000 if fast else 62000, warp=warp, log=log)
    bodyM.C = body_colors(b, bodyM.P, gender)
    if reuse:
        J, Wt = z["J"], z["W"]
    else:
        J, Wt = body_weights(b, bodyM.P, bodyM.F)
        if not head_only and not fast:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, P=bodyM.P, F=bodyM.F, N=bodyM.N, J=J, W=Wt)
    Wd = dense(J, Wt)
    ctx = Context()
    ctx.gender, ctx.body, ctx.mesh, ctx.Wd, ctx.fast, ctx.only = gender, b, bodyM, Wd, fast, only

    # ---------------------------------------------------------- hair & brows
    from garments import ray_surface
    hs, H0 = b.hs, b.head_c
    cc, cr = b.cran_c, b.cran_r
    cran = Ellipsoid(cc, cr)

    def lock(z_rel, phi0, phi1, lift, fwd, r0, n=9):
        """One hair lock: an arc over the scalp from angle phi0 to phi1 (degrees
        from the vertical, +x side positive) at depth z_rel, lifted off the scalp."""
        prims, pts, rads = [], [], []
        for t in np.linspace(0, 1, n):
            phi = np.radians(phi0 + (phi1 - phi0) * t)
            z = cc[2] + z_rel * hs + fwd * hs * np.sin(np.pi * t) ** 1.2
            sc = np.sqrt(max(1 - ((z_rel * hs) / cr[2]) ** 2, 0.06))
            p = np.array([cr[0] * sc * np.sin(phi), cc[1] + cr[1] * sc * np.cos(phi), z])
            nrm = np.array([p[0] / cr[0] ** 2, (p[1] - cc[1]) / cr[1] ** 2, (cc[2] + z_rel * hs - cc[2]) / cr[2] ** 2])
            nrm /= np.linalg.norm(nrm)
            pts.append(p + nrm * hs * (0.002 + lift * np.sin(np.pi * min(t * 1.1, 1.0)) ** 0.8))
            rads.append(r0 * hs * (0.50 + 0.50 * np.sin(np.pi * (0.12 + 0.80 * t))))
        for i in range(n - 1):
            prims.append(RoundCone(pts[i], pts[i + 1], rads[i], rads[i + 1]))
        return Union(prims, k=0.002 * hs)

    locks = []
    if gender == "male":
        F_, S_, B_ = 0.060, 0.020, -0.062
        base_cap = Offset(cran, 0.0055 * hs / 1.36)
        # side part on the character's right; locks sweep over the top to the left
        for i, z in enumerate(np.linspace(0.066, -0.074, 11)):
            front = np.clip((z + 0.005) / 0.07, 0, 1)
            locks.append(lock(z, -36 + 4 * (i % 2), 58 - 5 * (i % 3), 0.009 + 0.015 * front, 0.014 * front, 0.0150 - 0.0015 * (i % 2)))
        for i, z in enumerate(np.linspace(0.050, -0.070, 6)):          # short side below the part
            locks.append(lock(z, -40, -60, 0.004, 0.004 * (z > 0), 0.010))
        for i, phi in enumerate(np.linspace(-60, 60, 6)):               # crown / back
            locks.append(lock(-0.080, phi - 14, phi + 14, 0.002, -0.004, 0.012))
    else:
        F_, S_, B_ = 0.058, 0.016, -0.072
        base_cap = Offset(cran, 0.0055 * hs / 1.30)
        for i, z in enumerate(np.linspace(0.064, -0.074, 11)):
            front = np.clip((z + 0.005) / 0.07, 0, 1)
            locks.append(lock(z, -24 + 3 * (i % 2), 60, 0.010 + 0.010 * front, 0.012 * front, 0.0160 - 0.0015 * (i % 2)))
            locks.append(lock(z, -28, -60, 0.008 + 0.004 * front, 0.006 * front, 0.0140))
        for i, phi in enumerate(np.linspace(-70, 70, 7)):
            locks.append(lock(-0.082, phi - 14, phi + 14, 0.003, -0.005, 0.014))
        locks.append(Sphere(b.Hpt(0, 0.108, -0.052), 0.044 * hs))                    # high bun (top knot)
        locks.append(RoundCone(b.Hpt(0, 0.080, -0.035), b.Hpt(0, 0.100, -0.048), 0.030 * hs, 0.034 * hs))
    base_cap = Union([base_cap, Offset(b.field, 0.0038)], k=0.004)      # never dips into the neck at the nape
    hair_core = Union([base_cap, Union(locks, k=0.0105 * hs)], k=0.010 * hs)
    a_ = (F_ + B_) / 4 + S_ / 2
    c_ = (F_ + B_) / 4 - S_ / 2
    coef = (a_, F_ - a_ - c_, c_)

    def hairline(P):
        q = P - H0
        th = np.arctan2(q[:, 0], q[:, 2])
        a, bb, c = coef
        yh = (a + bb * np.cos(th) + c * np.cos(2 * th)) * hs
        if gender == "female":
            yh = yh - 0.016 * np.exp(-((th - 0.55) / 0.40) ** 2) * hs  # side-swept fringe
        else:
            yh = yh - 0.008 * np.exp(-((th - 0.45) / 0.35) ** 2) * hs  # fringe falls to the left
        return yh - q[:, 1]
    region = Func(hairline, name="hairline")
    ear_zone = Union([Offset(e, 0.016) for e in b.ear_prims])
    noise = lambda P: 0.00045 * np.sin(P[:, 2] * 700 + 3.0 * np.sin(P[:, 0] * 45)) * hs      # fine strands along the sweep
    hair_shell = Displace(hair_core, noise, 0.0008 * hs, name="hair")
    hair_reg = Intersect([region, Complement(ear_zone)], k=0.002)
    hair_field = Intersect([hair_shell, hair_reg], name="hair")
    lo, hi = H0 - 0.23, H0 + np.array([0.23, 0.27, 0.23])
    if skip_char:       # garments only need the hair field, not its mesh
        ctx.hair_shell, ctx.hair_field = hair_shell, hair_field
        return ctx
    hairM = extract(hair_shell, lo, hi, 0.0018, 6000 if fast else 16000, clip=hair_reg, log=log)

    # eyebrows + eyeliner: thin layer just above the skin
    brows = []
    bw = 1.0 if gender == "male" else 0.72
    for s in (1, -1):
        pts = [ray_surface(b.head_only, b.Hpt(x * s, y, 0.2), (0, 0, -1)) for x, y in
               ((0.014, 0.031), (0.034, 0.0375), (0.055, 0.031))]
        brows += [RoundCone(pts[0], pts[1], 0.0048 * hs * bw, 0.0046 * hs * bw),
                  RoundCone(pts[1], pts[2], 0.0046 * hs * bw, 0.0020 * hs)]
        c = b.eye_c[0 if s > 0 else 1]
        ox, oy = b.eye_open[0], b.eye_open[1]
        lid = []
        R_ = b.lid_r
        for t in np.linspace(-0.86, 0.90, 9):         # points on the upper lid, just above the opening
            dx = s * t * ox * 0.98 + 0.0006 * s
            dy = 0.0012 * hs + oy * np.sqrt(1 - t * t) * 1.04
            lid.append(c + np.array([dx, dy, np.sqrt(max(R_ * R_ - dx * dx - dy * dy, 1e-8))]))
        rl = (0.0016 if gender == "male" else 0.0022) * hs
        if gender == "female":                  # lashes line along the upper lid
            for p0, p1 in zip(lid[:-1], lid[1:]):
                brows.append(RoundCone(p0, p1, rl, rl))
        if gender == "female":                  # small outer flick
            brows.append(RoundCone(lid[-1], lid[-1] + np.array([0.004 * s, 0.002, -0.002]) * hs, rl, rl * 0.4))
    brow_off = Offset(b.head_only, 0.0009, name="brows")
    browM = extract(brow_off, b.Hpt(-0.075, -0.02, 0.02), b.Hpt(0.075, 0.06, 0.12), 0.0006,
                    2400, clip=Union(brows, k=0.002), min_frac=0.02, log=log)
    hair_all = merge([hairM, browM])

    # smiling mouth: dark crescent 0.5 mm above the skin with a row of teeth on top of it
    mo = b.mouth
    def crescent(xe, yc, sag, r_mid, r_end, n=11):
        prims, pts, rads = [], [], []
        for x in np.linspace(-xe, xe, n):
            u = 1 - (x / xe) ** 2
            p = ray_surface(b.head_only, b.Hpt(x, yc - sag * u, 0.2), (0, 0, -1))
            pts.append(p); rads.append((r_end + r_mid * u) * hs)
        for i in range(n - 1):
            prims.append(RoundCone(pts[i], pts[i + 1], rads[i], rads[i + 1]))
        return Union(prims)
    mouth_reg = crescent(mo["xe"], mo["yc"], mo["sag"], mo["r_mid"], mo["r_end"])
    teeth_reg = crescent(mo["xe"] * 0.80, mo["yc"] - 0.0012, 0.0028, 0.0026, 0.0003)
    mouthM = extract(Offset(b.head_only, 0.0005, name="mouth"), b.Hpt(-0.04, -0.075, 0.03), b.Hpt(0.04, -0.025, 0.12),
                     0.0005, 900, clip=mouth_reg, min_frac=0.05, log=log)
    teethM = extract(Offset(b.head_only, 0.0010, name="teeth"), b.Hpt(-0.04, -0.075, 0.03), b.Hpt(0.04, -0.025, 0.12),
                     0.0005, 600, clip=teeth_reg, min_frac=0.05, log=log)

    # ---------------------------------------------------------------- eyes
    eyes = merge([uv_sphere(c, b.eye_r, 24, 16, axis=(0, 0, 1)) for c in b.eye_c])

    # ----------------------------------------------------------- underwear
    uw = None if head_only else garments.underwear(ctx, log)

    ctx.hair_mesh = hair_all
    ctx.hair_shell = hair_shell
    ctx.hair_field = hair_field

    # --------------------------------------------------------------- write
    glb = GLB(copyright=COPYRIGHT)
    names = bone_names()
    parents = {n: bone_parent(n) for n in names}
    arm, idx = glb.skeleton(names, parents, b.joints)
    skin = glb.skin(names, idx, b.joints, name=f"{gender}_skin")
    m_skin = glb.material("Skin", [1, 1, 1, 1], roughness=0.55)
    m_hair = glb.material("Hair", rgba(HAIR[gender]), roughness=0.75, double_sided=True)
    iris_png = tx.png_bytes(tx.iris(IRIS[gender], seed=3 if gender == "male" else 5))
    tex = glb.image(iris_png, f"iris_{gender}")
    m_eye = glb.material("Eye", [1, 1, 1, 1], roughness=0.15, texture=tex)
    m_uw = glb.material("Underwear", rgba((58, 62, 72)), roughness=0.8, double_sided=True)
    glb.skinned_mesh("Body", bodyM, J, Wt, m_skin, skin)
    glb.skinned_mesh("Eyes", eyes, *rigid(len(eyes.P), "Head"), m_eye, skin)
    glb.skinned_mesh("Hair", hair_all, *rigid(len(hair_all.P), "Head"), m_hair, skin)
    m_mouth = glb.material("Mouth", rgba((112, 36, 42)), roughness=0.5)
    glb.skinned_mesh("Mouth", mouthM, *rigid(len(mouthM.P), "Head"), m_mouth, skin)
    m_teeth = glb.material("Teeth", rgba((250, 250, 246)), roughness=0.25)
    glb.skinned_mesh("Teeth", teethM, *rigid(len(teethM.P), "Head"), m_teeth, skin)
    if uw is not None:
        Juw, Wuw = transfer(bodyM.P, Wd, uw.P)
        glb.skinned_mesh("Underwear", uw, Juw, Wuw, m_uw, skin)
    t, tracks = animation.idle()
    glb.animation("Idle", t, tracks, idx)
    path = os.path.join(OUT, "characters", f"{gender}.glb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = glb.write(path)
    tex_path = os.path.join(OUT, "textures", f"iris_{gender}.png")
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    open(tex_path, "wb").write(iris_png)
    log(f"  wrote {path} ({size / 1024:.0f} KB)")
    ctx.underwear = uw
    return ctx


def write_idle_animation(body):
    glb = GLB(copyright=COPYRIGHT)
    names = bone_names()
    parents = {n: bone_parent(n) for n in names}
    arm, idx = glb.skeleton(names, parents, body.joints)
    t, tracks = animation.idle()
    glb.animation("Idle", t, tracks, idx)
    path = os.path.join(OUT, "animations", "idle.glb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    glb.write(path)
    log(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["male", "female"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-garments", action="store_true")
    ap.add_argument("--items", help="comma separated clothing ids to rebuild (reuses the cached body mesh)")
    ap.add_argument("--reuse-body", action="store_true", help="reuse the cached body mesh but rewrite the character (hair, face, animation)")
    a = ap.parse_args()
    t0 = time.time()
    genders = [a.only] if a.only else ["male", "female"]
    for g in genders:
        only = set(a.items.split(",")) if a.items else None
        ctx = build_character(g, a.fast, only, a.reuse_body)
        if not a.no_garments:
            garments.build_all(ctx, OUT, COPYRIGHT, log)
        if g == "male":
            write_idle_animation(ctx.body)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
