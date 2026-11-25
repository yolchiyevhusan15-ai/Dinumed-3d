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
                 Sphere, RoundCone, Displace, project)
from mesh import extract, uv_sphere, merge, Mesh
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


SKIN = {"male": (196, 148, 118), "female": (226, 182, 158)}
LIPS = {"male": (168, 100, 92), "female": (180, 96, 100)}
HAIR = {"male": (38, 28, 22), "female": (64, 40, 26)}
IRIS = {"male": (0.36, 0.22, 0.12), "female": (0.25, 0.33, 0.22)}


def log(*a):
    print(*a, flush=True)


class Context:
    """Everything a garment needs to know about a character."""
    pass


def body_colors(b, P, gender):
    C = np.tile(np.append(lin(SKIN[gender]), 1.0), (len(P), 1))
    # lips
    dl = np.minimum(b.lip_prims[0](P), b.lip_prims[1](P))
    t = np.clip(1 - (dl + 0.0005) / 0.0025, 0, 1)[:, None]
    C[:, :3] = C[:, :3] * (1 - t) + lin(LIPS[gender]) * t
    # cheeks blush (subtle)
    for s in (1, -1):
        c = b.Hpt(0.042 * s, -0.022, 0.08)
        d = np.linalg.norm(P - c, axis=1)
        t = np.clip(1 - d / 0.03, 0, 1)[:, None] * (0.10 if gender == "female" else 0.05)
        C[:, :3] = C[:, :3] * (1 - t) + lin((200, 110, 105)) * t
    # slightly darker soles / palms are lighter: palms
    return C


def build_character(gender, fast=False):
    log(f"== character: {gender}")
    b = Body(gender)
    h = 0.0035 if fast else 0.0026
    bodyM = extract(b.field, (-0.76, -0.01, -0.2), (0.76, b.height + 0.01, 0.27), h,
                    18000 if fast else 46000, log=log)
    bodyM.C = body_colors(b, bodyM.P, gender)
    J, Wt = body_weights(b, bodyM.P, bodyM.F)
    Wd = dense(J, Wt)
    ctx = Context()
    ctx.gender, ctx.body, ctx.mesh, ctx.Wd, ctx.fast = gender, b, bodyM, Wd, fast

    # ---------------------------------------------------------- hair & brows
    hs, H0 = b.hs, b.head_c
    cran = Ellipsoid(b.Hpt(0, 0.010, -0.012), np.array([0.077, 0.097, 0.099]) * hs)
    if gender == "male":
        coef = (0.01375, 0.0675, -0.01125)
        hair_core = Offset(cran, lambda P: 0.006 + 0.007 * np.clip((P[:, 1] - H0[1] - 0.04) / 0.06, 0, 1) * np.clip(0.5 + (P[:, 2] - H0[2]) / 0.12, 0, 1), omax=0.014)
    else:
        coef = (0.00925, 0.0715, -0.01275)
        bun = Sphere(b.Hpt(0, -0.012, -0.118), 0.040 * hs)
        back = Ellipsoid(b.Hpt(0, -0.02, -0.035), np.array([0.083, 0.085, 0.088]) * hs)
        hair_core = Union([Offset(cran, 0.012), Offset(back, 0.006), bun], k=0.02)

    def hairline(P):
        q = P - H0
        th = np.arctan2(q[:, 0], q[:, 2])
        a, bb, c = coef
        yh = (a + bb * np.cos(th) + c * np.cos(2 * th)) * hs
        if gender == "female":
            yh = yh - 0.012 * np.exp(-((th - 0.5) / 0.35) ** 2) * hs  # side part sweep
        return yh - q[:, 1]
    region = Func(hairline, name="hairline")
    ear_zone = Union([Offset(e, 0.013) for e in b.ear_prims])
    noise = lambda P: 0.0006 * np.sin(P[:, 0] * 820 + 2.0 * np.sin(P[:, 2] * 45)) + 0.0004 * np.sin(P[:, 1] * 300 - P[:, 2] * 140)
    hair_shell = Displace(hair_core, noise, 0.0011, name="hair")
    hair_reg = Intersect([region, Complement(ear_zone)], k=0.002)
    hair_field = Intersect([hair_shell, hair_reg], name="hair")
    lo, hi = H0 - 0.16, H0 + 0.16
    hairM = extract(hair_shell, lo, hi, 0.0018, 4000 if fast else 9000, clip=hair_reg, log=log)
    # eyebrows: thin layer 1 mm above the skin inside a tapered band located on the brow ridge
    from garments import ray_surface
    brows = []
    for s in (1, -1):
        pts = [ray_surface(b.head_only, b.Hpt(x * s, y, 0.2), (0, 0, -1)) for x, y in
               ((0.012, 0.019), (0.030, 0.0235), (0.050, 0.019))]
        brows += [RoundCone(pts[0], pts[1], 0.0040 * hs, 0.0036 * hs), RoundCone(pts[1], pts[2], 0.0036 * hs, 0.0018 * hs)]
    brow_off = Offset(b.head_only, 0.001, name="brows")
    browM = extract(brow_off, b.Hpt(-0.07, -0.01, 0.04), b.Hpt(0.07, 0.05, 0.13), 0.0007,
                    800, clip=Union(brows, k=0.003), min_frac=0.3, log=log)
    hair_all = merge([hairM, browM])

    # ---------------------------------------------------------------- eyes
    eyes = merge([uv_sphere(c, b.eye_r, 24, 16, axis=(0, 0, 1)) for c in b.eye_c])

    # ----------------------------------------------------------- underwear
    uw = garments.underwear(ctx, log)

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
    a = ap.parse_args()
    t0 = time.time()
    genders = [a.only] if a.only else ["male", "female"]
    for g in genders:
        ctx = build_character(g, a.fast)
        if not a.no_garments:
            garments.build_all(ctx, OUT, COPYRIGHT, log)
        if g == "male":
            write_idle_animation(ctx.body)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
