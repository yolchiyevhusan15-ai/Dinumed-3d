"""Clothing and accessories, fitted to a specific character body.

All garments are SDF offset surfaces of the character body (so they cannot
intersect it in the bind pose), skinned by transferring weights from the
character's body mesh, and exported with an identical skeleton.
"""
import os
import json
import numpy as np

from sdf import (Union, Intersect, Offset, Complement, HalfSpace, Func, Ellipsoid,
                 Sphere, RoundCone, RoundBox, gradient, normals, project, rot_from_axes)
from mesh import extract, merge, Mesh, tube, box, cylinder, face_normals, vertex_normals
from skinning import transfer, rigid, top4, dense, laplacian_smooth
from body import bone_names, bone_parent
from gltf_writer import GLB
import textures as tx


def lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgba(srgb, a=1.0):
    return list(lin(srgb)) + [a]


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ helpers
def _arm_dirs(b):
    dl = b.arm_dir.copy()
    dr = dl.copy(); dr[0] = -dr[0]
    return {"Left": dl, "Right": dr}


def _mir(p, side):
    p = np.array(p, float)
    if side == "Right":
        p[0] = -p[0]
    return p


def arm_cut(b, dist_from_wrist=None, frac_upper=None, radius=0.10):
    """Cut-away solids removing each arm beyond a plane perpendicular to it."""
    cuts = []
    for side, d in _arm_dirs(b).items():
        S, W = _mir(b.S, side), _mir(b.W, side)
        if frac_upper is not None:
            c = S + d * frac_upper * np.linalg.norm(b.E - b.S)
        else:
            c = W - d * dist_from_wrist
        hs = HalfSpace(-d, -(c @ d))
        cap = RoundCone(c - d * 0.03, W + d * 0.25, radius, radius)
        cuts.append(Intersect([hs, cap]))
    return Union(cuts)


def neck_cut(b, r, y0, zc=-0.02):
    Y = lambda v: v * b.sy
    return RoundCone((0, Y(y0), zc), (0, Y(y0) + 0.6, zc), r, r)


def v_cut(b, y_v, slope, z0=0.0, xmax=0.075):
    """V-neck opening; its half-width is capped at xmax so that it meets the
    neck hole as a slot instead of opening the shoulders."""
    def f(P):
        a = (y_v - P[:, 1] + slope * np.abs(P[:, 0])) / np.sqrt(1 + slope * slope)
        return np.maximum(np.maximum(a, z0 - P[:, 2]), np.abs(P[:, 0]) - xmax)
    return Func(f, lo=(-0.4, y_v - 0.01, z0 - 0.01), hi=(0.4, y_v + 0.6, 0.4))


def chest_filler(b):
    if not b.F:
        return None
    Y = lambda v: v * b.sy
    return Ellipsoid((0, Y(1.275), 0.068), (0.105, 0.052, 0.046))


def scrub_offset(b):
    Y = lambda v: v * b.sy
    def o(P):
        loose = 0.011 * smoothstep(Y(1.24) - P[:, 1], 0.0, 0.30)
        return 0.011 + loose
    return o


def coat_offset(b):
    so = scrub_offset(b)
    def o(P):
        torso = so(P) + (0.016 if not b.F else 0.013)
        armw = smoothstep(np.abs(P[:, 0]), 0.20, 0.27)
        # do not loosen the skirt (it has its own shape)
        armw = armw * smoothstep(P[:, 1], 0.95 * b.sy, 1.05 * b.sy)
        return torso * (1 - armw) + (0.021 if not b.F else 0.018) * armw
    return o


def trouser_offset(b):
    Y = lambda v: v * b.sy
    def o(P):
        return 0.0055 + 0.012 * np.clip((Y(0.95) - P[:, 1]) / (Y(0.95) - 0.1), 0, 1)
    return o


def skirt(b):
    """Coat skirt envelope: elliptic cross-sections interpolated from waist
    (inside the body) through the hips down to the hem, so the coat hangs
    straight over both legs instead of following them."""
    Y = lambda v: v * b.sy
    ys = np.array([Y(1.06), Y(0.93), Y(0.40)])
    rxs = np.array([0.095, 0.150, 0.198]) if not b.F else np.array([0.085, 0.165, 0.210])
    rzs = np.array([0.070, 0.104, 0.138]) if not b.F else np.array([0.068, 0.110, 0.142])
    def f(P):
        y = np.clip(P[:, 1], ys[-1], ys[0])
        rx = np.interp(-y, -ys, rxs)
        rz = np.interp(-y, -ys, rzs)
        zc = -0.018
        q = np.sqrt((P[:, 0] / rx) ** 2 + ((P[:, 2] - zc) / rz) ** 2)
        d = (q - 1) * np.minimum(rx, rz)
        return np.maximum(d, P[:, 1] - ys[0])
    return Func(f, lo=(-0.3, ys[-1] - 0.05, -0.25), hi=(0.3, ys[0] + 0.01, 0.25), name="skirt")


def ray_surface(field, origin, direction, target=0.0, tmax=0.8):
    """March from origin along direction until field <= target; returns point."""
    origin = np.asarray(origin, float); direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    # make sure the ray starts outside the solid
    for _ in range(300):
        if field(origin[None])[0] - target > 0:
            break
        origin = origin - direction * 0.002
    ts = np.linspace(0, tmax, 1200)
    P = origin + ts[:, None] * direction
    v = field(P) - target
    i = np.argmax(v <= 0)
    if v[i] > 0:
        raise RuntimeError("ray missed")
    lo, hi = ts[max(i - 1, 0)], ts[i]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if field((origin + mid * direction)[None])[0] - target > 0:
            lo = mid
        else:
            hi = mid
    return origin + hi * direction


def push_out(field, P, clearance, iters=8):
    P = np.array(P, float)
    for _ in range(iters):
        f = field(P)
        bad = f < clearance
        if not bad.any():
            break
        g = normals(field, P[bad])
        P[bad] += g * (clearance - f[bad])[:, None] * 1.05
    return P


def onto(field, P, target, iters=6):
    P = np.array(P, float)
    for _ in range(iters):
        f = field(P)
        g = normals(field, P)
        P -= g * (f - target)[:, None]
    return P


def smooth_path(P, iters=10, lam=0.5, fixed=()):
    P = np.array(P, float)
    for _ in range(iters):
        Q = P.copy()
        Q[1:-1] = (1 - lam) * P[1:-1] + lam * 0.5 * (P[:-2] + P[2:])
        for i in fixed:
            Q[i] = P[i]
        P = Q
    return P


def resample(P, step):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / step), 2)
    t = np.linspace(0, s[-1], n + 1)
    return np.stack([np.interp(t, s, P[:, i]) for i in range(3)], 1)


def catmull(points, n_per=12):
    P = np.asarray(points, float)
    P = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


# --------------------------------------------------------------- garments
def underwear(ctx, log):
    b = ctx.body
    Y = lambda v: v * b.sy
    o = 0.0028
    base_lo = Offset(b.trunk_legs, o)
    base_lo.name = "underwear_bottom"
    reg = Intersect([HalfSpace((0, 1, 0), Y(1.0)), HalfSpace((0, -1, 0), -Y(0.79))])
    parts = [extract(base_lo, (-0.3, Y(0.77), -0.2), (0.3, Y(1.02), 0.2), 0.0024,
                     3000 if ctx.fast else 6000, clip=reg, log=log)]
    if b.F:
        up = Union([b.trunk_arms, chest_filler(b)], k=0.02)
        base_up = Offset(up, o)
        cut = Union([RoundCone(_mir(b.S, s) - _arm_dirs(b)[s] * 0.01, _mir(b.W, s), 0.075, 0.06) for s in ("Left", "Right")])
        base_up.name = "underwear_top"
        reg = Intersect([HalfSpace((0, 1, 0), Y(1.345)), HalfSpace((0, -1, 0), -Y(1.185)), Complement(cut)])
        parts.append(extract(base_up, (-0.3, Y(1.17), -0.2), (0.3, Y(1.36), 0.2), 0.0024,
                             2500 if ctx.fast else 5000, clip=reg, log=log))
    return merge(parts)


class Garment:
    def __init__(self, name, title, layer, meshes, notes=""):
        self.name, self.title, self.layer, self.meshes, self.notes = name, title, layer, meshes, notes


def scrubs_top(ctx, log):
    b = ctx.body
    Y = lambda v: v * b.sy
    base = Union([b.no_head, chest_filler(b)], k=0.02)
    off = Offset(base, scrub_offset(b), omax=0.03)
    cuts = Union([arm_cut(b, frac_upper=0.48, radius=0.085), neck_cut(b, 0.084, 1.40),
                  v_cut(b, Y(1.335) if not b.F else Y(1.32), 1.7, z0=-0.01, xmax=0.068)])
    reg = Intersect([HalfSpace((0, -1, 0), -Y(0.86)), Complement(cuts)])
    field = Intersect([off, reg], name="scrubs_top")
    off.name = "scrubs_top"
    m = extract(off, (-0.5, Y(0.84), -0.22), (0.5, Y(1.58), 0.24), 0.0028,
                6000 if ctx.fast else 14000, clip=reg, log=log)
    ctx.scrub_solid = field
    return m, field


def trousers(ctx, log):
    b = ctx.body
    Y = lambda v: v * b.sy
    off = Offset(b.trunk_shanks, trouser_offset(b), omax=0.03)
    reg = Intersect([HalfSpace((0, 1, 0), Y(1.005)), HalfSpace((0, -1, 0), -0.080)])
    field = Intersect([off, reg], name="trousers")
    off.name = "trousers"
    m = extract(off, (-0.32, 0.07, -0.22), (0.32, Y(1.02), 0.22), 0.0028,
                6000 if ctx.fast else 14000, clip=reg, log=log)
    return m, field


def white_coat(ctx, log):
    b = ctx.body
    Y = lambda v: v * b.sy
    base = Union([b.no_head, chest_filler(b), skirt(b)], k=0.04)
    off = Offset(base, coat_offset(b), omax=0.05)
    y_v = Y(1.165) if not b.F else Y(1.15)
    slope = 3.0
    neck_r, neck_y, neck_z = 0.090, 1.43, -0.02
    # lapels + collar: raised strip that follows the V edges and the back neckline
    strip = []
    for s_ in (1, -1):
        for t in np.linspace(0.0, 1.0, 12):
            y = y_v + t * (Y(1.47) - y_v)
            x = s_ * (min((y - y_v) / slope, 0.078) + 0.016)
            sp = ray_surface(off, (x, y, 0.4), (0, 0, -1))
            strip.append(Sphere(sp, 0.021))
    for th in np.radians(np.linspace(50, 310, 15)):
        x = np.sin(th) * (neck_r + 0.014)
        z = neck_z + np.cos(th) * (neck_r + 0.014)
        sp = ray_surface(off, (x, Y(1.8), z), (0, -1, 0))
        strip.append(Sphere(sp, 0.020))
    lapel = Intersect([Offset(off, 0.0032), Union(strip, k=0.012)], k=0.004)
    pockets = []
    for (x, y, hw, hh) in ((0.115, Y(0.80), 0.066, 0.075), (-0.115, Y(0.80), 0.066, 0.075),
                           (0.100, Y(1.255) if not b.F else Y(1.33), 0.046, 0.052)):
        sp = ray_surface(off, (x, y, 0.4), (0, 0, -1))
        n = normals(off, sp[None])[0]
        R = rot_from_axes(z=n, x=np.cross((0, 1, 0), n))
        pockets.append(Intersect([Offset(off, 0.0026), RoundBox(sp, (hw, hh, 0.04), 0.008, R=R)], k=0.0035))
    solid = Union([off, lapel] + pockets, k=0.0025, name="white_coat")
    cuts = Union([arm_cut(b, dist_from_wrist=0.062, radius=0.11), neck_cut(b, neck_r, neck_y, neck_z),
                  v_cut(b, y_v, slope, z0=-0.02, xmax=0.078)])
    hem = Y(0.47)
    reg = Intersect([HalfSpace((0, -1, 0), -hem), Complement(cuts)])
    field = Intersect([solid, reg], name="white_coat")
    m = extract(solid, (-0.66, hem - 0.02, -0.26), (0.66, Y(1.60), 0.28), 0.0028,
                9000 if ctx.fast else 24000, clip=reg, log=log)
    ctx.coat_solid = field
    # buttons on the centre front below the V
    btn = []
    for y in np.linspace(y_v - 0.05, hem + 0.12, 4):
        sp = ray_surface(off, (0.0, y, 0.4), (0, 0, -1))
        n = normals(off, sp[None])[0]
        btn.append(cylinder(sp + n * 0.0016, n, 0.0085, 0.0032, seg=20))
    return m, merge(btn), field


def shoes(ctx, log):
    b = ctx.body
    ytop = 0.094
    feet = b.feet_field
    def sole_f(P):
        Q = P.copy(); Q[:, 1] = 0.012
        foot2d = feet(Q) - 0.009            # footprint grown by 9 mm
        slab = np.abs(P[:, 1] + 0.0035) - 0.0125
        r = 0.005                           # rounded sole edges
        a, b_ = foot2d + r, slab + r
        return np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(b_, 0) ** 2) + np.minimum(np.maximum(a, b_), 0) - r
    sole = Func(sole_f, lo=(-0.2, -0.02, -0.12), hi=(0.2, 0.01, 0.24), name="sole")
    upper = Offset(feet, 0.0055)
    base = Union([upper, sole], k=0.008, name="shoes")
    reg = HalfSpace((0, 1, 0), ytop)
    field = Intersect([base, reg], name="shoes")
    m = extract(base, (-0.2, -0.03, -0.12), (0.2, ytop + 0.01, 0.24), 0.0022,
                5000 if ctx.fast else 12000, clip=reg, log=log)
    ctx.sole_bottom = float(m.P[:, 1].min())
    return m, field


def medical_cap(ctx, log):
    b = ctx.body
    hs, H0 = b.hs, b.head_c
    base = Union([b.head_only, ctx.hair_field], k=0.006)
    o = 0.0032
    off = Offset(base, o, name="medical_cap")
    # cap edge height (relative to head centre) by azimuth: forehead, above ears, behind ears, nape
    ang = np.radians([0, 90, 128, 180])
    hgt = np.array([0.060, 0.036, 0.033, -0.050]) if not b.F else np.array([0.058, 0.036, 0.032, -0.065])
    def edge(P):
        q = P - H0
        th = np.abs(np.arctan2(q[:, 0], q[:, 2]))
        return np.interp(th, ang, hgt) * hs - q[:, 1]
    reg = Func(edge)
    field = Intersect([off, reg], name="medical_cap")
    m = extract(off, H0 - 0.17, H0 + 0.17, 0.0018, 3000 if ctx.fast else 7000, clip=reg, log=log)
    return m, field


def glasses(ctx, log):
    b = ctx.body
    hs, P = b.hs, b.Hpt
    support = Union([b.head_only, ctx.hair_field])
    r_frame = 0.0021
    lens_w, lens_h = 0.025 * hs, 0.0165 * hs
    frame_paths, lenses = [], []
    zc = 0.1035
    for s in (1, -1):
        c = P(0.0335 * s, 0.002, zc)
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        e = 2.6
        x = np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** (2 / e) * lens_w
        y = np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** (2 / e) * lens_h
        # slight wrap: outer side further back
        z = -0.004 * hs * np.clip(x * s / lens_w, 0, 1) ** 2
        loop = c + np.stack([x, y, z], 1)
        frame_paths.append(loop)
        cen = c + np.array([0, 0, -0.0008])
        fan_P = np.vstack([cen, loop * 0.985 + cen * 0.015 - np.array([0, 0, 0.0008])])
        fan_F = np.array([[0, 1 + i, 1 + (i + 1) % 48] for i in range(48)])
        lm = Mesh(fan_P, fan_F, np.tile([0, 0, 1.0], (len(fan_P), 1)))
        if (face_normals(lm.P, lm.F)[:, 2]).mean() < 0:
            lm.F = lm.F[:, ::-1]
        lenses.append(lm)
    # push frames out of the face/brow
    frames = []
    for loop in frame_paths:
        loop = push_out(support, loop, r_frame + 0.0006)
        frames.append(tube(loop, r_frame, seg=8, closed=True))
    inner = [fp[np.argmin(np.abs(fp[:, 0]))] for fp in frame_paths]
    bridge = catmull([inner[0] + [0, 0.004, 0], P(0, 0.010, zc + 0.002), inner[1] + [0, 0.004, 0]], 8)
    bridge = push_out(support, bridge, r_frame + 0.0006)
    parts = frames + [tube(bridge, r_frame * 0.9, seg=8)]
    pads = []
    # nose pads touching the nose
    for s in (1, -1):
        target = ray_surface(b.head_only, P(0.0105 * s, -0.008, 0.16), (-0.35 * s, 0, -1))
        n = normals(b.head_only, target[None])[0]
        pad_c = target + n * 0.0023
        attach = frame_paths[0 if s > 0 else 1]
        a0 = attach[np.argmin(np.linalg.norm(attach - P(0.012 * s, -0.004, zc), axis=1))]
        arm = catmull([a0, (a0 + pad_c) / 2 + np.array([0, 0, 0.002]), pad_c + n * 0.001], 6)
        arm = push_out(b.head_only, arm, 0.0012 + 0.0004)
        parts.append(tube(arm, 0.0009, seg=6))
        R = rot_from_axes(z=n, y=(0, 1, 0))
        pad = Ellipsoid(pad_c, (0.0045, 0.0065, 0.0013), R=R)
        pads.append(extract(pad, pad_c - 0.01, pad_c + 0.01, 0.0005, 120, log=None))
    # temples from hinge to over the ear, contact on the ear root
    for k, s in enumerate((1, -1)):
        loop = frame_paths[k]
        hinge = loop[np.argmax(loop[:, 0] * s)]
        ear = b.ear_prims[0 if s > 0 else 1]
        ear_top = ray_surface(b.head_only, ear.c + np.array([0.004 * s, 0.06, -0.004]), (0, -1, 0))
        pts = [hinge, hinge + np.array([0.008 * s, 0.002, -0.012]), P(0.083 * s, 0.012, 0.045),
               P(0.084 * s, 0.016, 0.005), ear_top + np.array([0, 0.0015, 0.004]),
               ear_top + np.array([-0.002 * s, -0.002, -0.011]), ear_top + np.array([-0.004 * s, -0.009, -0.016])]
        path = catmull(pts, 10)
        path = push_out(support, path, r_frame + 0.0006)
        # make the segment over the ear rest on it (contact)
        tail = np.arange(len(path)) >= len(path) * 0.55
        path[tail] = onto(support, path[tail], r_frame + 0.0004)
        path = push_out(support, path, r_frame + 0.0003)
        parts.append(tube(path, r_frame, seg=8))
    frame_m = merge(parts)
    lens_m = merge(lenses)
    return frame_m, lens_m, merge(pads)


def stethoscope(ctx, log, support, variant):
    b = ctx.body
    Y = lambda v: v * b.sy
    r = 0.0045
    tgt = r + 0.0013
    z_front = 0.4
    def front_pt(x, y):
        return ray_surface(support, (x, y, z_front), (0, 0, -1), target=tgt)
    yb = Y(1.10)
    # guide points: right chest -> right shoulder/neck -> back of neck -> left -> left chest
    # around the neck: points outside the garment's neck opening, dropped onto the collar
    ring_r = (0.090 if variant == "over_coat" else 0.084) + r + 0.012
    y_start = b.Hpt(0, -0.128, 0)[1]                      # just below the chin / occiput
    ring = []
    for th in np.radians(np.linspace(12, 168, 11)):        # back half only (front is the V opening)     # front-left -> back -> front-right
        x, z = np.cos(th) * ring_r, -0.02 - np.sin(th) * ring_r * 0.95
        ring.append(ray_surface(support, (x, y_start, z), (0, -1, 0), target=tgt))
    ring = ring[::-1]                                      # start on the character's right
    guide = [front_pt(-0.060, yb + 0.03), front_pt(-0.072, Y(1.25)), front_pt(-0.090, Y(1.37)), front_pt(-0.102, Y(1.43))] + ring + \
            [front_pt(0.102, Y(1.43)), front_pt(0.090, Y(1.37)), front_pt(0.072, Y(1.25)), front_pt(0.060, yb)]
    path = catmull(guide, 14)
    path = resample(path, 0.006)
    for _ in range(6):
        # rest on the support (gravity-free drape approximation) then relax
        path = onto(support, path, tgt, iters=2)
        path = smooth_path(path, 8, 0.5, fixed=(0, len(path) - 1))
    path = push_out(support, path, tgt)
    path = smooth_path(path, 2, 0.3, fixed=(0, len(path) - 1))
    path = push_out(support, path, tgt - 0.0002)
    tubing = tube(path, r, seg=10)
    metal = []
    # chest piece hangs from the left end
    end = path[-1]
    n = normals(support, end[None])[0]
    dn = end - path[-2]; dn /= np.linalg.norm(dn)
    stem_end = front_pt(0.052, yb - 0.035)
    stem = catmull([end, (end + stem_end) / 2, stem_end], 6)
    stem = push_out(support, stem, 0.0028)
    metal.append(tube(stem, 0.0025, seg=8))
    cp_c = front_pt(0.050, yb - 0.065)
    ncp = normals(support, cp_c[None])[0]
    disc_r, disc_h = 0.021, 0.008
    # place disc so its back face touches the support
    cp_c = cp_c + ncp * (disc_h / 2 - tgt + 0.0004)
    metal.append(cylinder(cp_c, ncp, disc_r, disc_h, seg=32))
    # binaurals: Y yoke on the right end -> two ear tubes with ear tips
    start = path[0]
    y0 = front_pt(-0.054, yb - 0.010)
    ears = []
    for dx in (-0.016, 0.016):
        a = front_pt(-0.054 + dx * 0.6, yb - 0.04)
        bb = front_pt(-0.054 + dx, yb - 0.10)
        p = catmull([start, y0, a, bb], 8)
        p = onto(support, p, 0.0028, iters=4)
        p = push_out(support, p, 0.0026)
        metal.append(tube(p, 0.0024, seg=8))
        tip = p[-1]
        tn = normals(support, tip[None])[0]
        tipc = tip + tn * (0.0055 - 0.0028)
        ears.append(Sphere(tipc, 0.0055))
    tips = extract(Union(ears), np.min([e.c for e in ears], 0) - 0.01, np.max([e.c for e in ears], 0) + 0.01,
                   0.0007, 400)
    return tubing, merge(metal), tips


def id_card(ctx, log, support, variant):
    b = ctx.body
    Y = lambda v: v * b.sy
    w, hgt, th = 0.054, 0.086, 0.0009
    # clipped to the breast pocket's upper edge, hanging over it
    x0 = 0.100
    y0 = Y(1.255) + 0.004 if not b.F else Y(1.33) + 0.004
    c = ray_surface(support, (x0, y0, 0.4), (0, 0, -1))
    samples = []
    for dx in (-w / 2, 0, w / 2):
        for dy in (-hgt / 2, 0, hgt / 2):
            samples.append(ray_surface(support, (x0 + dx, y0 + dy, 0.4), (0, 0, -1)))
    samples = np.array(samples)
    n = normals(support, samples).mean(0); n /= np.linalg.norm(n)
    up = np.array([0, 1.0, 0]); up = up - n * (up @ n); up /= np.linalg.norm(up)
    R = rot_from_axes(y=up, z=n)
    gx, gy = np.meshgrid(np.linspace(-w / 2, w / 2, 11), np.linspace(-hgt / 2, hgt / 2, 15))
    back = c + (R @ np.stack([gx.ravel(), gy.ravel(), -th / 2 * np.ones(gx.size)])).T
    # translate along n so the card rests on the support (min clearance 0.4 mm)
    f = support(back)
    shift = 0.0008 - f.min()
    c = c + n * shift
    card = grid_box(c, (w / 2, hgt / 2, th / 2), R, nx=10, ny=16)
    # clip at top touching the card and the support
    clip_c = c + R @ np.array([0, hgt / 2 + 0.006, 0.0])
    clip = box(clip_c, (0.009, 0.010, 0.0022), R=R)
    # push the clip out of the support if necessary
    f = support(clip.P)
    if f.min() < 0.0008:
        clip.P += n * (0.0008 - f.min())
    return card, clip


def watch(ctx, log):
    b = ctx.body
    d = b.arm_dir
    dorsal = -b.palm_n
    hw = b.hand_w
    c = b.W - d * 0.030
    angs = np.linspace(0, 2 * np.pi, 56, endpoint=False)
    U = np.outer(np.cos(angs), dorsal) + np.outer(np.sin(angs), hw)
    radii = []
    for u in U:
        p = ray_surface(b.field, c + u * 0.08, -u)
        radii.append(np.linalg.norm(p - c))
    radii = np.array(radii)
    strap_t, strap_w = 0.0028, 0.018
    rr = radii + 0.0009 + strap_t / 2
    loop = c + U * rr[:, None]
    # rectangular band
    Pn, Nn, Fn = [], [], []
    n = len(loop)
    for i in range(n):
        o = U[i]
        for (a, bb) in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            Pn.append(loop[i] + o * a * strap_t / 2 + d * bb * strap_w / 2)
            Nn.append(o * a * 0.7 + d * bb * 0.7)
    for i in range(n):
        j = (i + 1) % n
        for k in range(4):
            k2 = (k + 1) % 4
            a0, a1, b1, b0 = i * 4 + k, i * 4 + k2, j * 4 + k2, j * 4 + k
            Fn += [[a0, a1, b1], [a0, b1, b0]]
    band = Mesh(np.array(Pn), np.array(Fn), None)
    band.N = vertex_normals(band.P, band.F)
    cen = band.P[band.F].mean(1)
    fn = face_normals(band.P, band.F)
    radial = cen - c - np.outer((cen - c) @ d, d)
    # orientation: outward on outer face; inner face normals should point inward. Use axis-based test
    if np.mean(np.einsum("ij,ij->i", fn, cen - loop.mean(0))) < 0:
        band.F = band.F[:, ::-1]
    band.N = vertex_normals(band.P, band.F)
    # case on the dorsal side
    top_r = radii[0]
    case_h, case_r = 0.0095, 0.0195
    case_c = c + dorsal * (top_r + 0.0009 + strap_t + case_h / 2 - 0.0012)
    case = cylinder(case_c, dorsal, case_r, case_h, seg=40)
    # ensure case does not penetrate the wrist
    f = b.field(case.P)
    if f.min() < 0.0008:
        shift = 0.0008 - f.min()
        case.P += dorsal * shift
        case_c = case_c + dorsal * shift
    case.UV = np.full((len(case.P), 2), 0.999)
    # dial: 12 o'clock towards the little-finger side, 3 o'clock towards the hand
    e1, e2 = d, -hw
    seg = 48
    dc = case_c + dorsal * (case_h / 2 + 0.0004)
    Pd = [dc]; UVd = [[0.5, 0.5]]
    for j in range(seg):
        a = 2 * np.pi * j / seg
        Pd.append(dc + 0.0165 * (np.cos(a) * e1 + np.sin(a) * e2))
        UVd.append([0.5 + 0.5 * np.cos(a), 0.5 - 0.5 * np.sin(a)])
    Fd = np.array([[0, 1 + j, 1 + (j + 1) % seg] for j in range(seg)])
    dial = Mesh(np.array(Pd), Fd, np.tile(dorsal, (seg + 1, 1)), UV=np.array(UVd))
    if (face_normals(dial.P, dial.F) @ dorsal).mean() < 0:
        dial.F = dial.F[:, ::-1]
    return band, case, dial


def grid_box(c, half, R, nx=10, ny=16):
    """Thin box whose two large faces are subdivided grids (front face has UVs 0..1)."""
    P, N, UV, F = [], [], [], []
    for sgn in (1, -1):
        base = len(P)
        for j in range(ny + 1):
            for i in range(nx + 1):
                u, v = i / nx, j / ny
                p = np.array([(u * 2 - 1) * half[0], (1 - v * 2) * half[1], sgn * half[2]])
                P.append(c + R @ p); N.append(R @ np.array([0, 0, sgn]))
                UV.append([u, v] if sgn > 0 else [0.999, 0.999])
        for j in range(ny):
            for i in range(nx):
                a = base + j * (nx + 1) + i
                q = [a, a + 1, a + nx + 2, a + nx + 1]
                F += [[q[0], q[3], q[2]], [q[0], q[2], q[1]]] if sgn > 0 else [[q[0], q[2], q[3]], [q[0], q[1], q[2]]]
    m = Mesh(np.array(P), np.array(F), np.array(N), UV=np.array(UV))
    fn = face_normals(m.P, m.F)
    flip = np.einsum("ij,ij->i", fn, m.N[m.F[:, 0]]) < 0
    m.F[flip] = m.F[flip][:, ::-1]
    sides = box(c, half, R=R)
    keep = np.abs(sides.N @ (R @ np.array([0, 0, 1.0]))) < 0.5      # drop the two big faces
    fk = keep[sides.F].all(1)
    sides.F = sides.F[fk]
    sides.UV = np.full((len(sides.P), 2), 0.999)
    return merge([m, sides])


# ----------------------------------------------------------------- export
def _write(ctx, out_dir, copyright, name, parts, tex=None):
    """parts: list of (mesh_name, Mesh, (J, W), material_kwargs)"""
    b = ctx.body
    glb = GLB(copyright=copyright)
    names = bone_names()
    parents = {n: bone_parent(n) for n in names}
    arm, idx = glb.skeleton(names, parents, b.joints)
    skin = glb.skin(names, idx, b.joints, name=f"{ctx.gender}_skin")
    for (mname, m, jw, mat) in parts:
        mat = dict(mat)
        texname = mat.pop("texture_png", None)
        t = None
        if texname is not None:
            t = glb.image(texname[1], texname[0])
        mi = glb.material(mname, texture=t, **mat)
        glb.skinned_mesh(mname, m, jw[0], jw[1], mi, skin)
    path = os.path.join(out_dir, "clothing", ctx.gender, f"{name}.glb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = glb.write(path)
    log_ = getattr(ctx, "log", print)
    log_(f"  wrote {path} ({size / 1024:.0f} KB)")
    return path


def _tw(ctx, m, k=10, sigma=0.02):
    """Transfer weights from the body point found by projecting each garment
    vertex onto the body surface, so stacked layers get the same weights."""
    q = project(ctx.body.field, m.P, iters=6, max_step=0.05)
    J, W = transfer(ctx.mesh.P, ctx.Wd, q, k=k, sigma=sigma * 0.5)
    if len(m.F) > 50:
        W = laplacian_smooth(dense(J, W), m.F, iters=4, lam=0.5)
        J, W = top4(W)
    return J, W


WHITE = (238, 238, 234)
TEAL = (28, 112, 132)
CAP = (40, 130, 150)


def build_all(ctx, out_dir, copyright, log):
    ctx.log = log
    b = ctx.body
    g = ctx.gender
    log(f"== clothing: {g}")
    info = {}
    # scrubs top / trousers / coat
    top, top_field = scrubs_top(ctx, log)
    _write(ctx, out_dir, copyright, "scrubs_top",
           [("ScrubsTop", top, _tw(ctx, top), dict(color=rgba(TEAL), roughness=0.85, double_sided=True))])
    tr, tr_field = trousers(ctx, log)
    _write(ctx, out_dir, copyright, "medical_trousers",
           [("MedicalTrousers", tr, _tw(ctx, tr), dict(color=rgba(TEAL), roughness=0.85, double_sided=True))])
    coat, buttons, coat_field = white_coat(ctx, log)
    _write(ctx, out_dir, copyright, "white_coat",
           [("WhiteCoat", coat, _tw(ctx, coat, k=16, sigma=0.03), dict(color=rgba(WHITE), roughness=0.8, double_sided=True)),
            ("CoatButtons", buttons, _tw(ctx, buttons), dict(color=rgba((225, 225, 222)), roughness=0.35))])
    sh, sh_field = shoes(ctx, log)
    _write(ctx, out_dir, copyright, "medical_shoes",
           [("MedicalShoes", sh, _tw(ctx, sh, k=6, sigma=0.01), dict(color=rgba((240, 240, 240)), roughness=0.45, double_sided=True))])
    cap, cap_field = medical_cap(ctx, log)
    _write(ctx, out_dir, copyright, "medical_cap",
           [("MedicalCap", cap, rigid(len(cap.P), "Head"), dict(color=rgba(CAP), roughness=0.85, double_sided=True))])
    fr, lens, pads = glasses(ctx, log)
    _write(ctx, out_dir, copyright, "glasses",
           [("GlassesFrame", fr, rigid(len(fr.P), "Head"), dict(color=rgba((28, 28, 32)), roughness=0.35)),
            ("GlassesLenses", lens, rigid(len(lens.P), "Head"),
             dict(color=rgba((225, 238, 255), 0.18), roughness=0.05, alpha="BLEND", double_sided=True)),
            ("GlassesNosePads", pads, rigid(len(pads.P), "Head"), dict(color=rgba((235, 238, 240), 0.55), roughness=0.3, alpha="BLEND"))])
    # supports for the accessories that sit on clothing
    scrub_support = Union([top_field, Offset(b.field, 0.0015)])
    coat_support = Union([coat_field, top_field, Offset(b.field, 0.0015)])
    ctx.supports = {"over_coat": coat_support, "over_scrubs": scrub_support}
    card_png = tx.png_bytes(tx.id_card(g))
    open(os.path.join(out_dir, "textures", f"id_card_{g}.png"), "wb").write(card_png)
    for variant, sup in ctx.supports.items():
        tubing, metal, tips = stethoscope(ctx, log, sup, variant)
        _write(ctx, out_dir, copyright, f"stethoscope_{variant}",
               [("StethoscopeTubing", tubing, _tw(ctx, tubing, k=12, sigma=0.03), dict(color=rgba((22, 22, 26)), roughness=0.45)),
                ("StethoscopeMetal", metal, _tw(ctx, metal, k=12, sigma=0.03), dict(color=rgba((205, 207, 212)), metallic=1.0, roughness=0.22)),
                ("StethoscopeEartips", tips, _tw(ctx, tips, k=12, sigma=0.03), dict(color=rgba((30, 30, 34)), roughness=0.6))])
        card, clip = id_card(ctx, log, sup, variant)
        _write(ctx, out_dir, copyright, f"id_card_{variant}",
               [("IDCard", card, rigid(len(card.P), "Spine2"),
                 dict(color=[1, 1, 1, 1], roughness=0.35, texture_png=(f"id_card_{g}", card_png))),
                ("IDCardClip", clip, rigid(len(clip.P), "Spine2"), dict(color=rgba((190, 192, 196)), metallic=1.0, roughness=0.3))])
    band, case, dial = watch(ctx, log)
    face_png = tx.png_bytes(tx.watch_face())
    open(os.path.join(out_dir, "textures", "watch_face.png"), "wb").write(face_png)
    _write(ctx, out_dir, copyright, "watch",
           [("WatchStrap", band, rigid(len(band.P), "LeftForeArm"), dict(color=rgba((20, 20, 22)), roughness=0.6)),
            ("WatchCase", case, rigid(len(case.P), "LeftForeArm"), dict(color=rgba((200, 202, 206)), metallic=1.0, roughness=0.2)),
            ("WatchDial", dial, rigid(len(dial.P), "LeftForeArm"),
             dict(color=[1, 1, 1, 1], roughness=0.3, texture_png=("watch_face", face_png)))])
    ctx.ground_offset = -ctx.sole_bottom
    meta = {"gender": g, "shoeGroundOffset": round(-ctx.sole_bottom, 4)}
    with open(os.path.join(out_dir, "clothing", g, "_fit.json"), "w") as f:
        json.dump(meta, f, indent=2)
