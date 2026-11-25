"""Mesh extraction / processing helpers."""
import numpy as np
from skimage.measure import marching_cubes
import fast_simplification
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sdf import evaluate_grid, normals as sdf_normals, project


class Mesh:
    def __init__(self, P, F, N=None, UV=None, C=None):
        self.P = np.asarray(P, np.float64)
        self.F = np.asarray(F, np.int64)
        self.N = N
        self.UV = UV
        self.C = C  # per-vertex RGBA float 0..1

    @property
    def ntri(self):
        return len(self.F)


def face_normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def vertex_normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    N = np.zeros_like(P)
    for i in range(3):
        np.add.at(N, F[:, i], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)


def compact(P, F):
    used = np.unique(F)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    return P[used], remap[F], used


def largest_components(P, F, min_frac=0.02):
    """Drop tiny disconnected islands (marching-cubes debris)."""
    n = len(P)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    flab = lab[F[:, 0]]
    counts = np.bincount(flab, minlength=nc)
    keep = counts[flab] >= min_frac * counts.max()
    P2, F2, _ = compact(P, F[keep])
    return P2, F2


def clip_mesh(P, F, r):
    """Keep the part of a triangle mesh where r < 0 (exact edge interpolation)."""
    inside = r < 0
    n_in = inside[F].sum(1)
    keep = F[n_in == 3]
    mixed = F[(n_in == 1) | (n_in == 2)]
    newP = [P]
    cache = {}
    base = len(P)
    extra = []

    def cut(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            t = r[a] / (r[a] - r[b])
            extra.append(P[a] + t * (P[b] - P[a]))
            cache[key] = base + len(extra) - 1
        return cache[key]

    out = [keep]
    tris = []
    for tri in mixed:
        ins = [bool(inside[v]) for v in tri]
        # rotate so that the "odd one" is first
        k = ins.index(True) if sum(ins) == 1 else ins.index(False)
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        ab, ac = cut(a, b), cut(a, c)
        if sum(ins) == 1:
            tris.append([a, ab, ac])
        else:
            tris.append([ab, b, c]); tris.append([ab, c, ac])
    if tris:
        out.append(np.array(tris, np.int64))
    P2 = np.vstack([P] + ([np.array(extra)] if extra else []))
    F2 = np.vstack(out)
    P2, F2, _ = compact(P2, F2)
    return P2, F2


def extract(node, lo, hi, h, target_tris, normal_node=None, open_node=None,
            cap_eps=None, min_frac=0.02, project_iters=3, log=None, clip=None):
    """Marching cubes on ``node``.

    open_node: if given, faces whose centroid lies strictly inside open_node
    (value < -cap_eps) are removed -> produces an open (single sided) surface.
    normal_node: field used for shading normals (defaults to node).
    """
    vol, origin = evaluate_grid(node, lo, hi, h)
    if vol.min() >= 0 or vol.max() <= 0:
        raise RuntimeError(f"no surface for {node.name}")
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
    v = v + origin
    f = f.astype(np.int64)
    # consistent outward winding
    fn = face_normals(v, f)
    cen = v[f].mean(1)
    g = sdf_normals(node, cen[:: max(1, len(cen) // 4000)])
    if np.mean(np.einsum("ij,ij->i", fn[:: max(1, len(cen) // 4000)], g)) < 0:
        f = f[:, ::-1]
    if clip is not None:
        v, f = clip_mesh(v, f, clip(v))
    if open_node is not None:
        eps = cap_eps if cap_eps is not None else 0.35 * h
        val = open_node(cen)
        f = f[val >= -eps]
        v, f, _ = compact(v, f)
    v, f = largest_components(v, f, min_frac)
    n0 = len(f)
    if len(f) > target_tris:
        red = 1.0 - target_tris / len(f)
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), red, agg=6)
        v = v.astype(np.float64)
        f = f.astype(np.int64)
        v, f, _ = compact(v, f)
    if project_iters:
        # pin open-boundary vertices (they lie on the clip curve) and project the rest
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        bnd = np.zeros(len(v), bool)
        bnd[ue[cnt == 1].ravel()] = True
        v[~bnd] = project(node, v[~bnd], iters=project_iters)
    nn = normal_node if normal_node is not None else node
    N = sdf_normals(nn, v)
    bad = ~np.isfinite(N).all(1) | (np.linalg.norm(N, axis=1) < 0.5)
    if bad.any():
        N[bad] = vertex_normals(v, f)[bad]
    if log:
        log(f"  {node.name or 'mesh'}: MC {n0} tris -> {len(f)} tris, {len(v)} verts")
    return Mesh(v, f, N)


def merge(meshes):
    P, F, N, C, UV = [], [], [], [], []
    off = 0
    for m in meshes:
        P.append(m.P); F.append(m.F + off); N.append(m.N)
        if m.C is not None: C.append(m.C)
        if m.UV is not None: UV.append(m.UV)
        off += len(m.P)
    out = Mesh(np.concatenate(P), np.concatenate(F), np.concatenate(N))
    if len(C) == len(meshes): out.C = np.concatenate(C)
    if len(UV) == len(meshes): out.UV = np.concatenate(UV)
    return out


# ------------------------------------------------------------ explicit shapes
def _frames(path, closed=False, up_hint=None):
    """Rotation-minimising frames along a polyline."""
    path = np.asarray(path, np.float64)
    n = len(path)
    T = np.zeros_like(path)
    if closed:
        T = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    else:
        T[1:-1] = path[2:] - path[:-2]
        T[0] = path[1] - path[0]
        T[-1] = path[-1] - path[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0]) if up_hint is None else np.asarray(up_hint, float)
    if abs(T[0] @ up) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    Nn = np.zeros_like(path)
    Nn[0] = np.cross(T[0], up); Nn[0] /= np.linalg.norm(Nn[0])
    for i in range(1, n):
        v = Nn[i - 1] - (Nn[i - 1] @ T[i]) * T[i]
        Nn[i] = v / np.linalg.norm(v)
    B = np.cross(T, Nn)
    return T, Nn, B


def tube(path, radius, seg=10, closed=False, cap=True, up_hint=None):
    path = np.asarray(path, np.float64)
    n = len(path)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    T, Nn, B = _frames(path, closed, up_hint)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    dirs = Nn[:, None, :] * ca[None, :, None] + B[:, None, :] * sa[None, :, None]
    P = (path[:, None, :] + dirs * r[:, None, None]).reshape(-1, 3)
    N = dirs.reshape(-1, 3)
    F = []
    rings = n if closed else n - 1
    for i in range(rings):
        i2 = (i + 1) % n
        for j in range(seg):
            j2 = (j + 1) % seg
            a, b, c, d = i * seg + j, i * seg + j2, i2 * seg + j2, i2 * seg + j
            F += [[a, b, c], [a, c, d]]
    F = np.array(F, np.int64)
    m = Mesh(P, F, N)
    if cap and not closed:
        caps = []
        for end, sgn in ((0, -1), (n - 1, 1)):
            cp = path[end][None]
            cP = np.concatenate([cp, path[end] + dirs[end] * r[end]])
            cN = np.tile(T[end] * sgn, (seg + 1, 1))
            cF = np.array([[0, 1 + j, 1 + (j + 1) % seg] for j in range(seg)])
            caps.append(Mesh(cP, cF, cN))
        # fix orientation of caps against outward direction
        fixed = []
        for cm, (end, sgn) in zip(caps, ((0, -1), (n - 1, 1))):
            fn = face_normals(cm.P, cm.F)
            if (fn @ (T[end] * sgn)).mean() < 0:
                cm.F = cm.F[:, ::-1]
            fixed.append(cm)
        m = merge([m] + fixed)
    return m


def uv_sphere(c, r, nu=24, nv=16, axis=(0, 0, 1)):
    """Sphere whose pole is along ``axis``; UV v=polar angle/pi, u=azimuth."""
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    tmp = np.array([0, 1.0, 0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(tmp, axis); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    P, N, UV = [], [], []
    for i in range(nv + 1):
        th = np.pi * i / nv
        for j in range(nu + 1):
            ph = 2 * np.pi * j / nu
            d = axis * np.cos(th) + np.sin(th) * (e1 * np.cos(ph) + e2 * np.sin(ph))
            P.append(np.asarray(c) + r * d); N.append(d); UV.append([j / nu, i / nv])
    F = []
    for i in range(nv):
        for j in range(nu):
            a = i * (nu + 1) + j
            b, cc, d = a + 1, a + nu + 2, a + nu + 1
            if i > 0: F.append([a, d, b])
            if i < nv - 1: F.append([b, d, cc])
    m = Mesh(np.array(P), np.array(F), np.array(N), UV=np.array(UV))
    fn = face_normals(m.P, m.F)
    if (np.einsum("ij,ij->i", fn, m.P[m.F].mean(1) - c)).mean() < 0:
        m.F = m.F[:, ::-1]
    return m


def box(c, half, R=None, uv_front=False):
    """Box (optionally rotated, R columns = local axes) with flat normals.
    If uv_front, the local +z face gets UVs 0..1 (u along +x, v along -y)."""
    c = np.asarray(c, float); half = np.asarray(half, float)
    R = np.eye(3) if R is None else np.asarray(R, float)
    E = np.eye(3)
    P, N, F, UV = [], [], [], []
    for axis in range(3):
        for s in (-1, 1):
            n = E[axis] * s
            iu, iv = (axis + 1) % 3, (axis + 2) % 3
            base = len(P)
            for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = n * half[axis] + E[iu] * half[iu] * a + E[iv] * half[iv] * b
                P.append(c + R @ p); N.append(R @ n)
                if uv_front and axis == 2 and s == 1:
                    UV.append([0.5 + 0.5 * p[0] / half[0], 0.5 - 0.5 * p[1] / half[1]])
                else:
                    UV.append([0.999, 0.999])
            F += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    m = Mesh(np.array(P), np.array(F), np.array(N), UV=np.array(UV))
    fn = face_normals(m.P, m.F)
    flip = np.einsum("ij,ij->i", fn, m.N[m.F[:, 0]]) < 0
    m.F[flip] = m.F[flip][:, ::-1]
    return m


def disc(c, normal, r, seg=32, uv=True):
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    tmp = np.array([0, 1.0, 0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(tmp, normal); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    P = [np.asarray(c, float)]; UV = [[0.5, 0.5]]
    for j in range(seg):
        a = 2 * np.pi * j / seg
        P.append(c + r * (np.cos(a) * e1 + np.sin(a) * e2))
        UV.append([0.5 + 0.5 * np.cos(a), 0.5 - 0.5 * np.sin(a)])
    F = np.array([[0, 1 + j, 1 + (j + 1) % seg] for j in range(seg)])
    m = Mesh(np.array(P), F, np.tile(normal, (seg + 1, 1)), UV=np.array(UV))
    if (face_normals(m.P, m.F) @ normal).mean() < 0:
        m.F = m.F[:, ::-1]
    return m, e1, e2


def cylinder(c, axis, r, h, seg=32, cap_uv_top=False):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    c = np.asarray(c, float)
    path = [c - axis * h / 2, c + axis * h / 2]
    side = tube(path, r, seg=seg, cap=False)
    top, e1, e2 = disc(c + axis * h / 2, axis, r, seg)
    bot, _, _ = disc(c - axis * h / 2, -axis, r, seg)
    side.UV = np.full((len(side.P), 2), 0.999)
    if not cap_uv_top:
        top.UV = np.full((len(top.P), 2), 0.999)
    bot.UV = np.full((len(bot.P), 2), 0.999)
    # side normals
    side.N = side.P - c - np.outer((side.P - c) @ axis, axis)
    side.N /= np.linalg.norm(side.N, axis=1, keepdims=True)
    fn = face_normals(side.P, side.F)
    if np.einsum("ij,ij->i", fn, side.N[side.F[:, 0]]).mean() < 0:
        side.F = side.F[:, ::-1]
    return merge([side, top, bot])
