"""Signed distance field (SDF) primitives and composition nodes.

Convention: negative inside, positive outside, units are metres.
Every node exposes a conservative axis-aligned bounding box (``lo``, ``hi``)
which is used to skip evaluation of far-away sub-trees ("culling").  A culled
node returns ``FAR`` (a large positive value) which is the correct answer
for any operation in this module as long as the query region is further than
``CULL_MARGIN`` from the node's solid.
"""
import numpy as np

FAR = 10.0
CULL_MARGIN = 0.06
INF = 1e9


def _v(a):
    return np.asarray(a, dtype=np.float64)


class Node:
    lo = np.array([-INF] * 3)
    hi = np.array([INF] * 3)
    bone = None
    name = ""

    def overlaps(self, qlo, qhi):
        return np.all(self.lo - CULL_MARGIN <= qhi) and np.all(self.hi + CULL_MARGIN >= qlo)

    def __call__(self, P, qlo=None, qhi=None):
        if qlo is None:
            qlo, qhi = P.min(0), P.max(0)
        if not self.overlaps(qlo, qhi):
            return np.full(len(P), FAR)
        return self.eval(P, qlo, qhi)

    def leaves(self):
        return [self]


# ---------------------------------------------------------------- primitives
class Ellipsoid(Node):
    def __init__(self, c, r, R=None, bone=None, name=""):
        self.c, self.r = _v(c), _v(r)
        self.R = None if R is None else _v(R)  # columns = local axes in world
        self.bone, self.name = bone, name
        m = self.r.max()
        self.lo, self.hi = self.c - m, self.c + m

    def eval(self, P, qlo, qhi):
        q = P - self.c
        if self.R is not None:
            q = q @ self.R
        k0 = np.linalg.norm(q / self.r, axis=1)
        k1 = np.linalg.norm(q / (self.r * self.r), axis=1)
        k1 = np.maximum(k1, 1e-12)
        return k0 * (k0 - 1.0) / k1


class Sphere(Node):
    def __init__(self, c, r, bone=None, name=""):
        self.c, self.r = _v(c), float(r)
        self.bone, self.name = bone, name
        self.lo, self.hi = self.c - r, self.c + r

    def eval(self, P, qlo, qhi):
        return np.linalg.norm(P - self.c, axis=1) - self.r


class RoundCone(Node):
    """Capsule with linearly varying radius between points a and b (exact SDF)."""

    def __init__(self, a, b, ra, rb, bone=None, name=""):
        self.a, self.b, self.ra, self.rb = _v(a), _v(b), float(ra), float(rb)
        self.bone, self.name = bone, name
        m = max(ra, rb)
        self.lo = np.minimum(self.a, self.b) - m
        self.hi = np.maximum(self.a, self.b) + m

    def eval(self, P, qlo, qhi):
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        ba = b - a
        l2 = ba @ ba
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pa = P - a
        y = pa @ ba
        z = y - l2
        xv = pa * l2 - np.outer(y, ba)
        x2 = np.einsum("ij,ij->i", xv, xv)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
        d2 = np.sqrt(x2 + y2) * il2 - r1
        d1 = np.sqrt(x2 + z2) * il2 - r2
        out = np.where(np.sign(y) * a2 * y2 < k, d2, d3)
        out = np.where(np.sign(z) * a2 * z2 > k, d1, out)
        return out


class RoundBox(Node):
    def __init__(self, c, half, r=0.0, R=None, bone=None, name=""):
        self.c, self.h, self.r = _v(c), _v(half), float(r)
        self.R = None if R is None else _v(R)
        self.bone, self.name = bone, name
        m = np.linalg.norm(self.h) + self.r
        self.lo, self.hi = self.c - m, self.c + m

    def eval(self, P, qlo, qhi):
        q = P - self.c
        if self.R is not None:
            q = q @ self.R
        q = np.abs(q) - self.h + self.r
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(q.max(1), 0)
        return outside + inside - self.r


class HalfSpace(Node):
    """Solid = {p : dot(p, n) <= d}.  Infinite bbox."""

    def __init__(self, n, d, name=""):
        n = _v(n)
        self.n, self.d, self.name = n / np.linalg.norm(n), float(d), name

    def eval(self, P, qlo, qhi):
        return P @ self.n - self.d


class Func(Node):
    """Arbitrary vectorised field function with optional bbox."""

    def __init__(self, f, lo=None, hi=None, name=""):
        self.f, self.name = f, name
        if lo is not None:
            self.lo, self.hi = _v(lo), _v(hi)

    def eval(self, P, qlo, qhi):
        return self.f(P)


# ---------------------------------------------------------------- operators
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


class Union(Node):
    def __init__(self, children, k=0.0, name=""):
        self.ch = [c for c in children if c is not None]
        self.k, self.name = float(k), name
        self.lo = np.min([c.lo for c in self.ch], axis=0) - self.k
        self.hi = np.max([c.hi for c in self.ch], axis=0) + self.k

    def eval(self, P, qlo, qhi):
        out = None
        for c in self.ch:
            if not c.overlaps(qlo, qhi):
                continue
            d = c.eval(P, qlo, qhi)
            out = d if out is None else smin(out, d, self.k)
        return np.full(len(P), FAR) if out is None else out

    def leaves(self):
        return [l for c in self.ch for l in c.leaves()]


class Subtract(Node):
    """a minus b (smooth)."""

    def __init__(self, a, b, k=0.0, name=""):
        self.a, self.b, self.k, self.name = a, b, float(k), name
        self.lo, self.hi = a.lo, a.hi

    def eval(self, P, qlo, qhi):
        da = self.a(P, qlo, qhi)
        db = self.b(P, qlo, qhi)
        return smax(da, -db, self.k)

    def leaves(self):
        return self.a.leaves()


class Intersect(Node):
    def __init__(self, children, k=0.0, name=""):
        self.ch, self.k, self.name = children, float(k), name
        finite = [c for c in children if np.all(np.abs(c.lo) < INF / 2)]
        if finite:
            self.lo = np.max([c.lo for c in finite], axis=0)
            self.hi = np.min([c.hi for c in finite], axis=0)

    def eval(self, P, qlo, qhi):
        out = None
        for c in self.ch:
            d = c(P, qlo, qhi)
            out = d if out is None else smax(out, d, self.k)
        return out

    def leaves(self):
        return [l for c in self.ch for l in c.leaves()]


class Offset(Node):
    """Grow (o>0) a solid.  ``o`` may be a float or a function of P."""

    def __init__(self, child, o, omax=None, name=""):
        self.c, self.o, self.name = child, o, name
        m = omax if omax is not None else (o if np.isscalar(o) else 0.05)
        self.lo, self.hi = child.lo - m, child.hi + m

    def eval(self, P, qlo, qhi):
        o = self.o if np.isscalar(self.o) else self.o(P)
        return self.c(P, qlo, qhi) - o

    def leaves(self):
        return self.c.leaves()


class Complement(Node):
    """Everything except the child's solid (infinite bbox)."""

    def __init__(self, child, name=""):
        self.c, self.name = child, name

    def eval(self, P, qlo, qhi):
        return -self.c(P, qlo, qhi)


class Displace(Node):
    def __init__(self, child, f, amp, name=""):
        self.c, self.f, self.name = child, f, name
        self.lo, self.hi = child.lo - amp, child.hi + amp

    def eval(self, P, qlo, qhi):
        return self.c(P, qlo, qhi) + self.f(P)


# ---------------------------------------------------------------- helpers
def gradient(node, P, eps=4e-4):
    P = np.asarray(P, dtype=np.float64)
    lo, hi = P.min(0) - 2 * eps, P.max(0) + 2 * eps
    g = np.zeros_like(P)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        g[:, i] = node(P + e, lo, hi) - node(P - e, lo, hi)
    g /= 2 * eps
    return g


def normals(node, P, eps=4e-4):
    g = gradient(node, P, eps)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def project(node, P, iters=3, target=0.0, max_step=0.004):
    """Newton-project points onto the iso-surface node == target."""
    P = np.array(P, dtype=np.float64)
    for _ in range(iters):
        lo, hi = P.min(0) - 1e-3, P.max(0) + 1e-3
        f = node(P, lo, hi) - target
        g = gradient(node, P)
        gn2 = np.maximum(np.einsum("ij,ij->i", g, g), 1e-12)
        step = (f / gn2)[:, None] * g
        sl = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(sl, 1e-12))
        P -= step
    return P


def evaluate_grid(node, lo, hi, h, block=32):
    """Evaluate node on a regular grid with block culling. Returns (vol, origin)."""
    lo, hi = _v(lo), _v(hi)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    vol = np.full(tuple(n), FAR, dtype=np.float32)
    axes = [lo[i] + h * np.arange(n[i]) for i in range(3)]
    for i0 in range(0, n[0], block):
        for j0 in range(0, n[1], block):
            for k0 in range(0, n[2], block):
                xs = axes[0][i0:i0 + block]
                ys = axes[1][j0:j0 + block]
                zs = axes[2][k0:k0 + block]
                qlo = np.array([xs[0], ys[0], zs[0]])
                qhi = np.array([xs[-1], ys[-1], zs[-1]])
                if not node.overlaps(qlo, qhi):
                    continue
                X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
                P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
                d = node.eval(P, qlo, qhi)
                vol[i0:i0 + len(xs), j0:j0 + len(ys), k0:k0 + len(zs)] = d.reshape(X.shape)
    return vol, lo


def rot_from_axes(x=None, y=None, z=None):
    """Build an orthonormal matrix whose columns are local x,y,z (two given)."""
    if x is not None and y is not None:
        x = _v(x) / np.linalg.norm(x)
        z = np.cross(x, y); z /= np.linalg.norm(z)
        y = np.cross(z, x)
    elif y is not None and z is not None:
        y = _v(y) / np.linalg.norm(y)
        x = np.cross(y, z); x /= np.linalg.norm(x)
        z = np.cross(x, y)
    elif x is not None and z is not None:
        x = _v(x) / np.linalg.norm(x)
        y = np.cross(z, x); y /= np.linalg.norm(y)
        z = np.cross(x, y)
    return np.stack([x, y, z], 1)
