"""Skin weight computation and transfer."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

from body import bone_names, bone_parent

NAMES = bone_names()
IDX = {n: i for i, n in enumerate(NAMES)}


def bone_graph_dist():
    n = len(NAMES)
    D = np.full((n, n), 99, int)
    adj = [[] for _ in range(n)]
    for nm in NAMES:
        p = bone_parent(nm)
        if p:
            a, b = IDX[nm], IDX[p]
            adj[a].append(b); adj[b].append(a)
    for s in range(n):
        D[s, s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if D[s, v] > D[s, u] + 1:
                        D[s, v] = D[s, u] + 1
                        nxt.append(v)
            frontier = nxt
    return D


def top4(W):
    """Dense (V, nb) -> joints (V,4) uint8, weights (V,4) float32 normalised."""
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w[w < 1e-3] = 0
    s = w.sum(1, keepdims=True)
    w = w / np.maximum(s, 1e-12)
    # sort indices for determinism where weight 0 -> joint 0
    idx = np.where(w > 0, idx, 0)
    return idx.astype(np.uint8), w.astype(np.float32)


def dense(J, Wt, nb=len(NAMES)):
    W = np.zeros((len(J), nb))
    for k in range(4):
        np.add.at(W, (np.arange(len(J)), J[:, k].astype(int)), Wt[:, k])
    return W


def laplacian_smooth(W, F, iters=2, lam=0.5):
    n = len(W)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(iters):
        avg = (A @ W) / np.maximum(deg, 1)[:, None]
        W = (1 - lam) * W + lam * avg
    return W


def body_weights(body, P, F, tau_default=0.016):
    """Weights from tagged SDF primitives restricted by bone-graph adjacency."""
    prims = body.tagged_prims()
    D = np.stack([p(P) for p in prims], 1)               # (V, nprim)
    bones = np.array([IDX[p.bone] for p in prims])
    G = bone_graph_dist()
    nb = len(NAMES)
    # per bone minimal distance
    Db = np.full((len(P), nb), 1e3)
    for j, b in enumerate(bones):
        Db[:, b] = np.minimum(Db[:, b], D[:, j])
    dom = np.argmin(Db, 1)
    dmin = Db[np.arange(len(P)), dom]
    allowed = G[dom] <= 2
    fingerish = np.array(["Hand" in n and n[-1].isdigit() for n in NAMES])
    tau = np.where(fingerish[dom], 0.004, tau_default)
    W = np.exp(-(Db - dmin[:, None]) / tau[:, None]) * allowed
    # HeadTop_End / Toe_End are leaf helpers: never weighted
    for leaf in ("HeadTop_End", "LeftToe_End", "RightToe_End"):
        W[:, IDX[leaf]] = 0
    W /= W.sum(1, keepdims=True)
    W = laplacian_smooth(W, F, iters=3, lam=0.5)
    return top4(W)


def transfer(src_P, src_W_dense, dst_P, k=10, sigma=0.02):
    tree = cKDTree(src_P)
    d, i = tree.query(dst_P, k=k)
    w = np.exp(-(d / sigma) ** 2) + 1e-9
    w /= w.sum(1, keepdims=True)
    W = np.einsum("vk,vkb->vb", w, src_W_dense[i])
    return top4(W)


def rigid(n, bone):
    J = np.zeros((n, 4), np.uint8); J[:, 0] = IDX[bone]
    Wt = np.zeros((n, 4), np.float32); Wt[:, 0] = 1
    return J, Wt
