"""Procedural looping idle animation (rotation tracks only, so the same clip
drives both the male and female skeleton)."""
import numpy as np
from body import ARM_DOWN_DEG

DURATION = 8.0
FPS = 30


def quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    s = np.sin(ang / 2)[:, None]
    return np.concatenate([axis[None] * s, np.cos(ang / 2)[:, None]], 1)


def qmul(a, b):
    x1, y1, z1, w1 = a.T; x2, y2, z2, w2 = b.T
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], 1)


def idle():
    n = int(DURATION * FPS) + 1
    t = np.linspace(0, DURATION, n)
    w8 = 2 * np.pi * t / DURATION      # one cycle per loop
    w4 = 2 * w8                       # breathing: two cycles per loop
    deg = np.radians
    X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    tr = {}
    breathe = np.sin(w4)
    sway = np.sin(w8)
    tr["Hips"] = qmul(quat_axis_angle(Z, deg(1.0) * sway), quat_axis_angle(Y, deg(1.5) * np.sin(w8 + 0.6)))
    tr["Spine"] = qmul(quat_axis_angle(X, deg(0.8) * breathe), quat_axis_angle(Z, deg(-0.6) * sway))
    tr["Spine1"] = quat_axis_angle(X, deg(-1.0) * breathe)
    tr["Spine2"] = qmul(quat_axis_angle(X, deg(-1.2) * breathe), quat_axis_angle(Z, deg(-0.4) * sway))
    tr["Neck"] = quat_axis_angle(X, deg(1.0) * breathe)
    tr["Head"] = qmul(quat_axis_angle(Y, deg(4.0) * np.sin(w8 + 1.3)), quat_axis_angle(X, deg(1.5) * np.sin(w8 * 2 + 0.4)))
    # arms: lowered a little from the A-pose to a relaxed hang, gentle sway
    lower = deg(8)
    for side, s in (("Left", 1), ("Right", -1)):
        tr[f"{side}Shoulder"] = quat_axis_angle(Z, s * deg(-0.8) * breathe)
        tr[f"{side}Arm"] = qmul(quat_axis_angle(Z, -s * (lower + deg(0.8) * np.sin(w8 + (0 if s > 0 else 0.8)))),
                                quat_axis_angle(X, deg(1.5) * np.sin(w8 + (0.3 if s > 0 else 1.1))))
        d = np.array([s * np.cos(deg(ARM_DOWN_DEG)), -np.sin(deg(ARM_DOWN_DEG)), 0.0])
        elbow_axis = np.cross(d, (0, 0, 1))
        tr[f"{side}ForeArm"] = quat_axis_angle(elbow_axis, deg(4.0) + deg(1.0) * np.sin(w8 + 0.5))
        hq = tr["Hips"].copy()
        hq[:, :3] *= -1                       # inverse hip rotation keeps the feet planted
        tr[f"{side}UpLeg"] = hq
        tr[f"{side}Leg"] = quat_axis_angle(X, deg(0.5) * (1 + sway * s) * 0.5)
    for k in tr:
        q = tr[k]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        # keep hemisphere consistent
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
    return t, tr
