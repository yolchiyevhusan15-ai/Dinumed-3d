"""Procedural anatomy: skeleton + SDF body description for male / female.

Coordinate system (glTF): +Y up, character faces +Z, character's LEFT is +X.
Units: metres.  Bind pose: A-pose (arms ARM_DOWN_DEG below horizontal).
"""
import numpy as np
from sdf import (Ellipsoid, Sphere, RoundCone, RoundBox, HalfSpace, Union,
                 Subtract, Intersect, Complement, rot_from_axes)

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
ARM_DOWN_DEG = 55          # A-pose: arms this many degrees below horizontal

# Canonical joint order shared by EVERY asset (characters and clothing).
def bone_names():
    names = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End"]
    for side in ("Left", "Right"):
        names += [f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand"]
        for f in FINGERS:
            names += [f"{side}Hand{f}{i}" for i in (1, 2, 3)]
    for side in ("Left", "Right"):
        names += [f"{side}UpLeg", f"{side}Leg", f"{side}Foot", f"{side}ToeBase", f"{side}Toe_End"]
    return names


def bone_parent(name):
    fixed = {"Hips": None, "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1",
             "Neck": "Spine2", "Head": "Neck", "HeadTop_End": "Head"}
    if name in fixed:
        return fixed[name]
    side = "Left" if name.startswith("Left") else "Right"
    rest = name[len(side):]
    chain = {"Shoulder": "Spine2", "Arm": f"{side}Shoulder", "ForeArm": f"{side}Arm",
             "Hand": f"{side}ForeArm", "UpLeg": "Hips", "Leg": f"{side}UpLeg",
             "Foot": f"{side}Leg", "ToeBase": f"{side}Foot", "Toe_End": f"{side}ToeBase"}
    if rest in chain:
        return chain[rest]
    for f in FINGERS:
        if rest.startswith("Hand" + f):
            i = int(rest[-1])
            return f"{side}Hand" if i == 1 else f"{side}Hand{f}{i - 1}"
    raise KeyError(name)


def _n(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Body:
    """Holds skeleton joint positions, SDF prims with bone tags and landmarks."""

    def __init__(self, gender):
        self.gender = gender
        F = gender == "female"
        self.F = F
        m = lambda a, b: b if F else a       # male value, female value
        sy = m(1.0, 0.935)
        Y = lambda v: v * sy
        # stylised ("cartoon") proportions: larger head sitting on the same body
        self.hs_style = m(1.36, 1.30)
        self.height = Y(1.555) + 0.218 * self.hs_style
        J = {}
        # ------------------------------------------------------------ skeleton
        J["Hips"] = (0, Y(0.97), 0.0)
        J["Spine"] = (0, Y(1.06), -0.005)
        J["Spine1"] = (0, Y(1.18), -0.01)
        J["Spine2"] = (0, Y(1.30), -0.015)
        J["Neck"] = (0, Y(1.475), -0.02)
        J["Head"] = (0, Y(1.605), -0.005)
        J["HeadTop_End"] = (0, self.height, -0.005)
        shx = m(0.178, 0.152)
        arm_ang = np.radians(ARM_DOWN_DEG)
        self.arm_dir = d = np.array([np.cos(arm_ang), -np.sin(arm_ang), 0.0])
        self.palm_n = pn = np.array([-np.sin(arm_ang), -np.cos(arm_ang), 0.0])  # palm faces down/in
        self.hand_w = hw = np.array([0, 0, 1.0])                                   # thumb side
        L_up, L_fa, L_palm = m(0.295, 0.27), m(0.26, 0.235), m(0.098, 0.088)
        S = np.array([shx, Y(1.425), -0.02])
        E = S + d * L_up
        W = E + d * L_fa
        J["LeftShoulder"] = (0.03, Y(1.445), -0.015)
        J["LeftArm"] = tuple(S)
        J["LeftForeArm"] = tuple(E)
        J["LeftHand"] = tuple(W)
        # fingers
        fr = m(1.0, 0.88)
        self.finger_paths = {}
        base_w = np.array([0.031, 0.011, -0.009, -0.028]) * fr
        lens = np.array([[0.046, 0.027, 0.023], [0.050, 0.031, 0.024],
                         [0.046, 0.029, 0.023], [0.037, 0.022, 0.020]]) * fr
        spread = np.radians([7, 1, -5, -11])
        for k, f in enumerate(FINGERS[1:]):
            p = W + d * L_palm * (1.0 if k in (1, 2) else 0.96) + hw * base_w[k] + pn * 0.003
            dirn = _n(d * np.cos(spread[k]) + hw * np.sin(spread[k]))
            pts = [p]
            for i in range(3):
                curl = np.radians(6 + 7 * i)
                dirn = _n(dirn * np.cos(curl) + pn * np.sin(curl))
                p = p + dirn * lens[k, i]
                pts.append(p)
            self.finger_paths[f] = pts
            for i in range(3):
                J[f"LeftHand{f}{i + 1}"] = tuple(pts[i])
        # thumb
        tb = W + d * 0.018 * fr + hw * 0.020 * fr + pn * 0.011 * fr
        tdir = _n(d * 0.78 + hw * 0.50 + pn * 0.30)
        tl = np.array([0.036, 0.030, 0.025]) * fr
        pts = [tb]
        p = tb
        for i in range(3):
            if i > 0:
                tdir = _n(tdir * np.cos(np.radians(10)) + pn * np.sin(np.radians(10)) + d * 0.05)
            p = p + tdir * tl[i]
            pts.append(p)
        self.finger_paths["Thumb"] = pts
        for i in range(3):
            J[f"LeftHandThumb{i + 1}"] = tuple(pts[i])
        # legs
        hx = m(0.090, 0.088)
        Hj = np.array([hx, Y(0.905), 0.0])
        K = np.array([hx + 0.004, Y(0.50), 0.012])
        A = np.array([hx + 0.008, Y(0.085), -0.015])
        fl = m(1.0, 0.9)
        Tb = np.array([hx + 0.014, 0.024, 0.115 * fl])
        Te = np.array([hx + 0.016, 0.018, 0.175 * fl])
        J["LeftUpLeg"], J["LeftLeg"], J["LeftFoot"] = tuple(Hj), tuple(K), tuple(A)
        J["LeftToeBase"], J["LeftToe_End"] = tuple(Tb), tuple(Te)
        # mirror
        for k in list(J):
            if k.startswith("Left"):
                x, y, z = J[k]
                J["Right" + k[4:]] = (-x, y, z)
        self.joints = {k: np.array(v, float) for k, v in J.items()}
        self.S, self.E, self.W, self.Hj, self.K, self.A = S, E, W, Hj, K, A
        self.L_palm = L_palm
        self.sy = sy
        self._build_prims(m, Y)

    # ---------------------------------------------------------------- body
    def _build_prims(self, m, Y):
        F = self.F
        d, pn, hw = self.arm_dir, self.palm_n, self.hand_w
        S, E, W, Hj, K, A = self.S, self.E, self.W, self.Hj, self.K, self.A
        torso = [
            Ellipsoid((0, Y(0.955), -0.01), m((0.158, 0.115, 0.105), (0.170, 0.118, 0.110)), bone="Hips"),
            Ellipsoid((0.072, Y(0.905), -0.052), m((0.082, 0.092, 0.078), (0.090, 0.098, 0.088)), bone="Hips"),
            Ellipsoid((-0.072, Y(0.905), -0.052), m((0.082, 0.092, 0.078), (0.090, 0.098, 0.088)), bone="Hips"),
            Ellipsoid((0, Y(1.08), 0.0), m((0.138, 0.12, 0.098), (0.118, 0.12, 0.088)), bone="Spine"),
            Ellipsoid((0, Y(1.11), 0.022), m((0.105, 0.11, 0.066), (0.092, 0.10, 0.062)), bone="Spine"),
            Ellipsoid((0, Y(1.23), -0.01), m((0.150, 0.14, 0.108), (0.125, 0.13, 0.094)), bone="Spine1"),
            Ellipsoid((0, Y(1.335), -0.018), m((0.168, 0.12, 0.110), (0.138, 0.11, 0.094)), bone="Spine2"),
            Ellipsoid((0.062, Y(1.30), -0.07), m((0.09, 0.13, 0.055), (0.075, 0.12, 0.05)), bone="Spine2"),
            Ellipsoid((-0.062, Y(1.30), -0.07), m((0.09, 0.13, 0.055), (0.075, 0.12, 0.05)), bone="Spine2"),
            # trapezius / shoulder girdle
            RoundCone((0, Y(1.475), -0.035), (0.13 * m(1, 0.88), Y(1.44), -0.03), m(0.055, 0.045), m(0.045, 0.036), bone="LeftShoulder"),
            RoundCone((0, Y(1.475), -0.035), (-0.13 * m(1, 0.88), Y(1.44), -0.03), m(0.055, 0.045), m(0.045, 0.036), bone="RightShoulder"),
        ]
        if not F:
            torso += [
                Ellipsoid((0.068, Y(1.335), 0.052), (0.075, 0.058, 0.05), bone="Spine2"),
                Ellipsoid((-0.068, Y(1.335), 0.052), (0.075, 0.058, 0.05), bone="Spine2"),
            ]
        else:
            torso += [
                Ellipsoid((0.06, Y(1.30), 0.035), (0.068, 0.05, 0.05), bone="Spine2"),
                Ellipsoid((-0.06, Y(1.30), 0.035), (0.068, 0.05, 0.05), bone="Spine2"),
                Sphere((0.072, Y(1.275), 0.062), 0.056, bone="Spine2"),
                Sphere((-0.072, Y(1.275), 0.062), 0.056, bone="Spine2"),
            ]
        torso_u = Union(torso, k=m(0.035, 0.032), name="torso")
        neck = RoundCone((0, Y(1.44), -0.025), (0, Y(1.63), -0.005), m(0.063, 0.054), m(0.054, 0.046), bone="Neck")
        neck_u = Union([neck], name="neck")
        head = self._head(m, Y)
        arms = [self._arm(m, Y, side) for side in ("Left", "Right")]
        legs = [self._leg(m, Y, side) for side in ("Left", "Right")]
        self.parts = {"torso": torso_u, "neck": neck_u, "head": head,
                      "armL": arms[0], "armR": arms[1], "legL": legs[0], "legR": legs[1]}
        upper = Union([torso_u, neck_u], k=0.03, name="upper")
        with_head = Union([upper, head], k=0.018)
        limbs = Union([with_head] + arms, k=0.025)
        body = Union([limbs] + legs, k=0.03, name="body")
        # flat soles
        self.field = Intersect([body, HalfSpace((0, -1, 0), 0.0)], k=0.004, name="body")
        # useful fields for garments
        self.trunk_legs = Intersect([Union([Union([torso_u], k=0.0)] + legs, k=0.03), HalfSpace((0, -1, 0), 0.0)], name="trunk_legs")
        self.trunk_arms = Union([Union([torso_u, neck_u], k=0.03)] + arms, k=0.025, name="trunk_arms")
        # whole body except the head (garments that hang over the hips must see the thighs)
        self.no_head = Intersect([Union([Union([Union([torso_u, neck_u], k=0.03)] + arms, k=0.025)] + legs, k=0.03),
                                  HalfSpace((0, -1, 0), 0.0)], k=0.004, name="no_head")
        self.head_only = head
        self.trunk_shanks = Union([torso_u] + self.leg_nofeet, k=0.03, name="trunk_shanks")
        self.feet_field = Intersect([Union(self.feet), HalfSpace((0, -1, 0), 0.0)], k=0.004, name="feet")

    def _mirror(self, p, side):
        p = np.array(p, float)
        if side == "Right":
            p[0] = -p[0]
        return p

    def _arm(self, m, Y, side):
        d, pn, hw = self.arm_dir.copy(), self.palm_n.copy(), self.hand_w
        S, E, W = self.S, self.E, self.W
        mir = lambda p: self._mirror(p, side)
        if side == "Right":
            d[0] = -d[0]; pn[0] = -pn[0]
        B = lambda n: f"{side}{n}"
        r = m(1.0, 0.86)
        up = _n(np.array([0, 1.0, 0]) - d * d[1])  # perpendicular to arm, pointing up/out
        prims = [
            Ellipsoid(mir(S + np.array([0.012, 0.012, 0])), np.array([0.058, 0.052, 0.056]) * r * m(1, 0.90), bone=B("Arm")),
            RoundCone(mir(S), mir(E), 0.050 * r, 0.038 * r, bone=B("Arm")),
            Ellipsoid(mir(S + self.arm_dir * 0.15 + np.array([0, 0, 0.012])), np.array([0.044, 0.044, 0.044]) * r * m(1, 0.86), bone=B("Arm")),
            RoundCone(mir(E), mir(W), 0.039 * r, 0.026 * r, bone=B("ForeArm")),
            Ellipsoid(mir(E + self.arm_dir * 0.07), np.array([0.042, 0.042, 0.040]) * r, bone=B("ForeArm")),
        ]
        # hand
        Rh = rot_from_axes(x=d, y=-pn)  # local x along hand, y = back of hand, z = width
        palm_c = mir(W + self.arm_dir * self.L_palm * 0.52 + self.palm_n * 0.004)
        hr = m(1.0, 0.88)
        prims.append(RoundBox(palm_c, np.array([self.L_palm * 0.5, 0.012, 0.040]) * np.array([1, hr, hr]),
                              0.011 * hr, R=Rh, bone=B("Hand")))
        prims.append(Ellipsoid(mir(W + self.arm_dir * 0.01), np.array([0.024, 0.022, 0.030]) * hr, R=Rh, bone=B("Hand")))
        # thenar
        prims.append(Ellipsoid(mir(W + self.arm_dir * 0.035 + self.hand_w * 0.022 + self.palm_n * 0.008),
                               np.array([0.03, 0.014, 0.018]) * hr, R=Rh, bone=B("HandThumb1")))
        fingers = []
        for f, pts in self.finger_paths.items():
            rad = (0.0095 if f != "Pinky" else 0.0082) * hr
            if f == "Thumb":
                rad = 0.0112 * hr
            for i in range(3):
                ra = rad * (1 - 0.09 * i)
                rb = rad * (1 - 0.09 * (i + 1)) if i < 2 else rad * 0.78
                fingers.append(RoundCone(mir(pts[i]), mir(pts[i + 1]), ra, rb, bone=B(f"Hand{f}{i + 1}")))
        hand = Union([Union(fingers, k=0.0025)] + prims[5:], k=0.008)
        return Union([Union(prims[:5], k=0.03), hand], k=0.02, name=f"arm{side[0]}")

    def _leg(self, m, Y, side):
        Hj, K, A = self.Hj, self.K, self.A
        mir = lambda p: self._mirror(p, side)
        B = lambda n: f"{side}{n}"
        r = m(1.0, 1.0)
        fl = m(1.0, 0.9)
        hx = Hj[0]
        prims = [
            RoundCone(mir(Hj + np.array([0.005, 0.0, -0.005])), mir(K), m(0.086, 0.088), m(0.052, 0.050), bone=B("UpLeg")),
            Ellipsoid(mir((hx + 0.008, Y(0.72), 0.035)), m((0.058, 0.13, 0.05), (0.058, 0.13, 0.05)), bone=B("UpLeg")),
            Ellipsoid(mir((hx + 0.02, Y(0.80), -0.01)), m((0.075, 0.12, 0.07), (0.083, 0.12, 0.075)), bone=B("UpLeg")),
            Ellipsoid(mir((hx + 0.004, Y(0.505), 0.045)), (0.034, 0.035, 0.022), bone=B("Leg")),
            RoundCone(mir(K), mir(A), m(0.048, 0.045), m(0.030, 0.027), bone=B("Leg")),
            Ellipsoid(mir((hx + 0.004, Y(0.355), -0.025)), m((0.046, 0.095, 0.045), (0.043, 0.09, 0.042)), bone=B("Leg")),
            Sphere(mir(A + np.array([0.022, -0.005, 0.002])), 0.016, bone=B("Leg")),
            Sphere(mir(A + np.array([-0.022, 0.0, 0.002])), 0.016, bone=B("Leg")),
            # foot
            Ellipsoid(mir((A[0] - 0.002, 0.040, -0.035 * fl)), np.array([0.033, 0.040, 0.042]) * fl, bone=B("Foot")),
            RoundCone(mir((A[0], 0.045, -0.01)), mir((A[0] + 0.012, 0.028, 0.110 * fl)), 0.034 * fl, 0.028 * fl, bone=B("Foot")),
            RoundBox(mir((A[0] + 0.006, 0.020, 0.045 * fl)), np.array([0.034, 0.016, 0.07]) * fl, 0.014 * fl, bone=B("Foot")),
            RoundBox(mir((A[0] + 0.014, 0.017, 0.152 * fl)), np.array([0.036, 0.011, 0.032]) * fl, 0.010 * fl, bone=B("ToeBase")),
        ]
        self.__dict__.setdefault("leg_nofeet", []).append(Union(prims[:8], k=0.03))
        self.__dict__.setdefault("feet", []).append(Union(prims[4:], k=0.02))
        return Union([Union(prims[:8], k=0.03), Union(prims[8:], k=0.02)], k=0.025, name=f"leg{side[0]}")

    def _head(self, m, Y):
        """Stylised cartoon head: one smooth egg shape, large eyes, small round
        nose, round ears.  The mouth is a separate layered mesh (see build.py)."""
        hs = self.hs_style
        nz = m(1.0, 0.86)
        jw = m(1.0, 0.94)
        H0 = np.array([0, Y(1.555) + 0.111 * hs, 0.0])
        self.head_c = H0
        P = lambda x, y, z: H0 + np.array([x, y, z]) * hs
        self.cran_c, self.cran_r = P(0, 0.014, -0.010), np.array([0.082, 0.096, 0.097]) * hs
        skull = [
            Ellipsoid(self.cran_c, self.cran_r, bone="Head"),                                               # cranium
            Ellipsoid(P(0, -0.034, 0.006), np.array([0.067 * jw, 0.070, 0.070]) * hs, bone="Head"),         # cheeks / lower face
            Ellipsoid(P(0.034 * jw, -0.058, 0.016), np.array([0.030, 0.030, 0.036]) * hs, bone="Head"),     # jaw
            Ellipsoid(P(-0.034 * jw, -0.058, 0.016), np.array([0.030, 0.030, 0.036]) * hs, bone="Head"),
            Ellipsoid(P(0, -0.083, 0.038), np.array([0.026, 0.021, 0.023]) * hs * jw, bone="Head"),         # chin
        ]
        feats = [
            RoundCone(P(0, 0.004, 0.078), P(0, -0.017, 0.087), 0.0060 * hs * nz, 0.0085 * hs * nz, bone="Head"),   # nose bridge
            Sphere(P(0, -0.022, 0.089), 0.0105 * hs * nz, bone="Head"),                                     # round tip
            Ellipsoid(P(0.0095 * nz, -0.026, 0.082), np.array([0.0072, 0.0058, 0.0066]) * hs * nz, bone="Head"),
            Ellipsoid(P(-0.0095 * nz, -0.026, 0.082), np.array([0.0072, 0.0058, 0.0066]) * hs * nz, bone="Head"),
            Ellipsoid(P(0, -0.0665, 0.0690), np.array([0.0135, 0.0042, 0.0050]) * hs, bone="Head", name="lip_lower"),
        ]
        ears = []
        for sgn in (1, -1):
            R = rot_from_axes(x=(sgn * 1.0, 0, 0.40), y=(0, 1, 0.10))
            ears.append(Ellipsoid(P(0.084 * sgn, -0.010, -0.008), np.array([0.0068, 0.023, 0.0160]) * hs, R=R, bone="Head", name="ear"))
        # big cartoon eyes
        self.eye_c = [P(0.034 * s, 0.001, 0.064) for s in (1, -1)]
        self.eye_r = 0.0168 * hs
        lids, socks = [], []
        self.eye_open = np.array([0.0172, 0.0100, 0.013]) * hs
        self.lid_r = self.eye_r + 0.0016 * hs
        for s, c in zip((1, -1), self.eye_c):
            ball = Sphere(c, self.lid_r)
            Ro = rot_from_axes(x=(1, 0.05 * s, 0), z=(0, 0, 1))
            oc = c + np.array([0.0006 * s, 0.0012 * hs, self.eye_r])
            opening = Ellipsoid(oc, self.eye_open, R=Ro)
            # the socket is carved only inside the lid opening, so no crease forms around the eye
            socks.append(Ellipsoid(oc, self.eye_open + 0.0006 * hs, R=Ro))
            front = HalfSpace((0, 0, -1), -(c[2] - 0.003))
            lids.append(Intersect([ball, Complement(opening), front], k=0.0012))
        sockets = Union(socks)
        self.lid_field = Union(lids)
        # mouth guide (used by build.py for the layered smile mesh)
        self.mouth = dict(xe=0.0215, yc=-0.0445, sag=0.0058, r_mid=0.0047, r_end=0.0010, z=0.080)
        self.lip_prims = [feats[4]]
        self.ear_prims = ears
        base = Union(skull, k=0.036)
        head = Union([base, Union(feats, k=0.005)], k=0.007)
        head = Union([head, Union(ears)], k=0.0045)
        head = Subtract(head, sockets, k=0.002)
        head = Union([head, self.lid_field], k=0.006)
        head.name = "head"
        self.hs = hs
        self.Hpt = P
        return head

    def tagged_prims(self):
        return [l for l in self.field.leaves() if getattr(l, "bone", None)]
