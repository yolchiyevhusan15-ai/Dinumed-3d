"""Procedural texture generation (PNG)."""
import io
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu"]


def _font(size, bold=True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def iris(color, size=256, seed=1):
    """Eye texture for a UV sphere whose v = polar angle / pi (pole = pupil)."""
    rng = np.random.default_rng(seed)
    H, Wd = size, size
    v = (np.arange(H) + 0.5) / H          # 0 at pupil pole
    u = (np.arange(Wd) + 0.5) / Wd
    V, U = np.meshgrid(v, u, indexing="ij")
    th = V * np.pi
    img = np.zeros((H, Wd, 3))
    sclera = np.array([0.93, 0.91, 0.88])
    img[:] = sclera
    # faint veins near the back
    veins = (np.sin(U * 2 * np.pi * 23 + rng.uniform(0, 6)) * np.sin(V * 40) > 0.97) & (th > 0.9)
    img[veins] = [0.85, 0.6, 0.6]
    iris_r, pupil_r = 0.52, 0.20
    col = np.asarray(color, float)
    streak = 0.75 + 0.25 * np.sin(U * 2 * np.pi * 60 + rng.uniform(0, 6)) * np.sin(U * 2 * np.pi * 17)
    t = np.clip(th / iris_r, 0, 1)
    iris_col = col[None, None, :] * (streak[..., None] * (0.7 + 0.5 * t[..., None]))
    m_iris = th < iris_r
    img[m_iris] = iris_col[m_iris]
    ring = (th > iris_r - 0.05) & (th < iris_r)
    img[ring] *= 0.55
    img[th < pupil_r] = [0.03, 0.03, 0.035]
    # soft limbus blend
    blend = np.clip((th - iris_r) / 0.06, 0, 1)[..., None]
    m_bl = (th >= iris_r) & (th < iris_r + 0.06)
    img[m_bl] = (img * blend + col * 0.35 * (1 - blend))[m_bl]
    # catch-light
    hl = ((th - 0.30) ** 2 * 60 + (U - 0.35) ** 2 * 80) < 0.02
    img[hl & (th < iris_r)] = [0.95, 0.95, 0.95]
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def id_card(gender):
    W, H = 512, 816                     # CR80 portrait ratio 54 x 86 mm
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    teal = (0, 122, 140)
    d.rectangle([0, 0, W, 150], fill=teal)
    d.text((W // 2, 60), "DINUMED", font=_font(72), fill="white", anchor="mm")
    d.text((W // 2, 120), "MEDICAL STAFF", font=_font(30, False), fill=(220, 245, 248), anchor="mm")
    # photo frame with silhouette
    px0, py0, px1, py1 = 136, 190, 376, 470
    d.rectangle([px0, py0, px1, py1], fill=(222, 233, 238), outline=(150, 170, 180), width=4)
    cx = (px0 + px1) // 2
    d.ellipse([cx - 62, 230, cx + 62, 368], fill=(120, 140, 150))
    d.pieslice([cx - 110, 360, cx + 110, 560], 180, 360, fill=(120, 140, 150))
    d.rectangle([px0 - 2, py1 + 2, px1 + 2, py1 + 120], fill="white")
    d.text((W // 2, 530), "DOCTOR", font=_font(56), fill=(30, 30, 30), anchor="mm")
    d.text((W // 2, 590), "SHIFOKOR", font=_font(34, False), fill=(70, 70, 70), anchor="mm")
    # red cross emblem
    d.rectangle([40, 640, 120, 660], fill=(200, 30, 40))
    d.rectangle([70, 610, 90, 690], fill=(200, 30, 40))
    # barcode (deterministic)
    rng = np.random.default_rng(7 if gender == "male" else 8)
    x = 160
    while x < 470:
        w = int(rng.integers(2, 7))
        d.rectangle([x, 615, x + w, 690], fill=(20, 20, 20))
        x += w + int(rng.integers(2, 6))
    d.rectangle([0, H - 60, W, H], fill=teal)
    d.text((W // 2, H - 30), "dinumed", font=_font(28, False), fill="white", anchor="mm")
    # slot punch hole at top is geometric; draw a hint
    return im


def watch_face(size=256):
    im = Image.new("RGB", (size, size), (18, 22, 28))
    d = ImageDraw.Draw(im)
    c = size / 2
    d.ellipse([4, 4, size - 4, size - 4], fill=(245, 245, 242))
    for i in range(60):
        a = np.radians(i * 6)
        r0 = c - (24 if i % 5 == 0 else 14)
        r1 = c - 8
        w = 5 if i % 5 == 0 else 2
        d.line([(c + r0 * np.sin(a), c - r0 * np.cos(a)), (c + r1 * np.sin(a), c - r1 * np.cos(a))],
               fill=(30, 30, 30), width=w)
    d.text((c, c + 40), "DINUMED", font=_font(18), fill=(0, 122, 140), anchor="mm")
    def hand(angle_deg, length, width, col):
        a = np.radians(angle_deg)
        d.line([(c, c), (c + length * np.sin(a), c - length * np.cos(a))], fill=col, width=width)
    hand(10 * 30 + 5, c * 0.5, 8, (20, 20, 20))      # 10:10
    hand(60, c * 0.72, 5, (20, 20, 20))
    hand(200, c * 0.8, 2, (200, 30, 40))
    d.ellipse([c - 7, c - 7, c + 7, c + 7], fill=(20, 20, 20))
    return im
