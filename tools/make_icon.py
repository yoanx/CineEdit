"""生成应用图标：深色圆角底 + 带运动拖影的亮点。"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
S = 512


def rounded(size: int, radius: int, color):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=color)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    base = rounded(S, int(S * 0.22), (17, 21, 29, 255))

    # 拖影：沿水平方向的多个半透明椭圆
    streak = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(streak)
    cy = int(S * 0.5)
    for i in range(14):
        f = i / 13.0
        w = int(S * 0.10 * (0.55 + 0.45 * f))
        h = int(S * 0.30 * (0.35 + 0.65 * f))
        cx = int(S * (0.30 + 0.34 * f))
        a = int(26 + 150 * (1 - abs(f - 0.75) / 0.75))
        d.ellipse([cx - w, cy - h, cx + w, cy + h], fill=(61, 126, 255, max(0, a)))
    streak = streak.filter(ImageFilter.GaussianBlur(S * 0.018))
    base = Image.alpha_composite(base, streak)

    # 主光点
    dot = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(dot)
    r = int(S * 0.135)
    cx, cy = int(S * 0.68), int(S * 0.5)
    d.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6], fill=(0, 212, 184, 70))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))
    dot = dot.filter(ImageFilter.GaussianBlur(1.5))
    base = Image.alpha_composite(base, dot)

    base.save(os.path.join(OUT, "icon.png"))
    base.save(os.path.join(OUT, "icon.ico"),
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icon ->", OUT)


if __name__ == "__main__":
    main()
