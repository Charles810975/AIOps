# -*- coding: utf-8 -*-
"""Render agent trace markdown files into PNG screenshots for the PPT.

Uses Microsoft YaHei (msyh.ttc) for proper Chinese rendering.
"""
import os, sys, io
os.environ["PYTHONIOENCODING"] = "utf-8"

import subprocess
from pathlib import Path

try:
    import markdown
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "markdown", "Pillow"], check=True)
    import markdown
    from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"d:\刘从睿\软件测试与维护\Final\docs\screenshots")
DST = Path(r"d:\刘从睿\软件测试与维护\Final\docs\slides\figures")


def _pick_font(bold: bool = False, size: int = 18) -> ImageFont.FreeTypeFont:
    """Pick a Chinese-capable font (Microsoft YaHei), fall back to Consolas."""
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def md_to_pil(md_text: str, width: int = 1800) -> Image.Image:
    html = markdown.markdown(md_text, extensions=["fenced_code"])
    import re
    parts = re.split(r'(```[^\n]*\n.*?\n```)', html, flags=re.S)
    lines = []
    for p in parts:
        if p.startswith("```"):
            inner = re.sub(r'^```[^\n]*\n', '', p)
            inner = re.sub(r'\n```$', '', inner)
            inner = re.sub(r'<[^>]+>', '', inner)
            for l in inner.split("\n"):
                lines.append(("code", l))
        else:
            text = re.sub(r'<[^>]+>', '', p)
            for l in text.split("\n"):
                l = l.strip()
                if l:
                    lines.append(("text", l))
    font  = _pick_font(bold=False, size=20)
    fontb = _pick_font(bold=True,  size=22)
    fontc = _pick_font(bold=False, size=18)  # for code blocks (mono)
    line_h = 32
    pad = 30
    img_h = pad*2 + line_h * (len(lines) + 1)
    img = Image.new("RGB", (width, img_h), "white")
    d = ImageDraw.Draw(img)
    y = pad
    for kind, l in lines:
        if kind == "code":
            f = fontc
        elif l.startswith("#"):
            f = fontb
        else:
            f = font
        if len(l) > 130:
            l = l[:127] + "..."
        d.text((pad, y), l, font=f, fill="#1a1a1a")
        y += line_h
        if y > img_h - pad:
            break
    return img


for tag in ["S1-baseline", "S2-anomaly", "S3-selfheal"]:
    src = SRC / f"{tag}.md"
    if not src.exists():
        continue
    md = src.read_text(encoding="utf-8")
    img = md_to_pil(md, width=1800)
    if img.height > 2400:
        img = img.crop((0, 0, img.width, 2400))
    out = DST / f"agent_{tag.lower()}.png"
    img.save(out, "PNG", optimize=True)
    print(f"saved: {out}  ({img.size})")
