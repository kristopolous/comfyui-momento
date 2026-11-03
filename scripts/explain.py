"""Cut the how-it-works section of the demo: one take, shown stage by stage, with the narration.

    python scripts/explain.py ASSETS_DIR out.mp4

ASSETS_DIR holds one controlled take's pieces (local.py with CONTROL=1 EXPLAIN=1 saves them): post.jpg, depth.webp
(Momento Camera Path), pose.mp4 (Momento Photographer), place.mp4 (pass 1), final.mp4 (pass 2), body3.png (SAM 3D
Body's view), gate.json (Momento Gate's numbers), a prompted take for contrast (prompted.mp4), canvas screenshots
(camera_path_closeup.jpg, gate_closeup.jpg, full_graph.jpg) and the narration, explain_1..5.wav.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

W, H, FPS = 1280, 720, 24
BG, WHITE, GREY, DIM = (8, 8, 10), (245, 245, 245), (170, 170, 175), (110, 110, 118)
GOLD, BLUE, PURPLE, GREEN, RED = (184, 134, 11), (47, 111, 176), (138, 68, 168), (46, 139, 87), (220, 60, 60)
FONT = "/usr/share/fonts/truetype/montserrat/Montserrat-{}.ttf"
SYMBOLS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PAD = 0.5  # seconds of picture after each line


def font(weight, size):
    return ImageFont.truetype(FONT.format(weight), size)


def mark(d, xy, ok, size):
    """A tick or a cross (Montserrat has neither glyph)."""
    d.text(xy, "✓" if ok else "✗", font=ImageFont.truetype(SYMBOLS, size), fill=GREEN if ok else RED)


def frames(path, n=None):
    """A video (or animated webp) as a list of PIL frames."""
    path = Path(path)
    if path.suffix == ".webp":
        return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return [Image.fromarray(a) for a in arr[:n]]


def at(clip, t, start=0.3):
    """The clip's frame at scene time t: plays once from `start`, then holds its last frame."""
    i = int(max(0.0, t - start) * FPS)
    return clip[min(i, len(clip) - 1)]


def fit(im, w, h):
    im = im.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    return im


def panel(canvas, im, box, label=None, color=DIM, border=None, sub=None):
    """Paste `im` centred in box (x, y, w, h), with a caption under it."""
    x, y, w, h = box
    im = fit(im, w, h)
    px, py = x + (w - im.width) // 2, y + (h - im.height) // 2
    canvas.paste(im, (px, py))
    d = ImageDraw.Draw(canvas)
    if border:
        d.rectangle((px - 3, py - 3, px + im.width + 2, py + im.height + 2), outline=border, width=4)
    yy = py + im.height + 12
    for text, f, c in ((label, font("Bold", 17), color), (sub, font("Medium", 15), GREY)):
        if text:
            for line in text.split("\n"):
                tw = d.textlength(line, font=f)
                d.text((x + (w - tw) / 2, yy), line, font=f, fill=c)
                yy += f.size + 5


def title(canvas, text, sub=None):
    d = ImageDraw.Draw(canvas)
    d.text((48, 30), text, font=font("Bold", 30), fill=WHITE)
    if sub:
        d.text((48, 72), sub, font=font("Medium", 18), fill=GREY)


def duration(wav):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                         capture_output=True, text=True).stdout
    return float(out) + PAD


def main(src, out):
    src = Path(src)
    gate = json.loads((src / "gate.json").read_text())
    last = gate[-1]
    clip = {k: frames(src / f) for k, f in (("prompted", "prompted.mp4"), ("depth", "depth.webp"),
                                            ("pose", "pose.mp4"), ("place", "place.mp4"), ("final", "final.mp4"))}
    post = Image.open(src / "post.jpg").convert("RGB")
    body = Image.open(src / "body3.png").convert("RGB")
    shots = {k: Image.open(src / f"{k}.jpg").convert("RGB") for k in ("camera_path_closeup", "gate_closeup",
                                                                      "full_graph")}
    # the prompted take's measured miss, from the same gate (side-on, 25% of the frame off where it should be)
    prompted_note = "measured: 47° off the shot,\nstanding 25% of the frame away"

    def s1(t):
        c = Image.new("RGB", (W, H), BG)
        title(c, "Prompts can't place a person.", "a video model asked for “a photographer behind the camera”")
        panel(c, at(clip["prompted"], t), (140, 120, 360, 520), "PROMPTED", RED, RED, prompted_note)
        mark(ImageDraw.Draw(c), (430, 540), False, 64)
        d = ImageDraw.Draw(c)
        for i, line in enumerate(("The lens points at nothing.", "Turned side-on.", "Not where the camera stood.")):
            d.text((600, 250 + i * 56), line, font=font("Medium", 34), fill=WHITE if i == 0 else GREY)
        return c

    def s2(t):
        c = Image.new("RGB", (W, H), BG)
        title(c, "Rebuild the photo in 3D. Fly the camera through it.")
        panel(c, post, (40, 130, 300, 440), "THE POST", DIM)
        panel(c, at(clip["depth"], t), (350, 120, 280, 470), "DEPTH ANYTHING 3", BLUE,
              sub="the place + the ground,\nalong the dolly")
        panel(c, shots["camera_path_closeup"], (650, 140, 600, 380), "MOMENTO CAMERA PATH → MINIMAX H3", GOLD,
              sub="Fun ControlNet-Union 2.0 follows the depth")
        return c

    def s3(t):
        c = Image.new("RGB", (W, H), BG)
        title(c, "The photographer comes from the same geometry.")
        boxes = [(60, 110, 360, 500), (460, 110, 360, 500), (860, 110, 360, 500)]
        bones = at(clip["pose"], t)
        over = Image.composite(bones, at(clip["place"], t).resize(bones.size),
                               bones.convert("L").point(lambda v: 255 if v > 24 else 0))
        panel(c, over, boxes[0], "MOMENTO PHOTOGRAPHER", GOLD, sub="3D skeleton, projected:\nback to us, camera up")
        panel(c, at(clip["place"], t), boxes[1], "PASS 1 · MINIMAX H3", PURPLE, sub="the place, filmed\nalong the path")
        panel(c, at(clip["final"], t), boxes[2], "PASS 2 · H3 REF2VA", PURPLE, sub="painted in from a\nFLUX 3 character sheet")
        return c

    def s4(t):
        c = Image.new("RGB", (W, H), BG)
        title(c, "Then measure every take.")
        panel(c, body, (40, 110, 330, 520), "SAM 3D BODY", GREEN, sub="the body it reconstructs")
        d = ImageDraw.Draw(c)
        rows = [(f"{last['deg']:.0f}°", "off the shot's axis"), (f"+{last['hands_up_m'] * 100:.0f} cm", "camera at the eye"),
                (f"{last['off'] * 100:.1f}%", "from where the 3D put them")]
        for i, (big, small) in enumerate(rows):
            d.text((410, 170 + i * 92), big, font=font("Black", 44), fill=WHITE)
            d.text((410, 222 + i * 92), small, font=font("Medium", 18), fill=GREY)
        mark(d, (410, 452), True, 44)
        d.text((462, 460), "SHIPS", font=font("Black", 40), fill=GREEN)
        d.text((410, 512), "otherwise it's re-rolled", font=font("Medium", 18), fill=GREY)
        panel(c, shots["gate_closeup"], (700, 150, 560, 400), "MOMENTO GATE", GOLD)
        return c

    def s5(t):
        c = Image.new("RGB", (W, H), BG)
        title(c, "One ComfyUI graph.", "70 nodes · open weights · runs as a Comfy App")
        panel(c, shots["full_graph"], (40, 110, 1200, 520))
        d = ImageDraw.Draw(c)
        x = 60
        for color, name in ((BLUE, "Depth Anything 3"), (GOLD, "Momento nodes"), (PURPLE, "MiniMax H3 + Fun ControlNet"),
                            (GREEN, "SAM 3D Body")):
            d.rectangle((x, 668, x + 18, 686), fill=color)
            d.text((x + 26, 666), name, font=font("Medium", 18), fill=WHITE)
            x += 40 + int(d.textlength(name, font=font("Medium", 18)))
        return c

    tmp = Path(tempfile.mkdtemp())
    parts = []
    for i, scene in enumerate((s1, s2, s3, s4, s5), 1):
        wav = src / f"explain_{i}.wav"
        secs = duration(wav) + (1.0 if i == 5 else 0.0)
        n = int(secs * FPS)
        for f in range(n):
            scene(f / FPS).save(tmp / f"f{f:05d}.png")
        part = tmp / f"s{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"), "-i",
                        str(wav), "-af", f"apad=whole_dur={secs}", "-t", f"{secs:.3f}", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "16", "-c:a", "aac", "-ar", "48000", str(part)], check=True)
        for p in tmp.glob("f*.png"):
            p.unlink()
        parts.append(part)
    lst = tmp / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", out],
                   check=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
