"""Cut the demo video: title, the real profile, then for each post the profile with that post highlighted and its
Momento output on a 16:9 stage, a how-it-works card, an end card.

    python scripts/demo.py out/demo.mp4 --profile shot.png out/post04.mp4:1,0 out/post02.mp4:0,1 ...
Each input is a Momento output; ":row,col" is where its post sits in the profile screenshot's grid.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
FONT = "/usr/share/fonts/truetype/montserrat/Montserrat-{}.ttf"


def font(weight, size):
    return ImageFont.truetype(FONT.format(weight), size)


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def card(path, lines):
    """lines: (text, weight, size, color, gap_after) centered vertically as a block."""
    im = Image.new("RGB", (W, H), (8, 8, 10))
    d = ImageDraw.Draw(im)
    heights = [d.textbbox((0, 0), t, font=font(w, s))[3] + g for t, w, s, c, g in lines]
    y = (H - sum(heights)) // 2
    for (t, w, s, c, g), h in zip(lines, heights):
        f = font(w, s)
        x = (W - d.textlength(t, font=f)) // 2
        d.text((x, y), t, font=f, fill=c)
        y += h
    im.save(path)


# Where things are in the profile screenshot (Instagram zoomed out so the profile and 12 posts fit)
SHOT_CROP = (540, 20, 1045, 885)
PIC = (655, 86, 36)                 # center x, center y, radius
GRID = (596, 175, 131, 174.3)       # left, top, cell width, cell height
ACCENT = (255, 196, 64)


def _ants(d, points, phase, dash=14, width=5):
    """Marching ants along a closed polyline: alternating dark/accent dashes, shifted by `phase` pixels."""
    pts = points + points[:1]
    t = -phase
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        length = max(1e-6, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        pos = 0.0
        while pos < length:
            k = int((t + pos) // dash)
            end = min(length, max(pos + 0.5, pos + dash - ((t + pos) % dash)))  # always advance (float rounding)
            color = ACCENT if k % 2 == 0 else (20, 20, 24)
            p0 = (x0 + (x1 - x0) * pos / length, y0 + (y1 - y0) * pos / length)
            p1 = (x0 + (x1 - x0) * end / length, y0 + (y1 - y0) * end / length)
            d.line([p0, p1], fill=color, width=width)
            pos = end
        t += length


def _circle(cx, cy, r, n=72):
    import math
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def profile_card(path, shot, cell=None, caption=None, sub=None, phase=0.0):
    """One frame of the profile screenshot. With `cell`, everything but the profile picture and that post is dimmed
    and both get marching ants (advance `phase` frame to frame to make them march)."""
    im = Image.new("RGB", (W, H), (8, 8, 10))
    src = Image.open(shot).convert("RGB").crop(SHOT_CROP)
    scale = (H - 40) / src.height
    src = src.resize((round(src.width * scale), H - 40), Image.LANCZOS)
    ox, oy = 120, 20
    to = lambda x, y: ((x - SHOT_CROP[0]) * scale + ox, (y - SHOT_CROP[1]) * scale + oy)  # noqa: E731
    im.paste(src, (ox, oy))
    if cell is not None:
        x, y, r = PIC
        left, top, cw, ch = GRID
        row, col = cell
        (bx0, by0), (bx1, by1) = to(left + col * cw, top + row * ch), to(left + (col + 1) * cw, top + (row + 1) * ch)
        (cx, cy), rr = to(x, y), (r + 3) * scale
        mask = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(mask)
        md.rectangle((bx0, by0, bx1, by1), fill=255)
        md.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=255)
        im = Image.composite(im, Image.eval(im, lambda v: int(v * 0.28)), mask)
        d = ImageDraw.Draw(im)
        _ants(d, [(bx0 - 4, by0 - 4), (bx1 + 4, by0 - 4), (bx1 + 4, by1 + 4), (bx0 - 4, by1 + 4)], phase)
        _ants(d, _circle(cx, cy, rr + 6), phase)
    d = ImageDraw.Draw(im)
    tx = ox + src.width + 70
    if caption:
        d.text((tx, H // 2 - 50), caption, font=font("Bold", 40), fill=(245, 245, 245))
    if sub:
        d.text((tx, H // 2 + 6), sub, font=font("Light", 28), fill=ACCENT if cell is not None else (170, 170, 175))
    im.save(path)


def ants_clip(shot, cell, seconds, out, **text):
    """The highlighted profile with the ants marching, as a clip."""
    frames = Path(tempfile.mkdtemp())
    for i in range(round(seconds * FPS)):
        profile_card(frames / f"{i:04d}.png", shot, cell, phase=i * 2.0, **text)
    ff("-framerate", str(FPS), "-i", str(frames / "%04d.png"), "-f", "lavfi", "-t", str(seconds),
       "-i", "anullsrc=r=48000:cl=stereo", "-vf", f"setsar=1,format=yuv420p,fade=in:st=0:d=0.15",
       "-c:v", "libx264", "-c:a", "aac", "-shortest", out)


def still_clip(png, seconds, out, fade=0.4):
    ff("-loop", "1", "-t", str(seconds), "-i", png, "-f", "lavfi", "-t", str(seconds),
       "-i", "anullsrc=r=48000:cl=stereo", "-vf",
       f"fps={FPS},setsar=1,format=yuv420p,fade=in:st=0:d={fade},fade=out:st={seconds - fade}:d={fade}",
       "-c:v", "libx264", "-c:a", "aac", "-shortest", out)


def stage(clip, out):
    """The vertical clip, full height, over a blurred and darkened copy of itself."""
    ff("-i", clip, "-filter_complex",
       f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:3,"
       f"eq=brightness=-0.25[bg];[b]scale=-2:{H}:flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:0,fps={FPS},setsar=1,format=yuv420p",
       "-af", "aresample=48000,aformat=channel_layouts=stereo", "-c:v", "libx264", "-c:a", "aac", out)


def ui_segment(take, marks, out):
    """A screen recording of the page: real speed until `fast`, fast-forward (with a badge) to `slow`, real speed to
    `end`. marks = (fast, slow, end) in seconds."""
    fast, slow, end = marks
    tmp = Path(tempfile.mkdtemp())
    fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={FPS}"
    badge = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(badge)
    label = f"▶▶  {round(slow - fast) // 60} min {round(slow - fast) % 60:02d} s, sped up"
    f = font("Bold", 30)
    tw = d.textlength(label, font=f)
    d.rounded_rectangle((W - tw - 70, 30, W - 30, 86), 14, fill=(0, 0, 0, 200))
    d.text((W - tw - 50, 41), label, font=f, fill=ACCENT)
    badge.save(tmp / "badge.png")
    speed = (slow - fast) / 6.0
    ff("-ss", "0", "-to", str(fast), "-i", take, "-vf", fit, "-an", "-c:v", "libx264", str(tmp / "a.mp4"))
    ff("-ss", str(fast), "-to", str(slow), "-i", take, "-i", str(tmp / "badge.png"), "-filter_complex",
       f"[0:v]setpts=PTS/{speed},{fit}[v];[v][1:v]overlay=0:0", "-an", "-c:v", "libx264", str(tmp / "b.mp4"))
    ff("-ss", str(slow), "-to", str(end), "-i", take, "-vf", fit, "-an", "-c:v", "libx264", str(tmp / "c.mp4"))
    parts = [tmp / n for n in ("a.mp4", "b.mp4", "c.mp4")]
    inputs = sum((["-i", str(p)] for p in parts), [])
    seconds = fast + 6.0 + (end - slow)
    ff(*inputs, "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=48000:cl=stereo", "-filter_complex",
       "".join(f"[{i}:v]" for i in range(3)) + "concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out)


def main(out, clips, shot=None, ui=None):
    tmp = Path(tempfile.mkdtemp())
    white, grey, dim = (245, 245, 245), (170, 170, 175), (110, 110, 118)
    card(tmp / "title.png", [("MOMENTO", "Black", 120, white, 18),
                             ("Every photo has a moment.", "Light", 34, grey, 0)])
    card(tmp / "pitch.png", [("Type an Instagram name. Tap a post.", "Medium", 40, white, 14),
                             ("See the moment the photo was taken.", "Light", 34, grey, 0)])
    card(tmp / "how.png", [("ONE COMFYUI GRAPH", "Bold", 26, dim, 30),
                           ("Gemini 3.1 Pro plans the moment", "Medium", 36, white, 12),
                           ("FLUX 3 Klein builds the first frame", "Medium", 36, white, 12),
                           ("LTX-2.5 films it, with sound", "Medium", 36, white, 12),
                           ("MiniMax Music 3 scores it", "Medium", 36, white, 30),
                           ("open weights + the Comfy API  ·  runs as a Comfy App", "Light", 26, grey, 0)])
    card(tmp / "end.png", [("MOMENTO", "Black", 96, white, 20),
                           ("github.com/kristopolous/comfyui-momento", "Medium", 32, grey, 0)])
    parts = []
    for name, secs in (("title", 2.6), ("pitch", 3.0)):
        still_clip(str(tmp / f"{name}.png"), secs, str(tmp / f"{name}.mp4"))
        parts.append(tmp / f"{name}.mp4")
    if ui:
        take, result, marks = ui
        ui_segment(take, marks, str(tmp / "ui.mp4"))
        stage(result, str(tmp / "ui_result.mp4"))
        parts += [tmp / "ui.mp4", tmp / "ui_result.mp4"]
    if shot and not ui:
        profile_card(tmp / "profile.png", shot, caption="A real Instagram profile.",
                     sub="One profile picture. A grid of posts.")
        still_clip(str(tmp / "profile.png"), 3.2, str(tmp / "profile.mp4"))
        parts.append(tmp / "profile.mp4")
    staged = []
    for i, c in enumerate(clips):
        c, _, where = c.partition(":")
        seg = []
        if shot and where:
            row, col = (int(v) for v in where.split(","))
            ants_clip(shot, (row, col), 2.4, str(tmp / f"pick{i}.mp4"), caption="The profile picture",
                      sub="+ this post")
            seg.append(tmp / f"pick{i}.mp4")
        stage(c, str(tmp / f"clip{i}.mp4"))
        staged.append(seg + [tmp / f"clip{i}.mp4"])
    half = max(1, len(staged) // 2)
    still_clip(str(tmp / "how.png"), 4.5, str(tmp / "how.mp4"))
    still_clip(str(tmp / "end.png"), 3.5, str(tmp / "end.mp4"))
    parts += sum(staged[:half], []) + [tmp / "how.mp4"] + sum(staged[half:], []) + [tmp / "end.mp4"]
    inputs = sum((["-i", str(p)] for p in parts), [])
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(parts)))
    ff(*inputs, "-filter_complex", f"{streams}concat=n={len(parts)}:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]",
       "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
       "-movflags", "+faststart", out)
    print(out)


if __name__ == "__main__":
    args = sys.argv[2:]
    shot = ui = None
    if "--profile" in args:
        i = args.index("--profile")
        shot = args[i + 1]
        del args[i:i + 2]
    if "--ui" in args:  # --ui take.mp4 result.mp4 fast,slow,end
        i = args.index("--ui")
        ui = (args[i + 1], args[i + 2], tuple(float(v) for v in args[i + 3].split(",")))
        del args[i:i + 4]
    main(sys.argv[1], args, shot, ui)
