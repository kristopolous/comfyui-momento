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


def profile_card(path, shot, cell=None, caption=None, sub=None):
    im = Image.new("RGB", (W, H), (8, 8, 10))
    src = Image.open(shot).convert("RGB")
    d0 = ImageDraw.Draw(src)
    if cell is not None:
        x, y, r = PIC
        d0.ellipse((x - r - 4, y - r - 4, x + r + 4, y + r + 4), outline=ACCENT, width=5)
        left, top, cw, ch = GRID
        row, col = cell
        bx, by = left + col * cw, top + row * ch
        d0.rectangle((bx + 1, by + 1, bx + cw - 1, by + ch - 1), outline=ACCENT, width=6)
    crop = src.crop(SHOT_CROP)
    scale = (H - 40) / crop.height
    crop = crop.resize((round(crop.width * scale), H - 40), Image.LANCZOS)
    im.paste(crop, (120, 20))
    d = ImageDraw.Draw(im)
    tx = 120 + crop.width + 70
    if caption:
        d.text((tx, H // 2 - 50), caption, font=font("Bold", 40), fill=(245, 245, 245))
    if sub:
        d.text((tx, H // 2 + 6), sub, font=font("Light", 28), fill=ACCENT if cell is not None else (170, 170, 175))
    im.save(path)


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


def main(out, clips, shot=None):
    tmp = Path(tempfile.mkdtemp())
    white, grey, dim = (245, 245, 245), (170, 170, 175), (110, 110, 118)
    card(tmp / "title.png", [("MOMENTO", "Black", 120, white, 18),
                             ("Every photo has a moment.", "Light", 34, grey, 0)])
    card(tmp / "pitch.png", [("Give it a profile picture and a post.", "Medium", 40, white, 14),
                             ("It shows you the moment the photo was taken.", "Light", 34, grey, 0)])
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
    if shot:
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
            profile_card(tmp / f"pick{i}.png", shot, (row, col), caption="The profile picture", sub="+ this post")
            still_clip(str(tmp / f"pick{i}.png"), 2.0, str(tmp / f"pick{i}.mp4"), fade=0.15)
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
    shot = None
    if "--profile" in args:
        i = args.index("--profile")
        shot = args[i + 1]
        del args[i:i + 2]
    main(sys.argv[1], args, shot)
