"""Momento: give it someone's profile picture and one of their posts, get the moment that photo was taken.

Graph:  profile + post -> Gemini (Comfy API) writes a bystander's view of the moment -> Momento Plan
        -> FLUX 3 Klein builds that frame from the post's objects and the person -> LTX-2.5 animates the shot
        being taken (ambient sound only) -> Momento Reveal cuts to the real post.
Everything defaults to the LOWEST resolution.
"""
import json
import re

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, InputImpl, Types, io, ui

from .director import MUSIC, QUIET, VIDEO_NEGATIVE

WEB_DIRECTORY = "./web"

try:  # the Instagram-style page and its routes; only when running inside ComfyUI's server
    from . import server  # noqa: F401
except ImportError:
    pass




def _plan(text):
    """Gemini's reply -> {"still", "video"}; tolerates code fences and stray prose around the JSON."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"Gemini didn't return a plan: {text[:500]}")
    plan = json.loads(m.group(0))
    if not plan.get("still") or not plan.get("video"):
        raise ValueError(f"Gemini's plan is missing a field: {plan}")
    return plan


def _layout(boxes):
    """Gemini's boxes -> FLUX 3 layout rows ([y_min, x_min, y_max, x_max] on 0-1000), dropping malformed ones."""
    def box(b):
        ok = isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
        return [max(0, min(1000, round(v))) for v in b] if ok and b[2] > b[0] and b[3] > b[1] else None
    rows = []
    for r in boxes if isinstance(boxes, list) else []:
        if not isinstance(r, dict) or not r.get("id") or not box(r.get("tgt_bbox")):
            continue
        src = r.get("from") if str(r.get("from", "")).startswith("ref_image_") else "target"
        rows.append({"id": str(r["id"]), "from": src, "src_bbox": box(r.get("src_bbox")) if src != "target" else None,
                     "tgt_bbox": box(r["tgt_bbox"]), "desc": str(r.get("desc", ""))})
    return rows


def _encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


class MomentoPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoPlan",
            display_name="Momento Plan",
            category="momento",
            description="Read Gemini's plan and encode it: the still for FLUX 3 Klein, the motion for LTX-2.5. "
                        "Shows the plan in the app.",
            is_output_node=True,
            inputs=[
                io.String.Input("plan", force_input=True, tooltip="Gemini's JSON reply."),
                io.Clip.Input("image_clip", tooltip="FLUX 3 Klein's text encoder."),
                io.Clip.Input("video_clip", tooltip="LTX-2.5's text encoder."),
                io.String.Input("scene_box", force_input=True, optional=True,
                                tooltip="Where the post sits in the frame (Momento Canvas), added to the layout."),
            ],
            outputs=[io.Conditioning.Output(display_name="still"), io.Conditioning.Output(display_name="still_negative"),
                     io.Conditioning.Output(display_name="video"), io.Conditioning.Output(display_name="video_negative"),
                     io.String.Output(display_name="music"), io.Conditioning.Output(display_name="before")],
        )

    @classmethod
    def execute(cls, plan, image_clip, video_clip, scene_box=None) -> io.NodeOutput:
        p = _plan(plan)
        rows = _layout(p.get("boxes"))
        if scene_box:
            rows.append({"id": "scene", "from": "ref_image_0", "src_bbox": [0, 0, 1000, 1000],
                         "tgt_bbox": json.loads(scene_box),
                         "desc": "the post's scene, its subject in exactly the post's pose, seen from here as part of "
                                 "the real place (not a picture or a screen)"})
        if rows:  # FLUX 3 reads its layout as a JSON table after the prompt
            p["still"] = f"{p['still'].rstrip()} {json.dumps(rows)}"
        video = p["video"] + QUIET
        music = p.get("music") or MUSIC
        return io.NodeOutput(_encode(image_clip, p["still"]), _encode(image_clip, ""),
                             _encode(video_clip, video), _encode(video_clip, VIDEO_NEGATIVE), music,
                             _encode(image_clip, p.get("before") or p["still"]),
                             ui=ui.PreviewText(f"THE SHOT\n{p['still']}\n\nTHE MOMENT\n{video}\n\nTHE MUSIC\n{music}"))


class MomentoReveal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoReveal",
            display_name="Momento Reveal",
            category="momento",
            description="The moment, then the shutter flash, then the real post, letterboxed to the same frame.",
            inputs=[
                io.Video.Input("moment"),
                io.Image.Input("post"),
                io.Float.Input("hold", default=1.8, min=0.5, max=6.0, step=0.1, tooltip="Seconds the post stays up."),
                io.Float.Input("flash", default=0.25, min=0.0, max=1.0, step=0.05, tooltip="White flash fade, seconds."),
                io.Audio.Input("music", optional=True, tooltip="Soundtrack, mixed under the moment's own sound."),
                io.Float.Input("music_level", default=0.35, min=0.0, max=1.0, step=0.05, tooltip="Soundtrack volume."),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, moment, post, hold, flash, music=None, music_level=0.35) -> io.NodeOutput:
        c = moment.get_components()
        frames, fps = c.images, float(c.frame_rate)
        _, h, w, _ = frames.shape
        img = post[:1].movedim(-1, 1)
        scale = min(w / img.shape[3], h / img.shape[2])
        nh, nw = max(1, round(img.shape[2] * scale)), max(1, round(img.shape[3] * scale))
        img = F.interpolate(img, size=(nh, nw), mode="area").movedim(1, -1)[0].to(frames)
        card = torch.ones((h, w, 3), dtype=frames.dtype, device=frames.device)
        y, x = (h - nh) // 2, (w - nw) // 2
        card[y:y + nh, x:x + nw] = img
        n = max(1, round(hold * fps))
        white = torch.clamp(1 - torch.arange(n, device=frames.device) / max(1.0, flash * fps), 0, 1).view(n, 1, 1, 1)
        tail = card.expand(n, h, w, 3) * (1 - white) + white
        audio = c.audio
        if audio is not None:
            wave, rate = audio["waveform"], audio["sample_rate"]
            pad = torch.zeros((*wave.shape[:-1], round(n / fps * rate)), dtype=wave.dtype, device=wave.device)
            audio = {"waveform": torch.cat([wave, pad], -1), "sample_rate": rate}
        if music is not None:
            audio = _mix(audio, music, music_level, (len(frames) + n) / fps)
        return io.NodeOutput(InputImpl.VideoFromComponents(
            Types.VideoComponents(images=torch.cat([frames, tail]), audio=audio, frame_rate=c.frame_rate)))


def _mix(base, music, level, seconds, fade=0.8):
    """Lay `music` under `base` (or alone), trimmed to `seconds` with a fade-out."""
    import torchaudio
    rate = base["sample_rate"] if base is not None else music["sample_rate"]
    m = music["waveform"][:1].float()
    if music["sample_rate"] != rate:
        m = torchaudio.functional.resample(m, music["sample_rate"], rate)
    length = round(seconds * rate)
    m = F.pad(m, (0, max(0, length - m.shape[-1])))[..., :length]
    ramp = torch.clamp(torch.arange(length, dtype=m.dtype) / (fade * rate), 0, 1).flip(0)
    m = m * level * ramp
    if base is None:
        return {"waveform": m, "sample_rate": rate}
    b = base["waveform"].float()
    b = F.pad(b, (0, max(0, length - b.shape[-1])))[..., :length]
    if m.shape[1] != b.shape[1]:
        m = m.mean(1, keepdim=True).expand(-1, b.shape[1], -1)
    return {"waveform": torch.clamp(b + m.to(b.device), -1, 1), "sample_rate": rate}


class MomentoFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoFit",
            display_name="Momento Fit",
            category="momento",
            description="Scale and center-crop an image to fill width x height: the post as the phone's view, for the "
                        "video's last keyframe.",
            inputs=[io.Image.Input("image"), io.Int.Input("width", default=288, min=32, max=4096),
                    io.Int.Input("height", default=512, min=32, max=4096)],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, width, height) -> io.NodeOutput:
        img = image[:1].movedim(-1, 1)
        scale = max(width / img.shape[3], height / img.shape[2])
        nh, nw = max(height, round(img.shape[2] * scale)), max(width, round(img.shape[3] * scale))
        img = F.interpolate(img, size=(nh, nw), mode="bicubic", align_corners=False).clamp(0, 1)
        y, x = (nh - height) // 2, (nw - width) // 2
        return io.NodeOutput(img[:, :, y:y + height, x:x + width].movedim(1, -1))


class MomentoCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoCanvas",
            display_name="Momento Canvas",
            category="momento",
            description="The post, whole and at its own shape, placed in a bigger frame for outpainting: the wider "
                        "view of the same place. mask is 1 where the image generator paints (the surroundings, and "
                        "the post's lower part, where the photographer stands in front of it).",
            inputs=[io.Image.Input("post"), io.Int.Input("width", default=288, min=32, max=4096),
                    io.Int.Input("height", default=512, min=32, max=4096),
                    io.Float.Input("scale", default=0.62, min=0.1, max=1.0, tooltip="post width / frame width"),
                    io.Float.Input("center_y", default=0.4, min=0.0, max=1.0),
                    io.Float.Input("occlude", default=0.3, min=0.0, max=1.0,
                                   tooltip="share of the post, from its bottom, the photographer may cover"),
                    io.Int.Input("feather", default=10, min=0, max=128)],
            outputs=[io.Image.Output(display_name="canvas"), io.Mask.Output(display_name="mask"),
                     io.String.Output(display_name="box", tooltip="Where the post sits: [y_min, x_min, y_max, x_max] "
                                                                   "on FLUX 3's 0-1000 grid.")],
        )

    @classmethod
    def execute(cls, post, width, height, scale, center_y, occlude, feather) -> io.NodeOutput:
        img = post[:1].movedim(-1, 1)
        pw = round(width * scale)
        ph = min(round(img.shape[2] * pw / img.shape[3]), height)
        img = F.interpolate(img, size=(ph, pw), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (width - pw) // 2
        y = min(max(0, round(height * center_y - ph / 2)), height - ph)
        canvas = torch.full((1, 3, height, width), 0.5)
        canvas[:, :, y:y + ph, x:x + pw] = img
        keep = torch.zeros((1, 1, height, width))
        keep[:, :, y:y + ph - round(ph * occlude), x:x + pw] = 1.0
        if feather:
            k = feather * 2 + 1
            keep = F.avg_pool2d(F.pad(keep, (feather,) * 4, mode="replicate"), k, stride=1) * keep
        box = [round(v * 1000) for v in (y / height, x / width, (y + ph) / height, (x + pw) / width)]
        return io.NodeOutput(canvas.movedim(1, -1), (1.0 - keep)[:, 0], json.dumps(box))


class MomentoPaste(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoPaste",
            display_name="Momento Paste",
            category="momento",
            description="Put the real post's pixels back into the outpainted frame, so the photo itself is in the scene.",
            inputs=[io.Image.Input("image"), io.Image.Input("canvas"), io.Mask.Input("mask")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, canvas, mask) -> io.NodeOutput:
        img = image[:1]
        if img.shape[1:3] != canvas.shape[1:3]:
            img = F.interpolate(img.movedim(-1, 1), size=canvas.shape[1:3], mode="bicubic",
                                align_corners=False).clamp(0, 1).movedim(1, -1)
        m = mask[:1, :, :, None].to(img)
        return io.NodeOutput(img * m + canvas[:1].to(img) * (1 - m))


def review(text):
    """The critic's reply -> {"pass", "score", "problem"}; a reply that can't be read passes."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        r = json.loads(m.group(0)) if m else {}
    except ValueError:
        r = {}
    return {"pass": bool(r.get("pass", True)), "score": r.get("score"), "problem": str(r.get("problem") or "")}


class MomentoReview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoReview",
            display_name="Momento Review",
            category="momento",
            description="Show the critic's verdict on the take. The page reshoots a rejected take with a new seed.",
            is_output_node=True,
            inputs=[io.String.Input("verdict", force_input=True, tooltip="Gemini's JSON reply.")],
            outputs=[],
        )

    @classmethod
    def execute(cls, verdict) -> io.NodeOutput:
        r = review(verdict)
        return io.NodeOutput(ui=ui.PreviewText("REVIEW " + json.dumps(r)))


class MomentoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MomentoPlan, MomentoReveal, MomentoFit, MomentoCanvas, MomentoPaste, MomentoReview]


async def comfy_entrypoint() -> MomentoExtension:
    return MomentoExtension()
