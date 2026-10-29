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

from .director import DIRECTOR, QUIET, VIDEO_NEGATIVE  # noqa: F401

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
            ],
            outputs=[io.Conditioning.Output(display_name="still"), io.Conditioning.Output(display_name="still_negative"),
                     io.Conditioning.Output(display_name="video"), io.Conditioning.Output(display_name="video_negative")],
        )

    @classmethod
    def execute(cls, plan, image_clip, video_clip) -> io.NodeOutput:
        p = _plan(plan)
        video = p["video"] + QUIET
        return io.NodeOutput(_encode(image_clip, p["still"]), _encode(image_clip, ""),
                             _encode(video_clip, video), _encode(video_clip, VIDEO_NEGATIVE),
                             ui=ui.PreviewText(f"THE SHOT\n{p['still']}\n\nTHE MOMENT\n{video}"))


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


class MomentoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MomentoPlan, MomentoReveal]


async def comfy_entrypoint() -> MomentoExtension:
    return MomentoExtension()
