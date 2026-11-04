"""The Comfy Cloud pipeline that made the approved v2 clips (scripts/moment.py), on your own GPU.

    COMFY_URL=... PLANNER_URL=... PLANNER_MODEL=... python scripts/moment_local.py profile.jpg post.jpg [more posts...]

Same steps as moment.py; only the hosted nodes are replaced: FLUX 3 Image -> FLUX 3 Klein, Gemini -> the local planner.
  0. Framing: the bystander tilts up until the post covers the bottom of the frame (measured on the 3D render),
     so nothing below the photo has to be invented (that strip is where a sidewalk grew halfway up the bus).
  A. Plan; character sheet; the instant (the post's 3D place seen from the bystander, gaps filled seamlessly).
  B. MiniMax H3 films the scene with no photographer, camera nearly still, ending on the instant.
  C. Klein places the photographer into the scene's last frame, in the box projected from 3D; SAM 3 cuts them out
     onto the first frame, so both ends hold the same phone (two placements gave two phones, morphed mid-clip).
  D. LTX-2.5 + the in/outpainting IC-LoRA regenerates only the photographer's region (a mask projected from 3D)
     through the whole clip, pinned to the placed first and last frames; the rest stays H3's.
The camera dollies in (MOVE="dolly in", the default; MOVE=still holds it): it opens further back and higher, looking
level, and pushes in, tilting up to the framed end. The photographer is placed once, whole, in that wide opening and
carried to the close end by the 3D projection; H3 films the move between the two (no inpaint pass: the photographer
moves with the camera, which a fixed mask can't follow).
SEED, PLAN_JSON and TAG work as in local.py, for A/B tests.
"""
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import local  # noqa: E402  (reads --hires: 448x768)
from local import (CAMERA, FPS, H, H3, SHEET, W, Graph, ask, comfy, fetch, geo,  # noqa: E402
                   klein, rows, run)
from cloud import reveal  # noqa: E402

H3_FRAMES, LTX_FRAMES = 107, 105
LTX = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE, LTX_AUDIO_VAE = "ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors"
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
INPAINT = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
QUIET = " The only sounds are the place's own ambient sound; nobody speaks."
NEG = "blurry, distorted, extra limbs, extra fingers, warped hands, text, watermark, cartoon, speech"


MOVE = None if os.environ.get("MOVE", "dolly in") == "still" else os.environ.get("MOVE", "dolly in")
# the plan, with the place's sound only: the shutter belongs to the reverse shot's last beat, not the whole take
PLAN = local.PLAN.replace(", ending with the crisp click of a phone camera\n   shutter", "")
assert PLAN != local.PLAN
# the reverse shot cut in before the reveal, by what the photographer holds (REVERSE=none skips it):
#   smartphone: from the subject's side, the phone held sideways in both hands, its lenses aimed down ours, the eyes
#   over the top of it (the face stays in view; "phone front"/"phone front mid" hide it behind the phone)
#   dslr: Rear Window, the eye at the viewfinder, the long lens leaving the frame past us
REVERSE = os.environ.get("REVERSE") or {"smartphone": "phone landscape", "dslr": "standing dslr"}[CAMERA]
# skeletons (SDPose, whole body) of the films' photographer poses, each extracted from the source cropped to the
# photographer's upper body in the frame's own aspect, so the skeleton fills the frame where they will be
POSES = Path(__file__).parent / "poses"
# the pose ControlNet is built for the Qwen-Image 2.1 base model at ~40 steps; on Turbo it barely holds the skeleton
QWEN_BASE, QWEN_STEPS = "qwen_image_2.1_int8_convrot.safetensors", 40
HEAD_SWAP = "bfs_head_v1.1_qwen_2.1.safetensors"  # Alissonerdx/BFS-Best-Face-Swap, MIT
TILTS = [round(-0.1 + 0.2 * i, 1) for i in range(12)]  # look-at heights 4 m out, level to ~20 degrees up


def mesh(g, shot):
    geom = g.add("DA3Inference", image=shot, resolution=504, resize_method="upper_bound_resize", mode="mono",
                 da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                 weight_dtype="default")())
    return g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1, discontinuity_threshold=0.04,
                 confidence_threshold=0.1, use_sky_mask=True, texture=True)


def camera(g, eye=None, target=None):
    t = target or geo.TARGET
    return g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", eye or geo.START)},
                 target_x=t[0], target_y=t[1], target_z=t[2], roll=0.0, fov=geo.LENS, zoom=1.0,
                 camera_type="perspective")


def framing(post, prefix, eyes=None):
    """Sets geo.TARGET: the least upward tilt at which the post covers the bottom twentieth of the bystander's frame
    (the render's coverage mask) from every one of `eyes` (default: START). Below the post's own bottom edge there is
    nothing real to show."""
    g, level, eyes = Graph(), geo.TARGET, eyes or [geo.START]
    m = mesh(g, g.add("LoadImage", image=post)(0))
    for i, ty in enumerate(TILTS):
        geo.TARGET = (level[0], ty, level[2])
        for j, eye in enumerate(eyes):
            view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                         camera_info=camera(g, eye)())
            g.add("SaveImage", images=g.add("MaskToImage", mask=view(1))(), filename_prefix=f"{prefix}_cover{i}e{j}")
    files = run(g)
    for i, ty in enumerate(TILTS):
        covers = []
        for j in range(len(eyes)):
            f = files[f"cover{i}e{j}"][0]
            q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
            covers.append((np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{q}"))).convert("L"))[-H // 20:]
                           > 127).mean())
        cover = min(covers)
        if cover >= 0.85 or ty == TILTS[-1]:
            geo.TARGET = (level[0], ty, level[2])
            print(f"framing: look-at height {ty} m, bottom covered {cover:.2f}", flush=True)
            return


def h3(g, last, prompt, seed, first=None):
    model = g.add("LoraLoaderModelOnly", lora_name=H3["lora"], strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    cond = g.add("MiniMaxH3ImageToVideo", prompt=prompt, width=W, height=H, length=H3_FRAMES, last_frame=last, vae=vae(),
                 **({"first_frame": first} if first is not None else {}),
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    return g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                 audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())


def stage_ab(profile, post, plan, seed, prefix):
    g = Graph()
    who, shot = g.add("LoadImage", image=profile), g.add("LoadImage", image=post)
    sheet = klein(g, SHEET + plan["photographer"], [who(0)], seed, width=768, height=448)
    g.add("SaveImage", images=sheet(), filename_prefix=prefix + "_sheet")
    m = mesh(g, shot(0))
    view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                 camera_info=camera(g)())
    fill = ("ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around things: "
            "fill them seamlessly with more of the same place, same light, same lens. Keep everything else in "
            "ref_image_0 exactly as it is. No people in the foreground. " + plan["scene"])
    instant = klein(g, fill, [view(0)], seed)
    g.add("SaveImage", images=instant(), filename_prefix=prefix + "_keys")
    text = ("integrated_multimodal_description: Handheld bystander camera, nearly still. The subject is in motion and "
            f"ends exactly in the moment of <Picture 1>: {plan['action']} {plan['scene']}"
            f"\noverall_soundscape: {plan['soundscape']}\nnon_diegetic_music: N/A")
    g.add("SaveVideo", video=h3(g, instant(), text, seed)(), filename_prefix=prefix + "_scene", format="mp4",
          codec="h264")
    return g


BODY = [(x, y, z) for x in (-0.26, 0.26) for y in (0.18, -1.7) for z in (-0.5, 0.2)]


def mask_video(path, pad=None):
    """The photographer's region in every frame, projected from 3D (the bystander holds still), as a mask video."""
    tmp = tempfile.TemporaryDirectory()
    d = Path(tmp.name)
    pad = pad if pad is not None else round(W / 25)
    y0, x0, y1, x1 = geo.box(BODY, geo.START)
    for i in range(LTX_FRAMES):
        im = Image.new("RGB", (W, H), "black")
        ImageDraw.Draw(im).rectangle((x0 * W / 1000 - pad, y0 * H / 1000 - pad, x1 * W / 1000 + pad,
                                      y1 * H / 1000 + pad), fill="white")
        im.save(d / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "10", str(path)], check=True)
    tmp.cleanup()


LEVEL = geo.TARGET  # the opening of a move looks level; framing() tilts only the close end
PLACE = ("Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the foreground, "
         "seen from behind and slightly to the side, turned toward the subject, {hold}, aimed at the subject, taking "
         "a photo. Same hair and clothes as the sheet. Exactly one photographer. Change nothing else in ref_image_0. ")


def opening_layout(opening):
    end, geo.TARGET = geo.TARGET, LEVEL
    try:
        return geo.layout(opening, CAMERA), geo.box(BODY, opening)
    finally:
        geo.TARGET = end


def move_keys(profile, post, box, plan, seed, prefix, opening):
    """A camera move's two ends, with the photographer placed once. The 3D place seen from `opening` (further back,
    looking level) and from START (framed); gaps filled; Klein puts the photographer, whole, into the wide opening at
    twice the size; BiRefNet mattes them out (a real alpha), only inside their 3D box. compose() carries them to the close end."""
    g = Graph()
    who, shot = g.add("LoadImage", image=profile), g.add("LoadImage", image=post)
    sheet = klein(g, SHEET + plan["photographer"], [who(0)], seed, width=768, height=448)
    g.add("SaveImage", images=sheet(), filename_prefix=prefix + "_sheet")
    m = mesh(g, shot(0))
    fill = ("ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around things: "
            "fill them seamlessly with more of the same place, same light, same lens. Keep everything else in "
            "ref_image_0 exactly as it is. No people in the foreground. " + plan["scene"])
    ends = {}
    for name, eye, target, s_ in (("open", opening, LEVEL, seed + 2), ("end", geo.START, geo.TARGET, seed)):
        view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                     camera_info=camera(g, eye, target)())
        ends[name] = klein(g, fill, [view(0)], s_)
        g.add("SaveImage", images=ends[name](), filename_prefix=f"{prefix}_{name}fill")
    hi = klein(g, PLACE.format(hold=local.CAM["hold"]) + plan["photographer"] + rows(opening_layout(opening)[0]),
               [ends["open"](), sheet()], seed + 1, width=2 * W, height=2 * H)
    # an alpha matte (BiRefNet), not a segmentation: soft edges, and everything they carry (a pale backpack too)
    matte = g.add("RemoveBackground", image=hi(), bg_removal_model=g.add("LoadBackgroundRemovalModel",
                                                                       bg_removal_name="birefnet.safetensors")())
    mine = g.add("MaskComposite", destination=matte(0), operation="multiply", x=0, y=0,
                 source=g.add("ImageToMask", channel="red", image=g.add("LoadImage", image=box)(0))())
    g.add("SaveImage", images=hi(), filename_prefix=prefix + "_openhi")
    g.add("SaveImage", images=g.add("MaskToImage", mask=mine())(), filename_prefix=prefix + "_openmask")
    return g


def compose(open_fill, end_fill, hi, cut, opening, first_path, last_path):
    """First keyframe: the opening with the photographer cut from `hi`. Last keyframe: the framed end with the same
    cut-out photographer, moved and scaled to where the 3D projection puts them from START."""
    f0, f1 = Image.open(open_fill).convert("RGB"), Image.open(end_fill).convert("RGB")
    hi, cut = Image.open(hi).convert("RGB"), Image.open(cut).convert("L")  # the alpha matte, as is
    f0.paste(hi.resize((W, H), Image.LANCZOS), (0, 0), cut.resize((W, H), Image.LANCZOS))
    head, chest = (0.0, 0.18, -0.15), (0.0, -0.6, -0.15)
    (ax, ay), (bx, by) = (geo.project(p, opening, target=LEVEL) for p in (head, chest))
    (ex, ey), (fx, fy) = (geo.project(p, geo.START) for p in (head, chest))
    k = math.hypot(fx - ex, fy - ey) / math.hypot(bx - ax, by - ay) / 2  # hi is twice the frame
    size = (round(hi.width * k), round(hi.height * k))
    f1.paste(hi.resize(size, Image.LANCZOS), (round(ex - 2 * ax * k), round(ey - 2 * ay * k)),
             cut.resize(size, Image.LANCZOS))
    f0.save(first_path)
    f1.save(last_path)


def move_film(first, last, plan, seed, prefix):
    """H3 films the move between the two keyframes, photographer and all: the same person at both ends."""
    g = Graph()
    text = (f"integrated_multimodal_description: The camera makes {geo.MOVE_WORDS[MOVE]} and tilts up, moving "
            f"through the real space of the place. The photographer in the foreground, seen from behind, "
            f"{local.CAM['act']}. There is exactly one photographer. The subject is in motion and ends exactly in the "
            f"moment of <Picture 2>: {plan['action']} {plan['scene']}"
            f"\noverall_soundscape: {plan['soundscape']}\nnon_diegetic_music: N/A")
    g.add("SaveVideo", video=h3(g, g.add("LoadImage", image=last)(0), text, seed,
                                g.add("LoadImage", image=first)(0))(),
          filename_prefix=prefix + "_raw", format="mp4", codec="h264")
    return g


# Reverse shots of the photographer taking the picture, after the films the user pointed to. Their eyes are the origin,
# looking down -Z at the subject; each shot is a bystander position (angle off their line of sight, distance, height),
# a lens, and how the photographer stands, so the face and camera boxes come from the same 3D geometry.
SHOTS = {
    # Rear Window: from the front side, long lens; the lens leaves the frame's edge past us, eyes over the camera
    "rear window": dict(angle=35.0, dist=1.1, height=-0.03, fov=24.0, crouch=0.0,
                        words="A tight cinematic close-up on a long lens, shallow depth of field"),
    # Peeping Tom: almost down the lens; the lens fills the foreground, the face just behind the camera
    "peeping tom": dict(angle=10.0, dist=0.9, height=0.0, fov=28.0, crouch=0.0,
                        words="A cinematic close-up almost straight down the camera's lens, the lens huge in the "
                              "foreground, shallow depth of field"),
    # Blow-Up: low and wider, the photographer crouched and leaning in, shooting down at the subject
    "blow up": dict(angle=50.0, dist=1.6, height=-1.0, fov=50.0, crouch=0.55,
                    words="A low-angle 1960s fashion-shoot shot on a wider lens: the photographer crouched, leaning in, "
                          "shooting down at the subject with energy"),
}


def shot_eye(shot):
    a = math.radians(shot["angle"])
    return (shot["dist"] * math.sin(a), shot["height"] - shot["crouch"], -shot["dist"] * math.cos(a))


def close_layout(shot, camera=CAMERA):
    """Face and camera boxes for a reverse shot, projected from 3D (the photographer's own geometry)."""
    drop = shot["crouch"]
    eye, target = shot_eye(shot), (0.0, -0.03 - drop, -0.2)

    def bbox(points):
        px = [geo.project((x, y - drop, z), eye, target=target, fov=shot["fov"]) for x, y, z in points]
        clip = lambda v, m: max(0, min(1000, round(v / m * 1000)))  # noqa: E731
        return [clip(min(p[1] for p in px), H), clip(min(p[0] for p in px), W),
                clip(max(p[1] for p in px), H), clip(max(p[0] for p in px), W)]

    face = [(x, y, z) for x in (-0.08, 0.08) for y in (0.13, -0.13) for z in (-0.1, 0.08)]
    if camera == "dslr":
        held = [(x, y, z) for x in (-0.09, 0.09) for y in (-0.07, 0.07) for z in (-0.06, -0.38)]  # body and lens
        what = "a DSLR with a long lens at their eye, the lens pointing out past the viewer"
    else:
        held = [(x, y, -0.42) for x in (-0.08, 0.12) for y in (-0.06, 0.16)]
        what = "a phone held up in both hands, its back to the viewer, aimed past the viewer"
    return [{"type": "obj", "bbox": bbox(face), "desc": "face: image 2's person, eyes on the subject beyond the "
                                                          "camera"},
            {"type": "obj", "bbox": bbox(held), "desc": f"camera: {what}"}]


def close_still(profile, post, sheet, plan, seed, prefix, name="rear window"):
    """A reverse shot's still: the photographer from the profile picture and sheet, face and camera where the 3D
    layout puts them, in front of a soft blur of the post's own colours and light (no place to invent)."""
    shot = SHOTS[name]
    g = Graph()
    plate = g.add("ImageBlur", image=g.add("ImageScale", image=g.add("LoadImage", image=post)(0),
                                           upscale_method="lanczos", width=W, height=H, crop="center")(),
                  blur_radius=31, sigma=10.0)
    still = klein(g, shot["words"] + ": the photographer, the person from ref_image_2 (face) and ref_image_1 (clothes "
                     "and hair), in the act of taking a photo. Background: ref_image_0, out of focus. "
                     + plan["photographer"] + rows(close_layout(shot)),
                  [plate(), g.add("LoadImage", image=sheet)(0), g.add("LoadImage", image=profile)(0)], seed + 5)
    g.add("SaveImage", images=still(), filename_prefix=f"{prefix}_{name.replace(' ', '')}")
    return g


def reverse_still(profile, post, sheet, plan, seed, prefix, pose):
    """The reverse shot's still, in two stages, each tool doing one job:
    1. the pose: Qwen-Image 2.1 (base) held to a film's photographer skeleton by the Fun ControlNet-Union, from a
       description alone (identity references pull the pose back toward the profile picture's);
    2. who: the BFS head-swap LoRA (Qwen-Image 2.1) puts the profile picture's head on that person, copying head
       rotation, eye direction and expression from the posed frame (its documented prompt and image order)."""
    g = Graph()
    vae = g.add("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")
    model = g.add("UNETLoader", unet_name=QWEN_BASE, weight_dtype="default")
    model = g.add("QwenImageDiffsynthControlnet", model=model(), vae=vae(), strength=1.0,
                  image=g.add("LoadImage", image=pose)(0),
                  model_patch=g.add("ModelPatchLoader",
                                    name="qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors")())
    clip = g.add("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")
    enc = g.add("TextEncodeQwenImage21", clip=clip(), vae=vae(), resolution=1024,
                negative_prompt="cartoon, 3d render, cgi, plastic skin, deformed hands, extra fingers, text",
                prompt=f"A real cinematic film still, a tight close-up of a photographer, {plan['photographer']}, "
                       f"{local.CAM['hold']}, taking a photo. Long lens, shallow depth of field, natural light, "
                       "real skin and real adult hands.")
    posed = g.add("VAEDecode", vae=vae(), samples=g.add(
        "KSampler", model=model(), seed=seed + 7, steps=QWEN_STEPS, cfg=1.0, sampler_name="euler", scheduler="simple",
        positive=enc(0), negative=enc(1), denoise=1.0,
        latent_image=g.add("EmptyLatentImage", width=768, height=1312, batch_size=1)())())  # the frame's aspect
    g.add("SaveImage", images=posed(), filename_prefix=prefix + "_revposed")
    swap = g.add("LoraLoaderModelOnly", lora_name=HEAD_SWAP, strength_model=1.0,
                 model=g.add("UNETLoader", unet_name=QWEN_BASE, weight_dtype="default")())
    enc = g.add("TextEncodeQwenImage21", clip=clip(), vae=vae(), resolution=1024, negative_prompt="",
                prompt="head_swap: start with <image1> as the base image, keeping its lighting, environment, and "
                       "background. remove the head from <image1> completely and replace it with the head from "
                       "<image2>, strictly preserving the hair, eye color, nose structure from <image2>. copy the "
                       "direction of the eye, head rotation, micro expressions from <image1>, high quality, sharp "
                       "details, 4k",
                **{"images.image_1": posed(), "images.image_2": g.add("LoadImage", image=profile)(0)})
    who = g.add("VAEDecode", vae=vae(), samples=g.add(
        "KSampler", model=swap(), seed=seed + 8, steps=QWEN_STEPS, cfg=1.0, sampler_name="euler", scheduler="simple",
        positive=enc(0), negative=enc(1), denoise=1.0,
        latent_image=g.add("EmptyLatentImage", width=768, height=1312, batch_size=1)())())
    # 3. where: a long lens has nothing sharp behind them, so the background is the real place, defocused: the
    #    photographer matted (BiRefNet, a real alpha) over the post itself, scaled to the frame and blurred hard
    #    (no invented scenery behind them; Qwen's own background put a parked car at head height)
    still = matted(g, who(), defocused(g, post, 768, 1312))
    g.add("SaveImage", images=who(), filename_prefix=prefix + "_revswapped")
    g.add("SaveImage", images=still(), filename_prefix=prefix + "_revstill")
    return g


def defocused(g, post, width, height):
    """The real place behind a long-lens close-up: the post scaled to the frame, blurred hard (ImageBlur tops out
    at radius 31, so it is repeated)."""
    plate = g.add("ImageScale", image=g.add("LoadImage", image=post)(0), upscale_method="lanczos", width=width,
                  height=height, crop="center")
    for _ in range(3):
        plate = g.add("ImageBlur", image=plate(), blur_radius=31, sigma=10.0)
    return plate


def matted(g, images, plate, frames=1):
    """`images` (a batch) matted with BiRefNet (a real alpha) over `plate`, every frame."""
    matte = g.add("RemoveBackground", image=images, bg_removal_model=g.add(
        "LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")())
    behind = g.add("RepeatImageBatch", image=plate(), amount=frames) if frames > 1 else plate
    return g.add("ImageCompositeMasked", destination=behind(), source=images, x=0, y=0, resize_source=False,
                 mask=matte(0))


def reverse_film(still, pose, plan, seed, prefix, post=None):
    """H3 animates the reverse shot, the camera locked off. It continues the cut: the push-in ends on the placed
    keyframe, camera already up, so the pose is held there (the Fun ControlNet-Union on the same skeleton, every
    frame), and the shutter fires at the end."""
    g = Graph()
    text = ("integrated_multimodal_description: A tight cinematic close-up, the camera locked off on a long lens. The "
            f"photographer, {local.CAM['hold']}, holds it steady, intent behind the camera, and presses the shutter "
            "at the very end. Shallow depth of field, the background soft."
            f"\noverall_soundscape: {plan['soundscape']}"
            "\nnon_diegetic_music: N/A")
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    model = g.add("LoraLoaderModelOnly", lora_name=H3["lora"], strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    model = g.add("MiniMaxH3FunControlNetApply", model=model(), vae=vae(), strength=1.0, start_percent=0.0,
                  end_percent=1.0, model_patch=g.add(
                      "ModelPatchLoader", name="minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors")(),
                  control_video=g.add("RepeatImageBatch", image=g.add("LoadImage", image=pose)(0), amount=73)())
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=73, vae=vae(),
                 first_frame=g.add("LoadImage", image=still)(0),
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    frames = g.add("VAEDecode", samples=out(0), vae=vae())()
    if post:  # H3 only pins the first frame's background: hold the defocused place behind every frame
        frames = matted(g, frames, defocused(g, post, W, H), 73)()
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=frames,
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_reverse", format="mp4", codec="h264")
    return g


SHUTTER = Path(__file__).parent / "sfx" / "dslr_shutter.wav"  # a real DSLR (see sfx/CREDITS.md)


def cut(take, reverse, out, keep=2.0):
    """The take, then the reverse shot's first `keep` seconds (it starts where the take ends: the same pose), and
    the shutter on its last beat: a recorded shutter laid in like foley, not whatever H3 may or may not voice."""
    click_at = keep - 0.25  # the shutter, then a beat of it before the reveal's flash
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(take), "-i", str(reverse), "-i", str(SHUTTER),
                    "-filter_complex",
                    f"[1:v]trim=0:{keep:.3f},setpts=PTS-STARTPTS[r];"
                    f"[1:a]atrim=0:{keep:.3f},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[ra];"
                    f"[2:a]adelay={int(click_at * 1000)}:all=1,apad,atrim=0:{keep:.3f}[c];"
                    "[ra][c]amix=inputs=2:normalize=0[rc];"
                    "[0:a]aresample=48000,aformat=channel_layouts=stereo[ta];"
                    "[0:v][ta][r][rc]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-c:a", "aac", str(out)], check=True)


def stage_cd(sheet, scene, mask, plan, seed, prefix):
    g = Graph()
    frames = g.add("GetVideoComponents", video=g.add("LoadVideo", file=scene)())
    src = g.add("ImageFromBatch", image=frames(0), batch_index=0, length=LTX_FRAMES)
    hole_img = g.add("ImageFromBatch", batch_index=0, length=LTX_FRAMES,
                     image=g.add("GetVideoComponents", video=g.add("LoadVideo", file=mask)())(0))
    hole = g.add("ImageToMask", channel="red", image=hole_img())
    sheet_img = g.add("LoadImage", image=sheet)

    def place(index, s_):
        img = klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the "
                       "foreground, seen from behind and slightly to the side, turned toward the subject, "
                       f"{local.CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the "
                       "sheet. Exactly one photographer. Change nothing else in ref_image_0. " + plan["photographer"]
                    + rows(geo.layout(camera=CAMERA)),
                    [g.add("ImageFromBatch", image=frames(0), batch_index=index, length=1)(), sheet_img(0)], s_)
        return img

    last = place(LTX_FRAMES - 1, seed + 1)
    if os.environ.get("TWO_PLACES"):  # the old way, for A/B tests: two placements, two different phones to morph between
        first = place(0, seed)
    else:  # one placement: the same photographer, phone and grip cut out of the last frame onto the first
        matte = g.add("RemoveBackground", image=last(), bg_removal_model=g.add(
            "LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")())  # an alpha matte
        cut = g.add("MaskComposite", destination=matte(0), operation="multiply", x=0, y=0,  # only inside their 3D box
                    source=g.add("ImageToMask", channel="red", image=g.add("ImageFromBatch", batch_index=0, length=1,
                                                                           image=hole_img())())())
        first = g.add("ImageCompositeMasked", destination=g.add("ImageFromBatch", image=frames(0), batch_index=0,
                                                                 length=1)(), source=last(), x=0, y=0,
                      resize_source=False, mask=cut())
    g.add("SaveImage", images=g.add("ImageBatch", image1=first(), image2=last())(), filename_prefix=prefix + "_placed")
    vae, avae = g.add("VAELoader", vae_name=LTX_VAE), g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    ic = g.add("LTXICLoRALoaderModelOnly", model=g.add("UNETLoader", unet_name=LTX, weight_dtype="default")(),
               lora_name=INPAINT, strength_model=1.0)
    text = (f"{plan['photographer']}, seen from behind in the foreground, {local.CAM['act']}. {plan['action']} "
            f"{plan['scene']}{QUIET}")
    cond = g.add("LTXVConditioning", frame_rate=float(FPS), positive=g.add("CLIPTextEncode", text=text, clip=clip())(),
                 negative=g.add("CLIPTextEncode", text=NEG, clip=clip())())
    lat = g.add("LTXVImgToVideoInplace", vae=vae(), image=first(), strength=1.0, bypass=False,
                latent=g.add("EmptyLTXVLatentVideo", width=W, height=H, length=LTX_FRAMES, batch_size=1)())
    guide = g.add("LTXAddVideoICLoRAGuideAdvanced", positive=cond(0), negative=cond(1), vae=vae(), latent=lat(),
                  image=g.add("LTXVInpaintPreprocess", images=src(), mask=hole())(), frame_idx=0, strength=1.0,
                  latent_downscale_factor=ic(1), crop="disabled", use_tiled_encode=False, tile_size=256,
                  tile_overlap=64, attention_strength=1.0, use_streaming=False, tile_frames=97)
    guide = g.add("LTXVAddGuide", positive=guide(0), negative=guide(1), vae=vae(), latent=guide(2), image=last(),
                  frame_idx=-1, strength=1.0)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=LTX_FRAMES, frame_rate=FPS, batch_size=1, audio_vae=avae())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=ic(0), positive=guide(0), negative=guide(1), cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(),
                latent_image=g.add("LTXVConcatAVLatent", video_latent=guide(2), audio_latent=audio())())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    video = g.add("LTXVCropGuides", positive=guide(0), negative=guide(1), latent=sep(0))(2)
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=video, vae=vae())(),
                                   audio=frames(1))(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def main(profile, post, out="out"):
    name, t = Path(post).stem, time.time()
    seed = int(os.environ.get("SEED") or random.randrange(2**31))
    tag = f"{name}_{os.environ.get('TAG', 'mol')}"
    saved = os.environ.get("PLAN_JSON")
    if saved and Path(saved).exists():
        plan = json.loads(Path(saved).read_text())
    else:
        plan = ask(PLAN, "Left: the profile picture. Right: the photo they posted. Plan the moment.", [profile, post])
        if saved:
            Path(saved).write_text(json.dumps(plan))
    od = Path(out) / "steps"
    od.mkdir(parents=True, exist_ok=True)
    pu, su = comfy.upload(profile, f"momento_{Path(profile).name}"), comfy.upload(post, f"momento_{Path(post).name}")
    opening = geo.MOVES[MOVE] if MOVE else None
    if not os.environ.get("LEVEL"):  # LEVEL=1 keeps the old level camera, for A/B tests
        framing(su, f"momento/{tag}")
    if opening:  # the camera moves: H3 films it between two keyframes holding the one placement
        box = Image.new("RGB", (2 * W, 2 * H), "black")
        y0, x0, y1, x1 = opening_layout(opening)[1]
        ImageDraw.Draw(box).rectangle((x0 * W / 500, y0 * H / 500, x1 * W / 500, y1 * H / 500), fill="white")
        box.save(od / f"{tag}_openbox.png")
        files = run(move_keys(pu, su, comfy.upload(str(od / f"{tag}_openbox.png"), f"momento_{tag}_openbox.png"),
                              plan, seed, f"momento/{tag}", opening))
        for k in ("sheet", "openfill", "endfill", "openhi", "openmask"):
            fetch(files[k][0], od / f"{tag}_{k}.png")
        compose(*(od / f"{tag}_{k}.png" for k in ("openfill", "endfill", "openhi", "openmask")), opening,
                od / f"{tag}_first.png", od / f"{tag}_last.png")
        files = run(move_film(*(comfy.upload(str(od / f"{tag}_{k}.png"), f"momento_{tag}_{k}.png")
                                for k in ("first", "last")), plan, seed, f"momento/{tag}"))
        raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
        fetch(files["raw"][0], raw)
        if REVERSE != "none":  # cut to the photographer taking the picture; the shutter ends it
            pose = comfy.upload(str(POSES / f"{REVERSE.replace(' ', '_')}.png"), f"momento_pose_{REVERSE}.png",)
            files = run(reverse_still(pu, su, comfy.upload(str(od / f"{tag}_sheet.png"), f"momento_{tag}_sheet.png"),
                                      plan, seed, f"momento/{tag}", pose))
            fetch(files["revstill"][0], od / f"{tag}_revstill.png")
            # the still and its skeleton are both in the frame's aspect already: scaled, never cropped
            for src, dst in ((od / f"{tag}_revstill.png", od / f"{tag}_revkey.png"),
                             (POSES / f"{REVERSE.replace(' ', '_')}.png", od / f"{tag}_revpose.png")):
                Image.open(src).convert("RGB").resize((W, H), Image.LANCZOS).save(dst)
            files = run(reverse_film(comfy.upload(str(od / f"{tag}_revkey.png"), f"momento_{tag}_revkey.png"),
                                     comfy.upload(str(od / f"{tag}_revpose.png"), f"momento_{tag}_revpose.png"),
                                     plan, seed, f"momento/{tag}", post=su))
            fetch(files["reverse"][0], od / f"{tag}_reverse.mp4")
            cut(raw, od / f"{tag}_reverse.mp4", od / f"{tag}_cut.mp4")
            raw = od / f"{tag}_cut.mp4"
        reveal(str(raw), post, str(final), click=REVERSE == "none")
        print(f"[{tag}] ({time.time() - t:.0f}s) {final}", flush=True)
        return
    files = run(stage_ab(pu, su, plan, seed, f"momento/{tag}"))
    fetch(files["sheet"][0], od / f"{tag}_sheet.png")
    fetch(files["scene"][0], od / f"{tag}_scene.mp4")
    mask_video(od / f"{tag}_mask.mp4")
    files = run(stage_cd(comfy.upload(str(od / f"{tag}_sheet.png"), f"momento_{tag}_sheet.png"),
                         comfy.upload(str(od / f"{tag}_scene.mp4"), f"momento_{tag}_scene.mp4"),
                         comfy.upload(str(od / f"{tag}_mask.mp4"), f"momento_{tag}_mask.mp4"), plan, seed,
                         f"momento/{tag}"))
    raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
    fetch(files["final"][0], raw)
    reveal(str(raw), post, str(final))
    print(f"[{tag}] ({time.time() - t:.0f}s) {final}", flush=True)


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]
    for p in posts:
        main(profile, p)
