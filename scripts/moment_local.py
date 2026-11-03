"""The Comfy Cloud pipeline that made the approved v2 clips (scripts/moment.py), on your own GPU.

    COMFY_URL=... PLANNER_URL=... PLANNER_MODEL=... python scripts/moment_local.py profile.jpg post.jpg [more posts...]

Same steps as moment.py; only the hosted nodes are replaced: FLUX 3 Image -> FLUX 3 Klein, Gemini -> the local planner.
  0. Framing: the bystander tilts up until the post covers the bottom of the frame (measured on the 3D render),
     so nothing below the photo has to be invented (that strip is where a sidewalk grew halfway up the bus).
  A. Plan; character sheet; the instant (the post's 3D place seen from the bystander, gaps filled seamlessly).
  B. MiniMax H3 films the scene with no photographer, camera nearly still, ending on the instant.
  C. Klein places the photographer into the scene's first and last frames, in the box projected from 3D.
  D. LTX-2.5 + the in/outpainting IC-LoRA regenerates only the photographer's region (a mask projected from 3D)
     through the whole clip, pinned to the placed first and last frames; the rest stays H3's.
SEED, PLAN_JSON and TAG work as in local.py, for A/B tests.
"""
import io
import json
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
from local import (CAMERA, FPS, H, H3, PLAN, SHEET, W, Graph, ask, comfy, fetch, geo,  # noqa: E402
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


TILTS = [round(-0.1 + 0.2 * i, 1) for i in range(12)]  # look-at heights 4 m out, level to ~20 degrees up


def mesh(g, shot):
    geom = g.add("DA3Inference", image=shot, resolution=504, resize_method="upper_bound_resize", mode="mono",
                 da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                 weight_dtype="default")())
    return g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1, discontinuity_threshold=0.04,
                 confidence_threshold=0.1, use_sky_mask=True, texture=True)


def camera(g):
    return g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", geo.START)},
                 target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0, fov=geo.LENS,
                 zoom=1.0, camera_type="perspective")


def framing(post, prefix):
    """Sets geo.TARGET: the least upward tilt at which the post covers the bottom twentieth of the bystander's frame
    (the render's coverage mask). Below the post's own bottom edge there is nothing real to show."""
    g, level = Graph(), geo.TARGET
    m = mesh(g, g.add("LoadImage", image=post)(0))
    for i, ty in enumerate(TILTS):
        geo.TARGET = (level[0], ty, level[2])
        view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                     camera_info=camera(g)())
        g.add("SaveImage", images=g.add("MaskToImage", mask=view(1))(), filename_prefix=f"{prefix}_cover{i}")
    files = run(g)
    for i, ty in enumerate(TILTS):
        f = files[f"cover{i}"][0]
        q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
        cover = (np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{q}"))).convert("L"))[-H // 20:] > 127).mean()
        if cover >= 0.85 or ty == TILTS[-1]:
            geo.TARGET = (level[0], ty, level[2])
            print(f"framing: look-at height {ty} m, bottom covered {cover:.2f}", flush=True)
            return


def h3(g, last, prompt, seed):
    model = g.add("LoraLoaderModelOnly", lora_name=H3["lora"], strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    cond = g.add("MiniMaxH3ImageToVideo", prompt=prompt, width=W, height=H, length=H3_FRAMES, last_frame=last, vae=vae(),
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
    view = g.add("RenderMesh", mesh=mesh(g, shot(0))(), mode="texture", width=W, height=H, background="#000000",
                 camera_info=camera(g)())
    instant = klein(g, "ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around "
                       "things: fill them seamlessly with more of the same place, same light, same lens. Keep "
                       "everything else in ref_image_0 exactly as it is. No people in the foreground. " + plan["scene"],
                    [view(0)], seed)
    g.add("SaveImage", images=instant(), filename_prefix=prefix + "_keys")
    text = ("integrated_multimodal_description: Handheld bystander camera, nearly still. The subject is in motion and "
            f"ends exactly in the moment of <Picture 1>: {plan['action']} {plan['scene']}"
            f"\noverall_soundscape: {plan['soundscape']}\nnon_diegetic_music: N/A")
    g.add("SaveVideo", video=h3(g, instant(), text, seed)(), filename_prefix=prefix + "_scene", format="mp4",
          codec="h264")
    return g


def mask_video(path, pad=None):
    """The photographer's region in every frame, projected from 3D (the bystander holds still), as a mask video."""
    d = Path(tempfile.mkdtemp())
    pad = pad if pad is not None else round(W / 25)
    body = [(x, y, z) for x in (-0.26, 0.26) for y in (0.18, -1.7) for z in (-0.5, 0.2)]
    y0, x0, y1, x1 = geo.box(body, geo.START)
    for i in range(LTX_FRAMES):
        im = Image.new("RGB", (W, H), "black")
        ImageDraw.Draw(im).rectangle((x0 * W / 1000 - pad, y0 * H / 1000 - pad, x1 * W / 1000 + pad,
                                      y1 * H / 1000 + pad), fill="white")
        im.save(d / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "10", str(path)], check=True)


def stage_cd(sheet, scene, mask, plan, seed, prefix):
    g = Graph()
    frames = g.add("GetVideoComponents", video=g.add("LoadVideo", file=scene)())
    src = g.add("ImageFromBatch", image=frames(0), batch_index=0, length=LTX_FRAMES)
    hole = g.add("ImageToMask", channel="red", image=g.add("ImageFromBatch", batch_index=0, length=LTX_FRAMES,
                                                         image=g.add("GetVideoComponents", video=g.add(
                                                             "LoadVideo", file=mask)())(0))())
    sheet_img = g.add("LoadImage", image=sheet)

    def place(index, s_):
        img = klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the "
                       "foreground, seen from behind and slightly to the side, turned toward the subject, "
                       f"{local.CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the "
                       "sheet. Exactly one photographer. Change nothing else in ref_image_0. " + plan["photographer"]
                    + rows(geo.layout(camera=CAMERA)),
                    [g.add("ImageFromBatch", image=frames(0), batch_index=index, length=1)(), sheet_img(0)], s_)
        return img

    first, last = place(0, seed), place(LTX_FRAMES - 1, seed + 1)
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
    if not os.environ.get("LEVEL"):  # LEVEL=1 keeps the old level camera, for A/B tests
        framing(su, f"momento/{tag}")
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
