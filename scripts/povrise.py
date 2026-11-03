"""First person, with the hands actually rising into frame (not fading in).

    COMFY_URL=... python scripts/povrise.py post.jpg out.mp4 [seed]

Stage 1 (ComfyUI): the end frame (the real photo, hands and phone in the margin below), the same frame without the
hands, SAM 3.1 masks of the hands/phone and of the photo's movers.
Stage 2 (here): a guide video in which the hands-and-phone cutout physically slides up from below the frame edge to
where it ends, over the frame without hands. Rigid geometry: the path is fixed before any model runs.
Stage 3 (ComfyUI): MiniMax H3 + Fun ControlNet-Union inpaints only the moving regions (the rising hands, the movers)
with the guide as control and source, so it renders motion that's already there instead of dissolving.
"""
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
import pov  # noqa: E402
from local import Graph, fetch, klein, run  # noqa: E402

W, H, F, FPS, TOP, BOTTOM = pov.W, pov.H, pov.F, pov.FPS, pov.TOP, pov.BOTTOM
RISE = (0.15, 0.7)  # the hands come up over this part of the clip, then hold


def stage1(post, profile, scene, seed, prefix):
    g = Graph()
    who = g.add("LoadImage", image=profile)
    photo = g.add("ImageScale", image=g.add("LoadImage", image=post)(0), upscale_method="lanczos", width=W,
                  height=W, crop="center")
    pad = g.add("ImagePadForOutpaint", image=photo(), left=0, top=TOP, right=0, bottom=BOTTOM, feathering=0)
    keep = g.add("InvertMask", mask=pad(1))
    filled = klein(g, "ref_image_0 is a real photo with empty grey margins above and below it. Fill the top margin "
                      "with more of the same place and sky. In the bottom margin, seen from the photographer's own "
                      "eyes: their two hands holding a smartphone up toward the scene in portrait orientation, "
                      "thumbs on the edges, the lower half of the phone and the hands visible. Real photograph, same "
                      "light. The hands are the hands of the person in ref_image_1: exactly their skin tone. Keep the "
                      "photo itself exactly as it is. " + scene, [pad(0), who(0)], seed, width=W, height=H,
                   loras=[("klein_skin_texture.safetensors", 0.6)])
    bare = klein(g, "ref_image_0 is a real photo with empty grey margins above and below it. Fill them with more of "
                    "the same place, continuing the photo naturally: sky above, the ground in front below, nothing "
                    "else. Real photograph, same light. Keep the photo itself exactly as it is. " + scene,
                 [pad(0)], seed + 1, width=W, height=H)
    end = g.add("ImageCompositeMasked", destination=filled(), source=pad(0), x=0, y=0, resize_source=False,
                mask=keep())
    bare = g.add("ImageCompositeMasked", destination=bare(), source=pad(0), x=0, y=0, resize_source=False,
                 mask=keep())
    sam = g.add("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    held = g.add("SAM3_Detect", model=sam(0), image=end(), threshold=0.3, refine_iterations=4, individual_masks=False,
                 conditioning=g.add("CLIPTextEncode", text="hand:2, smartphone:1", clip=sam(1))())
    movers = g.add("SAM3_Detect", model=sam(0), image=end(), threshold=0.25, refine_iterations=2,
                   individual_masks=False, conditioning=g.add("CLIPTextEncode", text=pov.MOVERS, clip=sam(1))())
    inside = g.add("MaskComposite", destination=movers(0), source=keep(), x=0, y=0, operation="multiply")
    face = g.add("SAM3_Detect", model=sam(0), image=who(0), threshold=0.3, refine_iterations=2, individual_masks=False,
                 conditioning=g.add("CLIPTextEncode", text="face:1", clip=sam(1))())
    for name, img in (("end", end()), ("bare", bare()), ("hands", g.add("MaskToImage", mask=held(0))()),
                      ("movers", g.add("MaskToImage", mask=inside())()), ("profile", who(0)),
                      ("face", g.add("MaskToImage", mask=face(0))())):
        g.add("SaveImage", images=img, filename_prefix=f"{prefix}_{name}")
    return g


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def guide(end, bare, hands, movers, out_dir):
    """The guide video and the per-frame mask of what may change."""
    end, bare = np.asarray(end, np.float32), np.asarray(bare, np.float32)
    # a tight matte: SAM's hands/phone mask pulled in a pixel and feathered, so no ring of road rides along with
    # them (dilating it first is what dragged the background up with the hands)
    hm = np.asarray(hands.convert("L").filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2)),
                    np.float32) / 255.0
    mv = np.asarray(movers.convert("L").filter(ImageFilter.MaxFilter(9)), np.float32) / 255.0
    rows = np.where(hm.max(1) > 0.5)[0]
    travel = (H - rows.min() + 8) if len(rows) else H // 3  # far enough that the hands start below the frame
    frames, masks = [], []
    for i in range(F):
        dy = int(round((1 - ease((i / (F - 1) - RISE[0]) / (RISE[1] - RISE[0]))) * travel))
        shifted_img, shifted_m = np.zeros_like(end), np.zeros_like(hm)
        if dy < H:
            shifted_img[dy:] = end[:H - dy]
            shifted_m[dy:] = hm[:H - dy]
        a = shifted_m[..., None]
        frames.append((bare * (1 - a) + shifted_img * a).astype(np.uint8))
        # the hands' region now and a little around it (shading, motion blur), plus the photo's movers
        m = np.asarray(Image.fromarray((shifted_m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7)),
                       np.float32) / 255  # just the silhouette's edge: the road around it stays the photo's
        masks.append((np.maximum(m, mv) * 255).astype(np.uint8))
    paths = {}
    for name, seq in (("guide", frames), ("mask", [np.stack([x] * 3, -1) for x in masks])):
        p = Path(out_dir) / f"{name}.mp4"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv444p",
                                str(p)], stdin=subprocess.PIPE)
        enc.stdin.write(np.stack(seq).tobytes())
        enc.stdin.close()
        enc.wait()
        paths[name] = p
    return paths


def stage3(guide_name, mask_name, end_name, scene, action, seed, prefix):
    g = Graph()
    gv = g.add("GetVideoComponents", video=g.add("LoadVideo", file=guide_name)())
    mask = g.add("ImageToMask", channel="red",
                 image=g.add("GetVideoComponents", video=g.add("LoadVideo", file=mask_name)())(0))
    end = g.add("LoadImage", image=end_name)
    model = g.add("LoraLoaderModelOnly", lora_name=pov.TURBO, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=pov.H3U, weight_dtype="default")())
    model = g.add("LoraLoaderModelOnly", lora_name=pov.REAL, strength_model=1.0, model=model())
    vae, avae = g.add("VAELoader", vae_name=pov.VAE), g.add("VAELoader", vae_name=pov.AVAE)
    model = g.add("MiniMaxH3FunControlNetApply", model=model(), vae=vae(), strength=1.0, start_percent=0.0,
                  end_percent=1.0, model_patch=g.add("ModelPatchLoader", name=pov.CN)(), control_video=gv(0),
                  mask=mask(), source_video=gv(0))
    text = ("integrated_multimodal_description: r34l1sm Seen from the photographer's own eyes, handheld and nearly "
            f"still. {action} Their two hands bring a smartphone up from below the frame into view and steady it, "
            f"framing the shot of <Picture 1>. {scene}\noverall_soundscape: the place's ambient sound, then the click "
            "of a phone camera shutter\nnon_diegetic_music: N/A")
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=F, last_frame=end(0), vae=vae(),
                 clip=g.add("CLIPLoader", clip_name=pov.CLIP, type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=6, denoise=1.0)(),
                latent_image=cond(1))
    final = g.add("ImageCompositeMasked", destination=gv(0), source=g.add("VAEDecode", samples=out(0), vae=vae())(),
                  x=0, y=0, resize_source=False, mask=mask())
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=final(),
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


SKIN_DE = float(os.environ.get("SKIN_DE", 20.0))  # CIELAB: ~13 is the same skin under different light; a different
# skin tone measures well past 20


def skin_delta(tmp):
    """Median skin colour of the hands vs the face in the profile picture, as CIELAB delta E (2000-ish via ab+L)."""
    def lab_median(img, mask):
        im = Image.open(img).convert("RGB")
        m = np.asarray(Image.open(mask).convert("L").resize(im.size), np.float32) > 127
        lab = np.asarray(im.convert("LAB"), np.float32).reshape(-1, 3)[m.reshape(-1)]
        if len(lab) < 50:
            return None
        # skin, not sleeves or phone: keep the middle of the luminance range
        L = lab[:, 0]
        lab = lab[(L > np.percentile(L, 25)) & (L < np.percentile(L, 85))]
        return np.median(lab, 0)
    a, b = lab_median(tmp / "profile.png", tmp / "face.png"), lab_median(tmp / "end.png", tmp / "hands.png")
    if a is None or b is None:
        return 99.0
    return float(np.sqrt((((a - b) * np.array([0.5, 1.0, 1.0])) ** 2).sum()))  # chroma counts more than lighting


def main(post, out, seed=None):
    profile = os.environ["PROFILE"]
    seed = int(seed) if seed else random.randrange(2**31)
    scene, action = os.environ.get("SCENE", ""), os.environ.get("ACTION", "")
    tmp = Path(tempfile.mkdtemp())
    stem = Path(post).stem
    pu = comfy.upload(post, f"momento_{Path(post).name}")
    fu = comfy.upload(profile, f"momento_{Path(profile).name}")
    for attempt in range(4):  # the hands must be the person's skin tone: measured, re-rolled if not
        files = run(stage1(pu, fu, scene, seed, f"momento/rise_{stem}_{seed}"))
        for k in ("end", "bare", "hands", "movers", "profile", "face"):
            fetch(files[k][0], tmp / f"{k}.png")
        de = skin_delta(tmp)
        print(f"skin check seed {seed}: delta E {de:.1f}", flush=True)
        if de <= SKIN_DE:
            break
        seed += 1000
    else:
        raise SystemExit(f"no take's hands matched the person's skin tone (last delta E {de:.1f}); not shipping it")
    im = {k: Image.open(tmp / f"{k}.png").convert("RGB") for k in ("end", "bare", "hands", "movers")}
    paths = guide(im["end"], im["bare"], im["hands"], im["movers"], tmp)
    gn = comfy.upload(str(paths["guide"]), f"momento_rise_{stem}_{seed}_guide.mp4")
    mn = comfy.upload(str(paths["mask"]), f"momento_rise_{stem}_{seed}_mask.mp4")
    en = comfy.upload(str(tmp / "end.png"), f"momento_rise_{stem}_{seed}_end.png")
    files = run(stage3(gn, mn, en, scene, action, seed, f"momento/rise_{stem}_{seed}"))
    fetch(files["final"][0], out)
    subprocess.run(["cp", str(paths["guide"]), str(out).replace(".mp4", "_guide.mp4")])
    print("seed", seed, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
