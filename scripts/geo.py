"""Momento, built on geometry instead of prompts. Runs on Comfy Cloud.

    python scripts/geo.py profile.jpg post.jpg [more posts...]

The post's camera is where the photographer stood. Depth Anything 3 (metric) rebuilds the post as a 3D mesh around
that point, so where the photographer is and which way they face are facts, not prompts:
  1. RenderMesh flies a bystander camera from behind the photographer to over their shoulder: a depth video (the
     camera move) and textured frames.
  2. FLUX 3 Image fills the first frame's gaps and puts the photographer (profile picture) into the box where they
     actually stood, projected from 3D: FLUX 3 layout boxes.
  3. LTX-2.5 films it: the Union Control IC-LoRA follows the depth video (the real camera move through the real
     place), the first frame is FLUX 3's, and the last is pinned to the post's own geometry.
  4. Gemini plans the action and the music, and watches every take; the best one wins.
"""
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import cloud  # noqa: E402
from cloud import CRITIC, LTX, LTX_AUDIO_VAE, LTX_SIGMAS, LTX_TE, LTX_VAE, QUIET, VIDEO_NEGATIVE, Graph  # noqa: E402

W, H, FRAMES, FPS = 256, 512, 97, 24  # half-res IC-LoRA control needs latents divisible by 4
KEYS = 13  # rendered camera positions, one every 8 frames
UNION = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
POST_FOV = 85.0  # Depth Anything 3's default intrinsics, as a vertical fov for a portrait post (calibrated)
LENS = 62.0  # the bystander camera's vertical fov
TARGET = (0.0, -0.1, -4.0)  # what everyone looks at: the scene, in front of the photographer
PULL_BACK = float(os.environ.get("PULL_BACK", "1.5"))  # how far back the bystander stands (1 = the old 1.7 m)
back = lambda x, y, z: (x * PULL_BACK, y, z * PULL_BACK)  # noqa: E731
START = back(-0.4, 0.1, 1.7)  # the bystander: behind and a little left of the photographer, who lands in the right half
END = (0.22, 0.12, 0.38)  # over the photographer's right shoulder

PLAN = """Left image: someone's profile picture. Right image: a photo they posted. Reply with JSON only:
{"scene": "<one sentence: the place, the light, and the subject of the photo (who or what, what they wear)>",
 "action": "<one sentence: what the subject is doing in the second before the photo, moving into exactly the pose or
   instant the photo froze. Active, never posing.>",
 "photographer": "<one short phrase describing the person in the profile picture (or, if it isn't a person, an
   anthropomorphic rabbit in everyday clothes): hair, build, clothes that fit the place>",
 "music": "<an instrumental music prompt fitting this photo's place, time and mood, one or two sentences>"}"""


def ease(t):
    return t * t * (3 - 2 * t)


def path():
    return [tuple(a + (b - a) * ease(i / (KEYS - 1)) for a, b in zip(START, END)) for i in range(KEYS)]


def project(point, eye, target=TARGET, fov=LENS, width=None, height=None):
    """World point -> pixel, for RenderMesh's look-at camera (Y up). Uses the current W, H (set at run time)."""
    width, height = width or W, height or H
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    r = [f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]  # f x up
    n = math.sqrt(sum(v * v for v in r))
    r = [v / n for v in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    p = [a - b for a, b in zip(point, eye)]
    z = sum(a * b for a, b in zip(p, f))
    k = 1 / math.tan(math.radians(fov) / 2)
    x = sum(a * b for a, b in zip(p, r)) / z * k * height / width
    y = sum(a * b for a, b in zip(p, u)) / z * k
    return (x + 1) / 2 * width, (1 - y) / 2 * height


def box(points, eye):
    """FLUX 3 grid box [y_min, x_min, y_max, x_max] around projected world points, clipped to the frame."""
    px = [project(p, eye) for p in points]
    xs, ys = [p[0] for p in px], [p[1] for p in px]
    clip = lambda v, m: max(0, min(1000, round(v / m * 1000)))  # noqa: E731
    return [clip(min(ys), H), clip(min(xs), W), clip(max(ys), H), clip(max(xs), W)]


def layout(eye=None, camera="smartphone"):
    """Where the photographer and their camera are in the frame seen from `eye` (default START): they stand at the
    post's camera, eyes at its height, facing the scene. A smartphone is held out at arm's length in front of the
    eyes; a DSLR is held up to the eye."""
    eye = eye or START
    body = [(x, y, z) for x in (-0.24, 0.24) for y in (0.14, -1.6) for z in (-0.1, 0.15)]
    if camera == "dslr":
        phone = [(x, y, z) for x in (-0.1, 0.1) for y in (-0.06, 0.08) for z in (-0.12, -0.3)]  # at the eye, lens out
    else:
        phone = [(x, y, -0.42) for x in (-0.08, 0.12) for y in (-0.02, 0.16)]  # arm's length, in front of the eyes
    return [{"type": "obj", "bbox": box(body, eye), "desc": "photographer: image 2's person seen from behind and a "
                                                              "little from the side, facing into the scene"},
            {"type": "obj", "bbox": box(phone, eye), "desc": CAMERAS[camera]["box"]}]


# what the photographer holds: the option the app offers ("smartphone" or "dslr")
CAMERAS = {
    "smartphone": dict(hold="holding a phone up in both hands", sheet="hold a phone up at eye level in both hands",
                       act="holds the phone up, aimed at the subject, and taps the shutter at the end",
                       box="phone: held up in both hands, aimed at the subject",
                       click="the crisp click of a phone camera shutter"),
    "dslr": dict(hold="holding a DSLR camera with a lens up to their eye", sheet="hold a DSLR camera with a zoom lens "
                 "up to their eye, taking a photo", act="raises the DSLR to their eye, frames the subject through the "
                 "viewfinder and presses the shutter at the end", box="camera: a DSLR with a lens, held up to the eye, "
                 "aimed at the subject", click="the mechanical clack of a DSLR shutter"),
}


def orbit(eye, degrees):
    """`eye` swung around the photographer (the origin) by `degrees`, same height and distance."""
    a = math.radians(degrees)
    x, y, z = eye
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


# camera moves through the post's 3D space: where the take starts; it always ends at START
MOVES = {
    "dolly in": back(-0.55, 0.15, 3.0),      # from further back, pushing in behind the photographer
    "orbit": orbit(START, 28),               # swinging around from their right (a left orbit loses them)
    "crane down": back(-0.35, 1.3, 2.1),     # from above, settling to eye level
}
MOVE_WORDS = {"dolly in": "a slow dolly in from further back", "orbit": "a slow arc around the photographer",
              "crane down": "a slow crane down from above"}


def field(g, text, name):
    return cloud.field(g, text, name)


def graph(profile, post, seed, prefix):
    g = Graph()
    who = g.add("LoadImage", image=profile)
    shot = g.add("LoadImage", image=post)
    both = g.add("ImageStitch", image1=who(0), image2=shot(0), direction="right", match_image_size=True,
                 spacing_width=0, spacing_color="white")
    plan = g.add("GeminiNode", "Director (Gemini)", prompt="Plan the moment this photo was taken.",
                 model="gemini-3-1-pro", seed=seed, images=both(0), system_prompt=PLAN)
    scene, action, person = (field(g, plan(0), k) for k in ("scene", "action", "photographer"))

    # 1. the post as a place: Depth Anything 3 (metric) -> mesh; the post's camera is the origin
    geo = g.add("DA3Inference", da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                                weight_dtype="default")(),
                image=shot(0), resolution=504, resize_method="upper_bound_resize", mode="mono")
    mesh = g.add("DA3GeometryToMesh", da3_geometry=geo(), batch_index=0, decimation=1, discontinuity_threshold=0.0,  # no tears: no holes
                 confidence_threshold=0.1, use_sky_mask=True, texture=True)
    depth, views = None, []
    for i, eye in enumerate(path()):
        cam = g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", eye)},
                    target_x=TARGET[0], target_y=TARGET[1], target_z=TARGET[2], roll=0.0, fov=LENS, zoom=1.0,
                    camera_type="perspective")
        render = lambda mode: g.add("RenderMesh", mesh=mesh(), mode=mode, width=W, height=H,  # noqa: E731
                                    background="#000000", camera_info=cam())
        views.append(render("texture"))
        d = g.add("RepeatImageBatch", image=render("depth")(0), amount=8)()
        depth = d if depth is None else g.add("ImageBatch", image1=depth, image2=d)()
    depth = g.add("ImageFromBatch", image=depth, batch_index=0, length=FRAMES)()

    # 2. FLUX 3: the first frame, gaps filled, the photographer where they stood (layout boxes from the projection)
    face = g.add("ImageScale", image=who(0), upscale_method="lanczos", width=512, height=512, crop="center")
    first_prompt = g.add("StringConcatenate", delimiter=" ", string_a=(
        "Image 1 is a bystander's view of a real place, with black gaps where the camera sees around things: fill "
        "them seamlessly with more of the same place, same light. In the foreground stands the photographer "
        "<photographer>, the person from image 2, seen from behind and slightly to the side, turned toward the "
        "subject, holding a phone up in both hands <phone>, aimed at the subject. Photographer:"),
        string_b=g.add("StringConcatenate", delimiter=" Scene: ", string_a=person, string_b=scene)())
    boxes = g.add("CreateBoundingBoxes", width=W, height=H, bboxes=json.dumps(layout()),
                  editor_state={"__value__": []})
    first = g.add("Flux3ImageNode", "First frame (FLUX 3)", prompt=first_prompt(), seed=seed, bounding_boxes=boxes(2),
                  aspect_ratio="9:16", resolution="0.75K", grounding=False, safety_tolerance=4,
                  **{"images.image_1": views[0](0), "images.image_2": face()})
    first = g.add("ImageScale", image=first(0), upscale_method="lanczos", width=W, height=H, crop="center")

    # 3. LTX-2.5 + Union Control IC-LoRA on the depth video; last frame pinned to the post's geometry
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    ic = g.add("LTXICLoRALoaderModelOnly", model=g.add("UNETLoader", unet_name=LTX, weight_dtype="default")(),
               lora_name=UNION, strength_model=1.0)
    motion = g.add("StringConcatenate", delimiter=" ", string_a=(
        "A handheld bystander camera moves in behind the photographer, over their shoulder, as they take the photo. "
        "The photographer, seen from behind, holds the phone up, aimed at the subject, and taps the shutter; the "
        "phone's flash fires. The subject:"), string_b=g.add("StringConcatenate", delimiter=" ", string_a=action,
                                                              string_b=scene)())
    text = g.add("StringConcatenate", delimiter="", string_a=motion(), string_b=QUIET)
    cond = g.add("LTXVConditioning", frame_rate=float(FPS),
                 positive=g.add("CLIPTextEncode", text=text(), clip=clip())(),
                 negative=g.add("CLIPTextEncode", text=VIDEO_NEGATIVE, clip=clip())())
    lat = g.add("LTXVImgToVideoInplace", vae=vae(), image=first(), strength=1.0, bypass=False,
                latent=g.add("EmptyLTXVLatentVideo", width=W, height=H, length=FRAMES, batch_size=1)())
    guide = g.add("LTXAddVideoICLoRAGuide", positive=cond(0), negative=cond(1), vae=vae(), latent=lat(), image=depth,
                  frame_idx=0, strength=1.0, latent_downscale_factor=ic(1), crop="disabled", use_tiled_encode=False,
                  tile_size=256, tile_overlap=64, use_streaming=False, tile_frames=97)
    last = g.add("LTXVAddGuide", positive=guide(0), negative=guide(1), vae=vae(), latent=guide(2),
                 image=views[-1](0), frame_idx=-1, strength=0.8, attention_mask=views[-1](1))
    audio = g.add("LTXVEmptyLatentAudio", frames_number=FRAMES, frame_rate=FPS, batch_size=1, audio_vae=avae())
    av = g.add("LTXVConcatAVLatent", video_latent=last(2), audio_latent=audio())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=ic(0), positive=last(0), negative=last(1), cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    video = g.add("LTXVCropGuides", positive=last(0), negative=last(1), latent=sep(0))(2)
    moment = g.add("CreateVideo", images=g.add("VAEDecode", samples=video, vae=vae())(),
                   audio=g.add("LTXVAudioVAEDecode", samples=sep(1), audio_vae=avae())(), fps=float(FPS))
    g.add("SaveVideo", "Momento", video=moment(), filename_prefix=prefix, format="mp4", codec="h264")
    g.add("SaveImage", "Geometry", filename_prefix=prefix + "_geo", images=g.add(
        "ImageBatch", image1=g.add("ImageBatch", image1=views[0](0), image2=first())(), image2=depth)())
    critic = g.add("GeminiNode", "Critic (Gemini)", prompt="Review this take.", model="gemini-3-1-pro", seed=seed,
                   images=both(0), video=moment(), system_prompt=CRITIC)
    g.add("PreviewAny", "Review", source=critic(0))
    return g


def take(pic, img, post, name, n, out):
    t = time.time()
    video, texts = cloud.run(graph(pic, img, seed=random.randrange(2**31), prefix=f"momento/{name}"))
    raw, path_ = Path(out) / f"{name}_geo_take{n}_raw.mp4", Path(out) / f"{name}_geo_take{n}.mp4"
    cloud.download(video, raw)
    cloud.reveal(str(raw), post, str(path_))
    verdict = {}
    for text in texts:
        try:
            verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError:
            pass
    print(f"[{name}] take {n} ({time.time() - t:.0f}s) {path_} review {verdict}", flush=True)
    return (verdict.get("score") or 0) + (10 if verdict.get("pass") else 0), path_, verdict


def shoot(profile, post, takes=3, out="out"):
    name = Path(post).stem
    pic, img = cloud.upload(profile), cloud.upload(post)
    with ThreadPoolExecutor(takes) as pool:
        results = list(pool.map(lambda n: take(pic, img, post, name, n, out), range(1, takes + 1)))
    score, best, verdict = max(results, key=lambda r: r[0])
    print(f"[{name}] BEST {best} {verdict}", flush=True)


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]
    with ThreadPoolExecutor(len(posts)) as pool:
        list(pool.map(lambda p: shoot(profile, p), posts))
