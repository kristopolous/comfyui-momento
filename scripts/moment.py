"""Momento in steps, each on the model built for it. Runs on Comfy Cloud.

    python scripts/moment.py profile.jpg post.jpg [more posts...]

  A. Plan + character sheet + keyframes: Gemini plans the action. FLUX 3 Image draws a character sheet of the account
     owner (turnaround, phone up) from the profile picture. Depth Anything 3 rebuilds the post as a 3D place; the
     post's camera is where the photographer stood. RenderMesh renders the bystander's start and end views and FLUX 3
     fills them in: the start a beat before the instant, the end the instant itself.
  B. The scene, no photographer: MiniMax H3 first/last frame to video, the subject acting into the instant.
  C. The photographer, placed: FLUX 3 puts the character sheet's person into H3's first frame, in the layout box
     projected from where they stood (3D), facing the subject.
  D. The photographer, added: LTX-2.5 in/outpainting IC-LoRA regenerates only the photographer's region (a mask
     video, also projected from 3D) through the whole H3 clip; the rest stays H3's. Gemini reviews the take.
"""
import json
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import cloud  # noqa: E402
import geo  # noqa: E402
from cloud import CRITIC, LTX, LTX_AUDIO_VAE, LTX_SIGMAS, LTX_TE, LTX_VAE, QUIET, VIDEO_NEGATIVE, Graph  # noqa: E402

W, H = geo.W, geo.H
FLUX_RES = "0.75K"
if "--hires" in sys.argv:  # finals only; iterate at the lowest resolution
    sys.argv.remove("--hires")
    W, H, FLUX_RES = 448, 768, "1K"
    geo.W, geo.H = W, H
H3_FRAMES = 107  # MiniMax H3's 17k+5 grid, ~4.5 s
LTX_FRAMES = 105  # LTX's 8n+1, the H3 clip minus two frames
FPS = 24
INPAINT = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

PLAN = """Left image: someone's profile picture. Right image: a photo they posted. Reply with JSON only:
{"scene": "<one sentence: the place, the light, and the subject of the photo (who or what, what they wear)>",
 "before": "<one sentence: the scene three or four seconds BEFORE the photo, an earlier stage of the same action
   (surfers still climbing out of the helicopter with their boards; a singer crouched at the mic before throwing her
   arm up; a bus still down the street)>",
 "action": "<one sentence: what the subject does over four seconds, ending in exactly the pose or instant the photo
   froze. Active, never posing.>",
 "photographer": "<one short phrase: the person in the profile picture (or, if it isn't a person, an anonymous
   everyday person whose look fits the place): hair, build, and clothes that fit the place of the photo>",
 "soundscape": "<1-4 sentences of the place's real sound only: ambient sound, physical action sounds and
   non-verbal human sounds (crowd roar, footsteps, traffic, wind, guitar), ending with the crisp click of a phone camera
   shutter. No speech, no singing words.>",
 "music": "<an instrumental music prompt fitting this photo's place, time and mood, one or two sentences>"}"""

SHEET = ("Character turnaround sheet, four full-body views side by side on a plain light grey studio background: "
         "front, three-quarter front, side profile, and back. Same face, hair, build and clothes in every view. In the "
         "side and back views they hold a phone up at eye level in both hands, taking a photo. Photographic, even "
         "lighting. If image 1 shows a person, it is them, exactly; if it doesn't, the character is the anonymous "
         "everyday person described here. The character:")

CLASSIFY = """Look at this photo. Is it a CLOSEUP (a flat or near subject filling the frame: an object on a table, a
laptop lid, food, a document, a face) or a SCENE (a place with depth: a street, a stage, a building, a landscape)?
Reply with one word: CLOSEUP or SCENE."""


def h3(g, first, last, prompt, seed):
    """MiniMax H3 first/last frame to video+audio, as Comfy's template runs it (turbo 8 steps)."""
    model = g.add("LoraLoaderModelOnly", strength_model=1.0, lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                  model=g.add("UNETLoader", unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                              weight_dtype="default")())
    vae = g.add("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    avae = g.add("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    clip = g.add("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    frames = {k: v for k, v in (("first_frame", first), ("last_frame", last)) if v is not None}
    cond = g.add("MiniMaxH3ImageToVideo", clip=clip(), vae=vae(), prompt=prompt, width=W, height=H, length=H3_FRAMES,
                 **frames)
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    return g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                 audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())


def stage_a(profile, post, seed, prefix, closeup=False):
    """Plan, character sheet, the instant, then MiniMax H3 films the scene (A + B in one graph). A SCENE's bystander
    view is rendered from its 3D geometry; a CLOSEUP has no useful depth, so FLUX 3 zooms out from the post itself."""
    g = Graph()
    who = g.add("LoadImage", image=profile)
    shot = g.add("LoadImage", image=post)
    both = g.add("ImageStitch", image1=who(0), image2=shot(0), direction="right", match_image_size=True,
                 spacing_width=0, spacing_color="white")
    plan = g.add("GeminiNode", "Director (Gemini)", prompt="Plan the moment this photo was taken.",
                 model="gemini-3-1-pro", seed=seed, images=both(0), system_prompt=PLAN)
    g.add("PreviewAny", "Plan", source=plan(0))
    f = lambda k: cloud.field(g, plan(0), k)  # noqa: E731
    cat = lambda a, b, d=" ": g.add("StringConcatenate", string_a=a, string_b=b, delimiter=d)()  # noqa: E731
    flux = dict(resolution=FLUX_RES, grounding=False, safety_tolerance=4)
    face = g.add("ImageScale", image=who(0), upscale_method="lanczos", width=512, height=512, crop="center")
    sheet = g.add("Flux3ImageNode", "Character sheet (FLUX 3)", prompt=cat(SHEET, f("photographer")), seed=seed,
                  aspect_ratio="16:9", **flux, **{"images.image_1": face()})
    g.add("SaveImage", images=sheet(0), filename_prefix=prefix + "_sheet")

    if closeup:  # no depth worth rebuilding: FLUX 3 zooms out from the post to the bystander's view
        instant = g.add("Flux3ImageNode", "The instant, zoomed out (FLUX 3)", seed=seed, aspect_ratio="9:16", **flux,
                        prompt=cat("Zoom out from image 1: the same instant seen by a bystander a couple of steps back. "
                                   "Image 1's view, unchanged, now fills the middle of the frame, with the surroundings "
                                   "it cropped out all around it, in the same light and lens. No people in the "
                                   "foreground.", f("scene")),
                        **{"images.image_1": shot(0)})
    else:
        # the post as a 3D place, seen by the bystander at the start and the end of the move
        geom = g.add("DA3Inference", image=shot(0), resolution=504, resize_method="upper_bound_resize", mode="mono",
                     da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                     weight_dtype="default")())
        mesh = g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1, discontinuity_threshold=0.04,
                     confidence_threshold=0.1, use_sky_mask=True, texture=True)

        def view(eye):
            cam = g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", eye)},
                        target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0, fov=geo.LENS,
                        zoom=1.0, camera_type="perspective")
            return g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background="#000000",
                         camera_info=cam())(0)

        instant = g.add("Flux3ImageNode", "The instant, bystander's view (FLUX 3)", seed=seed, aspect_ratio="9:16", **flux,
                        prompt=cat("Image 1 is a bystander's view of a real place, with black gaps where the camera sees "
                                   "around things: fill them seamlessly with more of the same place, same light, same "
                                   "lens. Keep everything else in image 1 exactly as it is. No people in the foreground.",
                                   f("scene")),
                        **{"images.image_1": view(geo.START)})
    fit = lambda img: g.add("ImageScale", image=img, upscale_method="lanczos", width=W, height=H, crop="center")()  # noqa
    g.add("SaveImage", images=fit(instant(0)), filename_prefix=prefix + "_keys")
    # B. MiniMax H3: the scene, no photographer. Only the last frame is given; H3 invents the action into it.
    # MiniMax H3's prompt structure: speech only comes from speakers tagged <d>; nobody is, so nobody talks
    motion = cat("integrated_multimodal_description: Handheld bystander camera, nearly still. The subject is in "
                 "motion and ends exactly in the moment of <Picture 1>:", cat(f("action"), f("scene")))
    sound = cat("\noverall_soundscape:", f("soundscape"))
    scene = h3(g, None, fit(instant(0)), cat(cat(motion, sound, ""), "\nnon_diegetic_music: N/A", ""), seed)
    g.add("SaveVideo", "Scene (MiniMax H3)", video=scene(), filename_prefix=prefix + "_scene", format="mp4",
          codec="h264")
    return g


def masks(path, frames=LTX_FRAMES, pad=None):
    """The photographer's region in every frame, projected from 3D along the camera path, as a mask video."""
    d = Path(tempfile.mkdtemp())
    pad = pad if pad is not None else round(W / 25)
    body = [(x, y, z) for x in (-0.26, 0.26) for y in (0.18, -1.7) for z in (-0.5, 0.2)]
    for i in range(frames):
        eye = geo.START  # the bystander holds still; H3 moves the subject, not the camera
        y0, x0, y1, x1 = geo.box(body, eye)
        im = Image.new("RGB", (W, H), "black")
        ImageDraw.Draw(im).rectangle((x0 * W / 1000 - pad, y0 * H / 1000 - pad, x1 * W / 1000 + pad, y1 * H / 1000 + pad),
                                     fill="white")
        im.save(d / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "10", str(path)], check=True)


def stage_cd(profile, post, sheet, scene, mask, plan, seed, prefix):
    """C: FLUX 3 places the photographer into H3's first frame. D: LTX-2.5 inpaints them through the clip."""
    g = Graph()
    who = g.add("LoadImage", image=profile)
    frames = g.add("GetVideoComponents", video=g.add("LoadVideo", file=scene)())
    src = g.add("ImageFromBatch", image=frames(0), batch_index=0, length=LTX_FRAMES)
    hole = g.add("ImageToMask", channel="red", image=g.add("ImageFromBatch", batch_index=0, length=LTX_FRAMES,
                                                         image=g.add("GetVideoComponents", video=g.add(
                                                             "LoadVideo", file=mask)())(0))())
    boxes = g.add("CreateBoundingBoxes", width=W, height=H, bboxes=json.dumps(geo.layout()),
                  editor_state={"__value__": []})
    sheet_img = g.add("LoadImage", image=sheet)

    def place(index, label, s_):
        img = g.add("Flux3ImageNode", label, seed=s_, bounding_boxes=boxes(2), aspect_ratio="9:16",
                    resolution=FLUX_RES, grounding=False, safety_tolerance=4,
                    prompt=("Add the person from image 2's character sheet into image 1 as <photographer>: in the "
                            "foreground, seen from behind and slightly to the side, turned toward the subject, holding "
                            "a phone up in both hands <phone>, aimed at the subject, taking a photo. Same hair and "
                            "clothes as the sheet. Change nothing else in image 1. " + plan.get("photographer", "")),
                    **{"images.image_1": g.add("ImageFromBatch", image=frames(0), batch_index=index, length=1)(),
                       "images.image_2": sheet_img(0)})
        return g.add("ImageScale", image=img(0), upscale_method="lanczos", width=W, height=H, crop="center")

    first = place(0, "Photographer placed, first frame (FLUX 3)", seed)
    last = place(LTX_FRAMES - 1, "Photographer placed, the instant (FLUX 3)", seed + 1)
    g.add("SaveImage", images=g.add("ImageBatch", image1=first(), image2=last())(), filename_prefix=prefix + "_placed")
    # D. LTX-2.5 in/outpainting IC-LoRA: only the photographer's region is regenerated
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    ic = g.add("LTXICLoRALoaderModelOnly", model=g.add("UNETLoader", unet_name=LTX, weight_dtype="default")(),
               lora_name=INPAINT, strength_model=1.0)
    text = (f"{plan.get('photographer', 'The photographer')}, seen from behind in the foreground, holds a phone up "
            f"in both hands, aimed at the subject, frames the shot and taps the shutter at the end: the phone's flash "
            f"fires. {plan.get('action', '')} {plan.get('scene', '')}{QUIET}")
    cond = g.add("LTXVConditioning", frame_rate=float(FPS),
                 positive=g.add("CLIPTextEncode", text=text, clip=clip())(),
                 negative=g.add("CLIPTextEncode", text=VIDEO_NEGATIVE, clip=clip())())
    lat = g.add("LTXVImgToVideoInplace", vae=vae(), image=first(), strength=1.0, bypass=False,
                latent=g.add("EmptyLTXVLatentVideo", width=W, height=H, length=LTX_FRAMES, batch_size=1)())
    guide = g.add("LTXAddVideoICLoRAGuideAdvanced", positive=cond(0), negative=cond(1), vae=vae(), latent=lat(),
                  image=g.add("LTXVInpaintPreprocess", images=src(), mask=hole())(), frame_idx=0, strength=1.0,
                  latent_downscale_factor=ic(1), crop="disabled", use_tiled_encode=False, tile_size=256,
                  tile_overlap=64, attention_strength=1.0, use_streaming=False, tile_frames=97)
    guide = g.add("LTXVAddGuide", positive=guide(0), negative=guide(1), vae=vae(), latent=guide(2), image=last(),
                  frame_idx=-1, strength=1.0)  # the photographer at the instant, pinned
    audio = g.add("LTXVEmptyLatentAudio", frames_number=LTX_FRAMES, frame_rate=FPS, batch_size=1, audio_vae=avae())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=ic(0), positive=guide(0), negative=guide(1), cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(),
                latent_image=g.add("LTXVConcatAVLatent", video_latent=guide(2), audio_latent=audio())())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    video = g.add("LTXVCropGuides", positive=guide(0), negative=guide(1), latent=sep(0))(2)
    moment = g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=video, vae=vae())(),
                   audio=frames(1))  # the scene's own sound from H3: soundscape only, no tagged speakers
    g.add("SaveVideo", "Momento", video=moment(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    critic = g.add("GeminiNode", "Critic (Gemini)", prompt="Review this take.", model="gemini-3-1-pro", seed=seed,
                   images=g.add("ImageStitch", image1=who(0), image2=g.add("LoadImage", image=post)(0),
                                direction="right", match_image_size=True, spacing_width=0, spacing_color="white")(),
                   video=moment(), system_prompt=CRITIC)
    g.add("PreviewAny", "Review", source=critic(0))
    return g


def collect(graph):
    """Run a graph on Cloud and poll until it ends (survives dropped connections).
    -> ({filename_prefix tail: [files]}, texts)"""
    import requests
    key = cloud.key()
    hdr = {"X-API-Key": key}
    res = cloud.retry(requests.post, f"{cloud.CLOUD}/api/prompt", headers=hdr, timeout=120,
                      json={"prompt": graph, "extra_data": {"api_key_comfy_org": key}}).json()
    if "prompt_id" not in res:
        raise RuntimeError(json.dumps(res)[:3000])
    pid = res["prompt_id"]
    while True:
        time.sleep(5)
        try:
            status = requests.get(f"{cloud.CLOUD}/api/job/{pid}/status", headers=hdr, timeout=30).json()["status"]
        except (requests.RequestException, ValueError, KeyError):
            continue
        if status in ("success", "error", "non_retryable_error", "lost", "cancelled"):
            break
    hist = cloud.retry(requests.get, f"{cloud.CLOUD}/api/history_v2/{pid}", headers=hdr, timeout=60).json()[pid]
    if status != "success":
        msgs = [m for m in hist.get("status", {}).get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError(f"{status}: {json.dumps(msgs)[:2000]}")
    files, texts = {}, []
    for node, out in hist["outputs"].items():
        tag = graph[node]["inputs"].get("filename_prefix", node).split("_")[-1] if node in graph else node
        for f in out.get("images", []) + out.get("video", []) + out.get("videos", []):
            if graph.get(node, {}).get("class_type") in ("SaveImage", "SaveVideo"):  # not the loaded inputs
                files.setdefault(tag, []).append(f)
        texts += [t for t in out.get("text", []) if isinstance(t, str)]
    return files, texts


def classify(img):
    """CLOSEUP or SCENE, from Gemini: which route builds the bystander's view."""
    g = Graph()
    g.add("PreviewAny", source=g.add("GeminiNode", prompt="Classify this photo.", model="gemini-3-1-pro", seed=1,
                                     images=g.add("LoadImage", image=img)(0), system_prompt=CLASSIFY)(0))
    _, texts = collect(g)
    return any("CLOSEUP" in t.upper() for t in texts)


def take(pic, img, post, name, n, out, closeup=False):
    t = time.time()
    seed = random.randrange(2**31)
    tag = f"{name}_mo{n}"
    files, texts = collect(stage_a(pic, img, seed, f"momento/{tag}", closeup))
    plan = {}
    for text in texts:
        try:
            plan = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError:
            pass
    od = Path(out) / "steps"
    od.mkdir(parents=True, exist_ok=True)
    cloud.download(files["sheet"][0], od / f"{tag}_sheet.png")
    for i, f in enumerate(files.get("keys", [])):  # the instant
        cloud.download(f, od / f"{tag}_key{i}.png")
    cloud.download(files["scene"][0], od / f"{tag}_scene.mp4")
    print(f"[{tag}] A+B done ({time.time() - t:.0f}s): sheet, keys, H3 scene", flush=True)
    masks(od / f"{tag}_mask.mp4")
    up = lambda p: cloud.upload(str(p))  # noqa: E731
    (od / f"{tag}_plan.json").write_text(json.dumps(plan))
    files, texts = collect(stage_cd(pic, img, up(od / f"{tag}_sheet.png"), up(od / f"{tag}_scene.mp4"),
                                    up(od / f"{tag}_mask.mp4"), plan, seed, f"momento/{tag}"))
    raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
    cloud.download(files["final"][0], raw)
    for i, f in enumerate(files.get("placed", [])):
        cloud.download(f, od / f"{tag}_placed{i}.png")
    cloud.reveal(str(raw), post, str(final))
    verdict = {}
    for text in texts:
        try:
            verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError:
            pass
    print(f"[{tag}] take {n} ({time.time() - t:.0f}s) {final} review {verdict}", flush=True)
    return (verdict.get("score") or 0) + (10 if verdict.get("pass") else 0), final, verdict


def shoot(profile, post, takes=3, out="out"):
    name = Path(post).stem
    pic, img = cloud.upload(profile), cloud.upload(post)
    closeup = classify(img)
    print(f"[{name}] {'CLOSEUP: zoom out with FLUX 3' if closeup else 'SCENE: 3D geometry'}", flush=True)
    with ThreadPoolExecutor(takes) as pool:
        results = list(pool.map(lambda n: take(pic, img, post, name, n, out, closeup), range(1, takes + 1)))
    score, best, verdict = max(results, key=lambda r: r[0])
    print(f"[{name}] BEST {best} {verdict}", flush=True)


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]
    with ThreadPoolExecutor(len(posts)) as pool:
        list(pool.map(lambda p: shoot(profile, p), posts))
