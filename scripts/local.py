"""Momento on your own GPU, open weights only: no Comfy Cloud, no partner nodes.

    COMFY_URL=http://<gpu-host>:8188 PLANNER_URL=http://<ollama-host>:11434/api/chat PLANNER_MODEL=<vision model> \
        [PLANNER=<ssh host, if only it can reach the planner>] [CAMERA=smartphone|dslr] [RES=576x1024] \
        python scripts/local.py [--hires] profile.jpg post.jpg [more posts...]

  1. A local vision LLM (any Ollama-API chat model) looks at the profile picture and the post and plans the moment.
  2. FLUX 3 Klein draws a character sheet of the account owner; Depth Anything 3 rebuilds the post as a 3D place
     (the post's camera is where the photographer stood) and RenderMesh renders the bystander's view; Klein fills it
     in, then puts the photographer into it with FLUX 3 layout boxes projected from 3D.
  3. MiniMax H3 films the seconds before, ending on that frame. Its soundscape prompt keeps it free of speech.
  4. The planner reviews frames of each take; the best take wins.
"""
import base64
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
HIRES = "--hires" in sys.argv  # read before moment (imported below) consumes the flag
import comfy  # noqa: E402
import geo  # noqa: E402
from cloud import reveal  # noqa: E402
from graph import Graph  # noqa: E402
from moment import CLASSIFY, PLAN  # noqa: E402

W, H, MP = (448, 768, 0.4) if HIRES else (256, 512, 0.15)
if os.environ.get("RES"):  # e.g. RES=576x1024 on a bigger GPU
    W, H = (int(v) for v in os.environ["RES"].split("x"))
    MP = round(W * H / 1e6, 2)
if "--hires" in sys.argv:
    sys.argv.remove("--hires")
geo.W, geo.H = W, H
H3_FRAMES, FPS = 107, 24
CAMERA = os.environ.get("CAMERA", "smartphone")  # what the photographer holds: smartphone or dslr
assert CAMERA in geo.CAMERAS, f"CAMERA must be one of {list(geo.CAMERAS)}"
CAM = geo.CAMERAS[CAMERA]
PLANNER = os.environ.get("PLANNER")  # optional: an ssh host to call the planner through
PLANNER_URL = os.environ.get("PLANNER_URL", "http://127.0.0.1:11434/api/chat")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "qwen3-vl")

KLEIN, KLEIN_TE, FLUX_VAE = ("flux3-image-klein-int8-convrot.safetensors", "qwen3vl_4b_int8_convrot.safetensors",
                             "flux2-vae.safetensors")
H3 = dict(unet=os.environ.get("H3_UNET", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
          lora="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
          vae="minimax_h3_video_vae_int8_convrot.safetensors", audio_vae="minimax_h3_audio_vae_fp32.safetensors",
          clip="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")

SHEET = ("Character turnaround sheet, four full-body views side by side on a plain light grey studio background: "
         "front, three-quarter front, side profile, and back. Same face, hair, build and clothes in every view. In the "
         f"side and back views they {CAM['sheet']}, taking a photo. Photographic, even "
         "lighting. If ref_image_0 shows a person, it is them, exactly; if not, the character is the anonymous "
         "everyday person described here. The character: ")

CRITIC = """These are frames, in order, from a generated video of a photographer taking a photo, seen by a bystander:
the photographer is in the foreground (often from behind), aiming a phone or camera at the scene. The LAST image is the
real photo they took. The video is NOT supposed to end on that photo: it ends on the bystander's view of the same
scene, photographer included, and then cuts to the photo. Judge the video frames as a short film of that moment.
Reject only for failures a viewer would notice at a glance: a person melting, morphing, multiplying or popping in and
out; extra limbs; hands and camera fusing into a blob; the subject frozen instead of moving; the scene not being the
real photo's place and subject; text or watermarks. Score how good a take it is overall. Reply with JSON only:
{"pass": true or false, "score": 1-10, "problem": "<the worst problem in one short sentence, or empty>"}"""


def ask(system, text, images):
    """The planner: an Ollama-style chat API (through PLANNER over ssh, if set)."""
    body = json.dumps({"model": PLANNER_MODEL, "stream": False, "format": "json",
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": text,
                                     "images": [base64.b64encode(Path(i).read_bytes()).decode() for i in images]}]})
    for i in range(5):
        curl = ["curl", "-s", "-m", "600", PLANNER_URL, "-d", "@-"]
        r = subprocess.run(["ssh", PLANNER, " ".join(curl)] if PLANNER else curl, input=body, capture_output=True,
                           text=True)
        try:
            reply = json.loads(r.stdout)["message"]["content"]
            return json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
        except (ValueError, KeyError):
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"planner failed: {r.stdout[:500]} {r.stderr[:300]}")


def klein(g, prompt, refs, seed, width=W, height=H, steps=16):
    """FLUX 3 Klein (open weights): prompt + reference images (ref_image_0, ref_image_1, ...)."""
    clip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    vae = g.add("VAELoader", vae_name=FLUX_VAE)
    model = g.add("ModelSamplingFlux", model=g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")(),
                  max_shift=2.77, base_shift=2.77, width=width, height=height)
    refd = g.add("ReferenceMultiLatent", vae=vae(), resolution="resize", **{"resolution.megapixels": MP},
                 positive=g.add("CLIPTextEncode", text=prompt, clip=clip())(),
                 negative=g.add("CLIPTextEncode", text="", clip=clip())(),
                 **{f"images.image{i}": r for i, r in enumerate(refs)})
    out = g.add("KSampler", model=model(), seed=seed, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple",
                positive=refd(0), negative=refd(1), denoise=1.0,
                latent_image=g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)())
    return g.add("VAEDecode", samples=out(), vae=vae())


def rows(layout):
    """FLUX 3 layout rows (the photographer and phone boxes from 3D) for Klein's prompt."""
    return " " + json.dumps([{"id": r["desc"].split(":")[0], "bbox": r["bbox"], "desc": r["desc"]} for r in layout])


def graph(profile, post, plan, seed, prefix, closeup, move=None):
    g = Graph()
    who = g.add("LoadImage", image=profile)
    shot = g.add("LoadImage", image=post)
    sheet = klein(g, SHEET + plan["photographer"], [who(0)], seed, width=768, height=448)
    g.add("SaveImage", images=sheet(), filename_prefix=prefix + "_sheet")
    if closeup:
        instant = klein(g, "Zoom out from ref_image_0: the same instant seen by a bystander a couple of steps back. "
                           "ref_image_0's view, unchanged, fills the middle of the frame, with the surroundings it "
                           "cropped out all around it, same light and lens. No people in the foreground. "
                        + plan["scene"], [shot(0)], seed)
    else:  # the post as a 3D place; the bystander stands behind the photographer
        geom = g.add("DA3Inference", image=shot(0), resolution=504, resize_method="upper_bound_resize", mode="mono",
                     da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                     weight_dtype="default")())
        mesh = g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1,
                     discontinuity_threshold=0.04, confidence_threshold=0.1, use_sky_mask=True, texture=True)
        cam = g.add("CreateCameraInfo", mode="look_at",
                    **{f"mode.position_{a}": v for a, v in zip("xyz", geo.START)},
                    target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0, fov=geo.LENS,
                    zoom=1.0, camera_type="perspective")
        view = g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background="#000000",
                     camera_info=cam())
        if move:  # the opening view of the camera move, from the same 3D place
            cam0 = g.add("CreateCameraInfo", mode="look_at",
                         **{f"mode.position_{a}": v for a, v in zip("xyz", geo.MOVES[move])},
                         target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0,
                         fov=geo.LENS, zoom=1.0, camera_type="perspective")
            view0 = g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background="#000000",
                          camera_info=cam0())
        instant = klein(g, "ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees "
                           "around things: fill them seamlessly with more of the same place, same light, same lens. "
                           "Keep everything else exactly as it is. No people in the foreground. " + plan["scene"],
                        [view(0)], seed)
    fill = ("ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around things: "
            "fill them seamlessly with more of the same place, same light, same lens. Keep everything else exactly as "
            "it is. No people in the foreground. " + plan["scene"])

    def place(img, eye, s_):
        return klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in "
                        f"the foreground, seen from behind and slightly to the side, turned toward the subject, "
                        f"{CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the sheet. "
                        "Exactly one photographer, no other copies of them. Change nothing else in ref_image_0. "
                     + plan["photographer"] + rows(geo.layout(eye, CAMERA)), [img, sheet()], s_)

    opening = place(klein(g, fill, [view0(0)], seed + 2)(), geo.MOVES[move], seed + 3) if move and not closeup else None
    placed = klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the "
                      "foreground, seen from behind and slightly to the side, turned toward the subject, "
                      f"{CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the "
                      "sheet. Exactly one photographer, no other copies of them. Change nothing else in ref_image_0. "
                   + plan["photographer"] + rows(geo.layout(camera=CAMERA)),
                   [instant(), sheet()], seed + 1)
    g.add("SaveImage", images=g.add("ImageBatch", image1=instant(), image2=placed())(), filename_prefix=prefix + "_keys")
    # MiniMax H3, open weights: the seconds before, ending on the placed instant; soundscape only, nobody speaks
    model = g.add("LoraLoaderModelOnly", lora_name=H3["lora"], strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    vae = g.add("VAELoader", vae_name=H3["vae"])
    avae = g.add("VAELoader", vae_name=H3["audio_vae"])
    text = (f"integrated_multimodal_description: Handheld bystander camera, nearly still. The photographer in the "
            f"foreground, seen from behind, {CAM['act']}. "
            f"There is exactly one photographer; no one else appears who looks like them. "
            f"The subject is in motion and ends exactly in the moment of <Picture 1>: {plan['action']} {plan['scene']}"
            f"\noverall_soundscape: {plan['soundscape'].replace('phone camera shutter', CAMERA == 'dslr' and 'DSLR shutter' or 'phone camera shutter')}"
            f"\nnon_diegetic_music: N/A")
    if opening:
        text = text.replace("Handheld bystander camera, nearly still.", f"The camera makes {geo.MOVE_WORDS[move]}, "
                            f"moving through the real space of <Picture 1> to <Picture 2>.")
    ends = {"first_frame": opening(), "last_frame": placed()} if opening else {"last_frame": placed()}
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=H3_FRAMES, **ends,
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")(), vae=vae())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    video = g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                  audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())
    g.add("SaveVideo", video=video(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def run(g, timeout=3600):
    """Queue on the GPU host and wait. -> {filename_prefix tail: [file infos]}"""
    body = json.dumps({"prompt": g, "client_id": comfy.CLIENT_ID}).encode()
    res = json.loads(comfy._req("/prompt", body, {"Content-Type": "application/json"}))
    if res.get("node_errors"):
        raise RuntimeError(json.dumps(res["node_errors"])[:3000])
    pid, t = res["prompt_id"], time.time()
    while not (h := json.loads(comfy._req(f"/history/{pid}")).get(pid)) or h["status"].get("completed") is None:
        if time.time() - t > timeout:
            raise TimeoutError(pid)
        time.sleep(3)
    if h["status"].get("status_str") == "error":
        raise RuntimeError(json.dumps([m for m in h["status"]["messages"] if m[0] == "execution_error"])[:3000])
    files = {}
    for node, out in h["outputs"].items():
        if g.get(node, {}).get("class_type") in ("SaveImage", "SaveVideo"):
            tag = g[node]["inputs"]["filename_prefix"].split("_")[-1]
            files[tag] = out.get("images", []) + out.get("videos", [])
    return files


def fetch(f, dest):
    q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
    Path(dest).write_bytes(comfy._req(f"/view?{q}"))


def take(profile_up, post_up, profile, post, plan, closeup, n, out):
    name = Path(post).stem
    tag = f"{name}_{os.environ.get('TAG', 'local')}{'_dslr' if CAMERA == 'dslr' else ''}{n}"
    t = time.time()
    moves = list(geo.MOVES) if os.environ.get("FORCE_MOVE") else [None, *geo.MOVES]  # FORCE_MOVE: never still
    move = None if closeup else random.choice(moves)  # variety: a still bystander or a camera move
    print(f"[{tag}] camera: {move or 'still'}", flush=True)
    files = run(graph(profile_up, post_up, plan, random.randrange(2**31), f"momento/{tag}", closeup, move))
    od = Path(out) / "steps"
    od.mkdir(parents=True, exist_ok=True)
    fetch(files["sheet"][0], od / f"{tag}_sheet.png")
    for i, f in enumerate(files["keys"]):
        fetch(f, od / f"{tag}_key{i}.png")
    raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
    fetch(files["final"][0], raw)
    reveal(str(raw), post, str(final))
    frames = Path(tempfile.mkdtemp())
    for i, s in enumerate((0.5, 2.0, 3.5, 4.3)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s), "-i", str(raw), "-frames:v", "1",
                        "-vf", "scale=-2:384", str(frames / f"{i}.jpg")], check=True)
    verdict = ask(CRITIC, "Review this take.", sorted(map(str, frames.glob("*.jpg"))) + [post])
    print(f"[{tag}] take {n} ({time.time() - t:.0f}s) {final} review {verdict}", flush=True)
    return (verdict.get("score") or 0) + (10 if verdict.get("pass") else 0), final, verdict


def shoot(profile, post, takes=int(os.environ.get("TAKES", 2)), out="out"):
    name = Path(post).stem
    closeup = "CLOSEUP" in json.dumps(ask(CLASSIFY + ' Reply as JSON: {"kind": "CLOSEUP" or "SCENE"}', "Classify.",
                                          [post])).upper()
    plan = ask(PLAN, "Left: the profile picture. Right: the photo they posted. Plan the moment.", [profile, post])
    print(f"[{name}] {'CLOSEUP' if closeup else 'SCENE'} plan: {json.dumps(plan)[:300]}", flush=True)
    pu, su = comfy.upload(profile, f"momento_{Path(profile).name}"), comfy.upload(post, f"momento_{Path(post).name}")
    results = [take(pu, su, profile, post, plan, closeup, n, out) for n in range(1, takes + 1)]  # one GPU: in turn
    score, best, verdict = max(results, key=lambda r: r[0])
    print(f"[{name}] BEST {best} {verdict}", flush=True)


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]
    for p in posts:
        shoot(profile, p)
