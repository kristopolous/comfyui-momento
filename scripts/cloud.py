"""Shoot moments on Comfy Cloud instead of your own GPU: many takes at once, the critic picks.

    python scripts/cloud.py profile.jpg post1.jpg [post2.jpg ...]

The Cloud graph uses only nodes Comfy Cloud has: Gemini plans and reviews, FLUX 3 Image (partner node) paints the
instant and the second before it, LTX-2.5 films it. Momento's own nodes aren't there, so the reveal (flash to the
real post) is cut locally with ffmpeg. Needs COMFY_API_KEY (.env) on a paid Comfy Cloud plan.
"""
import json
import random
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import websocket

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comfyui_momento"))
from director import CRITIC, QUIET, VIDEO_NEGATIVE  # noqa: E402
from graph import LTX, LTX_AUDIO_VAE, LTX_SIGMAS, LTX_TE, LTX_VAE, Graph  # noqa: E402
from run import key  # noqa: E402

CLOUD = "https://cloud.comfy.org"


def retry(fn, *args, tries=30, **kw):
    """Keep trying through dropped connections (laptop on a train)."""
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except (requests.RequestException, OSError):
            if i == tries - 1:
                raise
            time.sleep(min(30, 2 + 2 * i))
TAKES = 3


PLAN = """Left image: someone's profile picture. Right image: a photo they posted.
Plan the real moment that photo was taken, as a bystander saw it: BOTH people in frame. The photographer (the person
in the profile picture; if it isn't a person, an anthropomorphic rabbit in everyday clothes) is in the foreground to
one side, seen in three-quarter view so their face is visible, phone raised toward the subject; the subject of the
post is beyond them, in the real place, ACTIVE, never posing. The photographer catches a MOMENT: the instant the post
froze, lit by the phone's flash. Keep the person exactly as the profile picture shows them (no added jewelry, glasses,
hats). Realistic, flattering. No jokes, no dialogue. Reply with JSON only:
{"still": "<prompt for FLUX 3 Image, image 1 = the post, image 2 = the photographer: the shutter instant. Cite the
   layout ids <photographer>, <phone>, <subject>. The photographer from image 2, face visible, phone raised, its flash
   firing a bright white burst that lights the scene; the subject from image 1 in exactly the post's pose and outfit,
   in the real place of image 1 (name it, its light). One vertical frame.>",
 "boxes": [{"id": "photographer", "bbox": [y_min, x_min, y_max, x_max], "desc": "..."},
           {"id": "phone", "bbox": [...], "desc": "..."}, {"id": "subject", "bbox": [...], "desc": "..."}],
 "before": "<prompt for FLUX 3 Image, image 1 = that shutter frame, image 2 = the photographer: the same view one
   second EARLIER, no flash. The subject is mid-action, a beat before the instant (say what they are doing); the
   photographer, face visible, is raising the phone.>",
 "video": "<prompt for an image-to-video model, 4 seconds, no speech: the subject's action (name it) builds into the
   instant; the photographer, face visible, raises the phone, frames, and taps the shutter exactly on it: the phone's
   flash fires a bright white burst. Handheld bystander camera, steady. Sound: the place's ambient sound and a crisp
   shutter click. Nobody talks.>",
 "music": "<an instrumental music prompt fitting this post's place, time and mood, one or two sentences>"}
Boxes are [y_min, x_min, y_max, x_max] on a 0-1000 grid of the vertical frame."""


def field(g, text, name):
    """One string field of Gemini's JSON reply, with core nodes."""
    return g.add("RegexExtract", string=text, regex_pattern=rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"',
                 mode="First Group", case_insensitive=True, multiline=False, dotall=True, group_index=1)()


def cloud_graph(profile, post, seed, width=288, height=512, frames=97, fps=24, prefix="momento/cloud"):
    """Keyframes: 0 a second earlier, the last frame the instant, flash firing. The reveal's white flash continues
    the camera flash into the real post."""
    g = Graph()
    who = g.add("LoadImage", image=profile)
    shot = g.add("LoadImage", image=post)
    both = g.add("ImageStitch", image1=who(0), image2=shot(0), direction="right", match_image_size=True,
                 spacing_width=0, spacing_color="white")
    plan = g.add("GeminiNode", "Director (Gemini)", prompt="Plan the moment this post was taken.",
                 model="gemini-3-1-pro", seed=seed, images=both(0), system_prompt=PLAN)
    # FLUX 3 layout: Gemini's boxes for the photographer, the phone and the subject
    boxes = g.add("RegexExtract", string=plan(0), regex_pattern=r'"boxes"\s*:\s*(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])',
                  mode="First Group", case_insensitive=True, multiline=False, dotall=True, group_index=1)
    layout = g.add("CreateBoundingBoxes", width=width, height=height, bboxes=boxes(),
                   editor_state={"__value__": []})  # a list widget value, not a link
    flux = dict(aspect_ratio="9:16", resolution="0.75K", grounding=False, safety_tolerance=4)
    face = g.add("ImageScale", image=who(0), upscale_method="lanczos", width=512, height=512, crop="center")  # >= 256
    fit = lambda img: g.add("ImageScale", image=img, upscale_method="lanczos", width=width, height=height,  # noqa
                            crop="center")
    instant = g.add("Flux3ImageNode", "The instant (FLUX 3)", prompt=field(g, plan(0), "still"), seed=seed,
                    bounding_boxes=layout(2), **{"images.image_1": shot(0), "images.image_2": face()}, **flux)
    before = g.add("Flux3ImageNode", "A second earlier (FLUX 3)", prompt=field(g, plan(0), "before"), seed=seed + 1,
                   **{"images.image_1": instant(0), "images.image_2": face()}, **flux)
    # LTX-2.5: three keyframes
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    video_text = g.add("StringConcatenate", string_a=field(g, plan(0), "video"), string_b=QUIET, delimiter="")()
    cond = g.add("LTXVConditioning", frame_rate=float(fps),
                 positive=g.add("CLIPTextEncode", text=video_text, clip=clip())(),
                 negative=g.add("CLIPTextEncode", text=VIDEO_NEGATIVE, clip=clip())())
    lat = g.add("LTXVImgToVideoInplace", vae=vae(), image=fit(before(0))(), strength=1.0, bypass=False,
                latent=g.add("EmptyLTXVLatentVideo", width=width, height=height, length=frames, batch_size=1)())
    pos, neg, latent = cond(0), cond(1), lat()
    for image, idx in ((fit(instant(0))(), -1),):
        guide = g.add("LTXVAddGuide", positive=pos, negative=neg, vae=vae(), latent=latent, image=image,
                      frame_idx=idx, strength=1.0)
        pos, neg, latent = guide(0), guide(1), guide(2)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=avae())
    av = g.add("LTXVConcatAVLatent", video_latent=latent, audio_latent=audio())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=g.add("UNETLoader", unet_name=LTX, weight_dtype="default")(),
                             positive=pos, negative=neg, cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    video = g.add("LTXVCropGuides", positive=pos, negative=neg, latent=sep(0))(2)
    moment = g.add("CreateVideo", images=g.add("VAEDecode", samples=video, vae=vae())(),
                   audio=g.add("LTXVAudioVAEDecode", samples=sep(1), audio_vae=avae())(), fps=float(fps))
    g.add("SaveVideo", "Momento", video=moment(), filename_prefix=prefix, format="mp4", codec="h264")
    g.add("SaveImage", "Keyframes", images=g.add("ImageBatch", image1=fit(before(0))(), image2=fit(instant(0))())(),
          filename_prefix=prefix + "_keys")
    critic = g.add("GeminiNode", "Critic (Gemini)", prompt="Review this take.", model="gemini-3-1-pro", seed=seed,
                   images=both(0), video=moment(), system_prompt=CRITIC)
    g.add("PreviewAny", "Review", source=critic(0))
    return g


def upload(path):
    with open(path, "rb") as f:
        data = f.read()
    r = retry(requests.post, f"{CLOUD}/api/upload/image", headers={"X-API-Key": key()}, timeout=120,
              files={"image": (Path(path).name, data)}, data={"type": "input", "overwrite": "true"})
    r.raise_for_status()
    return r.json()["name"]


def run(graph):
    """Submit, follow the websocket until done. -> (video file info, texts)"""
    cid = str(uuid.uuid4())
    ws = websocket.create_connection(f"wss://cloud.comfy.org/ws?clientId={cid}&token={key()}", timeout=1800)
    r = requests.post(f"{CLOUD}/api/prompt", headers={"X-API-Key": key()},
                      json={"prompt": graph, "client_id": cid, "extra_data": {"api_key_comfy_org": key()}})
    res = r.json()
    if "prompt_id" not in res:
        raise RuntimeError(json.dumps(res)[:3000])
    pid, video, texts = res["prompt_id"], None, []
    while True:
        msg = ws.recv()
        if not isinstance(msg, str):
            continue
        m = json.loads(msg)
        data = m.get("data", {})
        if data.get("prompt_id") not in (None, pid):
            continue
        if m["type"] == "executed":
            out = data.get("output") or {}
            for f in out.get("video", []) + out.get("videos", []) + out.get("images", []):
                if f["filename"].endswith(".mp4"):
                    video = f
            texts += [t for t in out.get("text", []) if isinstance(t, str)]
        elif m["type"] == "execution_error":
            raise RuntimeError(f"{data.get('node_type')}: {data.get('exception_message')}"[:2000])
        elif m["type"] == "execution_success":
            ws.close()
            return video, texts


def download(f, dest):
    r = retry(requests.get, f"{CLOUD}/api/view", params={k: f[k] for k in ("filename", "subfolder", "type")},
              headers={"X-API-Key": key()}, allow_redirects=False, timeout=120)
    url = r.headers.get("Location")
    Path(dest).write_bytes(retry(requests.get, url, timeout=300).content if url else r.content)


def reveal(moment, post, out, hold=1.8, flash=0.25, click=True):
    """Momento Reveal with ffmpeg: the moment, a white flash with a shutter click, then the real post. click=False when
    the moment already ends on its own shutter."""
    w, h = (int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", moment],
                                            capture_output=True, text=True).stdout.split(",")[:2])
    w += w % 2
    h += h % 2
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   moment], capture_output=True, text=True).stdout.strip() or 4)
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", moment], capture_output=True, text=True).stdout.strip()
    pad = f"atrim=0:{length + hold + flash:.3f}"
    bed = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,apad,{pad},afade=out:st={length:.3f}:d={hold + flash}"
           if has_audio else f"anullsrc=r=48000:cl=stereo,{pad}")  # the scene's sound fades under the post
    sound = "aevalsrc='0.8*(random(0)*2-1)*exp(-t*60)|0.8*(random(1)*2-1)*exp(-t*60)':d=0.15:s=48000"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", moment, "-loop", "1", "-t", str(hold + flash), "-i", post,
         "-filter_complex",
         f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1,"
         f"fade=in:st=0:d={flash}:color=white[p];[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
         f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1[m];[m][p]concat=n=2:v=1:a=0[v];"
         + (f"{bed}[q];{sound},adelay={int(length * 1000)}:all=1[c];[q][c]amix=inputs=2:normalize=0[a]" if click
            else f"{bed}[a]"),
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out], check=True)


def take(pic, img, post, name, n, out):
    """One take, revealed and reviewed. -> (score, path, verdict)"""
    t = time.time()
    video, texts = run(cloud_graph(pic, img, seed=random.randrange(2**31), prefix=f"momento/{name}"))
    raw, path = Path(out) / f"{name}_cloud_take{n}_raw.mp4", Path(out) / f"{name}_cloud_take{n}.mp4"
    download(video, raw)
    reveal(str(raw), post, str(path))
    verdict = {}
    for text in texts:
        try:
            verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError:
            pass
    print(f"[{name}] take {n} ({time.time() - t:.0f}s) {path} review {verdict}", flush=True)
    return (verdict.get("score") or 0) + (10 if verdict.get("pass") else 0), path, verdict


def shoot(profile, post, out="out"):
    """TAKES takes side by side on Cloud; the critic's best one wins."""
    name = Path(post).stem
    pic, img = upload(profile), upload(post)
    with ThreadPoolExecutor(TAKES) as pool:
        takes = list(pool.map(lambda n: take(pic, img, post, name, n, out), range(1, TAKES + 1)))
    score, path, verdict = max(takes, key=lambda t: t[0])
    best = Path(out) / f"{name}_cloud.mp4"
    best.write_bytes(path.read_bytes())
    print(f"[{name}] BEST {best} <- {path.name} {verdict}", flush=True)


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]
    with ThreadPoolExecutor(len(posts)) as pool:  # Cloud runs them side by side
        list(pool.map(lambda p: shoot(profile, p), posts))
