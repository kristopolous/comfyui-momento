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
H3 = dict(unet=os.environ.get("H3_UNET", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
          lora="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
          vae="minimax_h3_video_vae_int8_convrot.safetensors", audio_vae="minimax_h3_audio_vae_fp32.safetensors",
          clip="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
GAP = "#FF00FF"  # what RenderMesh draws where the camera sees past the photo; the fill outpaints it
END_ONLY = os.environ.get("END_ONLY", "1") == "1"  # H3 gets only the end frame: a separate start frame morphs
CONTROL = os.environ.get("CONTROL") == "1"  # geometry as conditioning: depth-driven camera, pose-inpainted photographer
# depth sets the camera path and the layout early, then lets go so the subject can move (0.7/0.6 froze the bus)
CN_DEPTH, CN_DEPTH_END = float(os.environ.get("CN_DEPTH", "0.6")), float(os.environ.get("CN_DEPTH_END", "0.3"))
# H3 steps and whether the turbo LoRA is on, per pass: the fast default (8 / 4 turbo steps), or e.g. P2_STEPS=30
# P2_TURBO=0 for the full model (Fun ControlNet-Union's own results use 40 steps)
EXPLAIN = os.environ.get("EXPLAIN") == "1"  # also save the pose video and SAM 3D Body's view, for the demo
P1 = (int(os.environ.get("P1_STEPS", "8")), os.environ.get("P1_TURBO", "1") == "1")
P2 = (int(os.environ.get("P2_STEPS", "4")), os.environ.get("P2_TURBO", "1") == "1")
CN_PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
REF2VA = dict(unet="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
              lora="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
FLEX = os.environ.get("FLEX") == "1"  # Pika H3 Flex: directed camera moves, Comfy's 6-step recipe
if FLEX:
    H3["lora"] = "pika_h3_flex_fl2va_v1_bf16.safetensors"

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


def aim(eye=None):
    """Which way the photographer faces in the bystander's frame, from 3D: they stand at the post's camera and shoot
    along its axis, so their camera points at where that axis lands in this frame."""
    eye = eye or geo.START
    me = geo.project((0.0, -0.3, 0.0), eye)[0] / geo.W
    there = geo.project((0.0, -0.1, -8.0), eye)[0] / geo.W
    side = "to their left" if there < me - 0.05 else "to their right" if there > me + 0.05 else "straight ahead"
    return (f"seen from directly behind, their back to us, facing away from us into the scene, turned slightly {side} "
            f"so their camera points at the subject near the {'left' if there < 0.4 else 'right' if there > 0.6 else 'center'}"
            " of the frame, never toward us or out of the frame")


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
        view = g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background=GAP,
                     camera_info=cam())
        if move:  # the opening view of the camera move, from the same 3D place
            cam0 = g.add("CreateCameraInfo", mode="look_at",
                         **{f"mode.position_{a}": v for a, v in zip("xyz", geo.MOVES[move])},
                         target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0,
                         fov=geo.LENS, zoom=1.0, camera_type="perspective")
            view0 = g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background=GAP,
                          camera_info=cam0())
        instant = klein(g, "ref_image_0 is a bystander's view of a real place. The solid magenta areas are where the "
                           "camera sees past what was photographed: they are missing, not part of the place. Extend the "
                           "place outward into every magenta area so it fills the whole frame, with more of the same "
                           "place, same light, same lens: no border, no frame, no magenta left. Keep everything else "
                           "exactly as it is. No people in the foreground. " + plan["scene"],
                        [view(0)], seed)
    if CONTROL and not closeup:
        return graph_control(g, mesh, instant, sheet, plan, seed, prefix, move)
    fill = ("ref_image_0 is a bystander's view of a real place. The solid magenta areas are where the camera sees past "
            "what was photographed: they are missing, not part of the place. Extend the place outward into every magenta "
            "area so it fills the whole frame, with more of the same place, same light, same lens: no border, no frame, "
            "no magenta left. Keep everything else exactly as it is. No people in the foreground. " + plan["scene"])

    def place(img, eye, s_):
        return klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in "
                        f"the foreground, {aim(eye)}, "
                        f"{CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the sheet. "
                        "Exactly one photographer, no other copies of them. Change nothing else in ref_image_0. "
                     + plan["photographer"] + rows(geo.layout(eye, CAMERA)), [img, sheet()], s_)

    opening = (place(klein(g, fill, [view0(0)], seed + 2)(), geo.MOVES[move], seed + 3)
               if move and not closeup and not END_ONLY else None)
    placed = klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the "
                      f"foreground, {aim()}, "
                      f"{CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the "
                      "sheet. Exactly one photographer, no other copies of them. Change nothing else in ref_image_0. "
                   + plan["photographer"] + rows(geo.layout(camera=CAMERA)),
                   [instant(), sheet()], seed + 1)
    g.add("SaveImage", images=g.add("ImageBatch", image1=instant(), image2=placed())(), filename_prefix=prefix + "_keys")
    if opening:
        g.add("SaveImage", images=opening(), filename_prefix=prefix + "_open")
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
    if move and END_ONLY:  # one image, the end; the move and the earlier action are text, so nothing pops in
        text = text.replace("Handheld bystander camera, nearly still.", f"The camera makes {geo.MOVE_WORDS[move]}, "
                            f"through this real place, coming to rest exactly on the framing of <Picture 1>.")
        text = text.replace("The subject is in motion and ends", f"At the start, {plan.get('before', '')} The subject is in "
                            "continuous motion the whole time and ends")
    elif opening:
        text = text.replace("Handheld bystander camera, nearly still.", f"The camera makes {geo.MOVE_WORDS[move]}, "
                            f"moving through the real space of <Picture 1> to <Picture 2>.")
    ends = ({"first_frame": opening(), "last_frame": placed()} if opening and not END_ONLY
            else {"last_frame": placed()})
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=H3_FRAMES, **ends,
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")(), vae=vae())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="lcm" if FLEX else "res_multistep")(),
                sigmas=g.add("ManualSigmas", sigmas="1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0")() if FLEX
                else g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    video = g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                  audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())
    g.add("SaveVideo", video=video(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def h3_sample(g, model, cond, seed, steps):
    return g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                 guider=g.add("BasicGuider", model=model, conditioning=cond(0))(),
                 sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                 sigmas=g.add("BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.0)(),
                 latent_image=cond(1))


def graph_control(g, mesh, instant, sheet, plan, seed, prefix, move):
    """Geometry as conditioning. Pass 1 films the place: the camera follows a depth video rendered from the post's
    3D mesh along the computed path, ending on the outpainted instant (nobody in the foreground). Pass 2 inpaints the
    photographer into that clip: MiniMax H3 Ref2VA (identity from the character sheet) under a DWPose skeleton and a
    mask, both projected from 3D (Momento Photographer), so where they stand, which way they face and where their hands and
    camera are come from the geometry, not from a prompt."""
    g.add("SaveImage", images=instant(), filename_prefix=prefix + "_keys")
    clip = g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")
    vae = g.add("VAELoader", vae_name=H3["vae"])
    avae = g.add("VAELoader", vae_name=H3["audio_vae"])
    patch = g.add("ModelPatchLoader", name=CN_PATCH)

    # the camera path through the post's 3D place (and the ground under the photographer), as depth
    mv = move or "still"
    depth = g.add("MomentoCameraPath", mesh=mesh(), move=mv, frames=H3_FRAMES, width=W, height=H, ground=True)(0)
    g.add("SaveAnimatedWEBP", images=depth, fps=float(FPS), lossless=False, quality=60, method="default",
          filename_prefix=prefix + "_depth")

    # pass 1: the place and the subject's action, the camera locked to the depth video
    unet = g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")
    m1 = (g.add("LoraLoaderModelOnly", lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                strength_model=1.0, model=unet()) if P1[1] else unet)
    m1 = g.add("MiniMaxH3FunControlNetApply", model=m1(), model_patch=patch(), vae=vae(), strength=CN_DEPTH,
               start_percent=0.0, end_percent=CN_DEPTH_END, control_video=depth)
    sound = plan["soundscape"].replace("phone camera shutter", CAMERA == "dslr" and "DSLR shutter" or
                                       "phone camera shutter")
    scene = (f"integrated_multimodal_description: The camera makes {geo.MOVE_WORDS[move] if move else 'no move'}, "
             f"through this real place, coming to rest exactly on the framing of <Picture 1>. At the start, "
             f"{plan.get('before', '')} The subject is in continuous motion the whole time and ends exactly in the "
             f"moment of <Picture 1>: {plan['action']} {plan['scene']}"
             f"\noverall_soundscape: {sound}\nnon_diegetic_music: N/A")
    c1 = g.add("MiniMaxH3ImageToVideo", prompt=scene, width=W, height=H, length=H3_FRAMES, last_frame=instant(),
               clip=clip(), vae=vae())
    s1 = h3_sample(g, m1(), c1, seed, P1[0])
    place = g.add("VAEDecode", samples=s1(0), vae=vae())
    audio = g.add("VAEDecodeAudio", samples=s1(0), vae=avae())
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=place(), audio=audio())(),
          filename_prefix=prefix + "_place", format="mp4", codec="h264")

    # pass 2: the photographer, inpainted under the projected skeleton and mask
    who_3d = g.add("MomentoPhotographer", move=mv, camera=CAMERA, frames=H3_FRAMES, width=W, height=H)
    pose, mask = (lambda: who_3d(0)), (lambda: who_3d(1))
    unet2 = g.add("UNETLoader", unet_name=REF2VA["unet"], weight_dtype="default")
    m2 = (g.add("LoraLoaderModelOnly", lora_name=REF2VA["lora"], strength_model=1.0, model=unet2()) if P2[1]
          else unet2)
    m2 = g.add("MiniMaxH3FunControlNetApply", model=m2(), model_patch=patch(), vae=vae(), strength=1.0,
               start_percent=0.0, end_percent=1.0, control_video=pose(), mask=mask(), source_video=place())
    who = (f"integrated_multimodal_description: The same shot of the same place, unchanged. In the foreground stands "
           f"the person from <Picture 1>, {plan['photographer']}, seen from behind, {CAM['act']}. {plan['scene']}"
           f"\noverall_soundscape: {sound}\nnon_diegetic_music: N/A")
    c2 = g.add("MiniMaxH3ReferenceToVideo", prompt=who, width=W, height=H, length=H3_FRAMES, ref_image_size="match",
               clip=clip(), vae=vae(), audio_vae=avae(), **{"ref_images.ref_image_0": sheet()})
    s2 = h3_sample(g, m2(), c2, seed + 7, P2[0])
    # keep pass 1 exactly outside the photographer: the model redraws the whole frame, the mask decides what's kept
    who_px = g.add("VAEDecode", samples=s2(0), vae=vae())
    final = g.add("ImageCompositeMasked", destination=place(), source=who_px(), x=0, y=0, resize_source=False,
                  mask=mask())
    video = g.add("CreateVideo", fps=float(FPS), images=final(), audio=audio())
    g.add("SaveVideo", video=video(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    # measured, not judged: SAM 3D Body on frames after the raise, inside the box the 3D puts the photographer in
    picks = g.add("MomentoGateFrames", images=final(), move=mv, camera=CAMERA)
    body = g.add("SAM3DBody_Predict", image=picks(0), bboxes=picks(1), run_hand_refinement=False, fov=geo.LENS,
                 batch_size=4, sam3d_body_model=g.add("SAM3DBody_Loader",
                                                      model_file="sam_3d_body_dinov3_bf16.safetensors")())
    g.add("MomentoGate", pose_data=body(), move=mv, camera=CAMERA, frames=H3_FRAMES, width=W, height=H)
    if EXPLAIN:  # what the demo shows: the projected skeleton, and the body SAM 3D Body measured, over the take
        g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=pose())(), filename_prefix=prefix + "_pose",
              format="mp4", codec="h264")
        g.add("SaveImage", filename_prefix=prefix + "_body",
              images=g.add("SAM3DBody_Render", pose_data=body(), width=W, height=H, background=picks(0),
                           **{"render_style": "mesh", "render_style.shader": "default"})())
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
        for text in out.get("text", []):
            if text.startswith("GATE "):
                files["gate"] = json.loads(text[5:])
    return files


def fetch(f, dest):
    q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
    Path(dest).write_bytes(comfy._req(f"/view?{q}"))


def take(profile_up, post_up, profile, post, plan, closeup, n, out):
    name = Path(post).stem
    tag = f"{name}_{os.environ.get('TAG', 'local')}{'_dslr' if CAMERA == 'dslr' else ''}{n}"
    t = time.time()
    # a camera move by default (it looks much better); ALLOW_STILL=1 lets a take keep the camera still
    moves = [None, *geo.MOVES] if os.environ.get("ALLOW_STILL") else list(geo.MOVES)
    move = None if closeup else os.environ.get("MOVE") or random.choice(moves)  # variety, or MOVE= to pin it
    print(f"[{tag}] camera: {move or 'still'}", flush=True)
    seed = int(os.environ.get("SEED") or random.randrange(2**31))  # SEED= to repeat a take
    files = run(graph(profile_up, post_up, plan, seed, f"momento/{tag}", closeup, move))
    od = Path(out) / "steps"
    od.mkdir(parents=True, exist_ok=True)
    fetch(files["sheet"][0], od / f"{tag}_sheet.png")
    for i, f in enumerate(files["keys"]):
        fetch(f, od / f"{tag}_key{i}.png")
    for f in files.get("open", []):
        fetch(f, od / f"{tag}_open.png")
    raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
    fetch(files["final"][0], raw)
    reveal(str(raw), post, str(final))
    frames = Path(tempfile.mkdtemp())
    for i, s in enumerate((0.5, 2.0, 3.5, 4.3)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s), "-i", str(raw), "-frames:v", "1",
                        "-vf", "scale=-2:384", str(frames / f"{i}.jpg")], check=True)
    verdict = ask(CRITIC, "Review this take.", sorted(map(str, frames.glob("*.jpg"))) + [post])
    if "gate" in files:  # the Momento Gate node measured the photographer against the 3D (SAM 3D Body)
        verdict["geometry"] = files["gate"]["frames"]
        if not files["gate"]["ok"]:
            verdict.update({"pass": False, "score": 0, "problem": f"geometry gate failed: {files['gate']['frames']}"})
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
