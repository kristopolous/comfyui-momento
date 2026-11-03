"""Momento in one continuous shot: the person and the place are rendered together, so nothing slides or passes
through anyone.

    COMFY_URL=... PLANNER_URL=... PLANNER_MODEL=... CAMERA=smartphone|dslr python scripts/single.py profile.jpg post.jpg

  1. FLUX 3 Klein makes the end frame: the place re-photographed from the bystander's position (the post's 3D
     render as the layout, the post as the look), with the photographer from their real (restored) profile picture placed where the
     3D puts them. Real objects stay whole: the render guides the layout, it isn't kept pixel for pixel.
  2. Sam 3D Body measures that still (Momento Gate): facing the shot, camera at the eye, standing where projected.
     A still that fails is re-rolled before any video is made.
  3. MiniMax H3 (Pika H3 Flex) films the seconds before, ending on the still: one shot, one camera move.
  4. The take is gated again, then cut to the real post.
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.argv = [sys.argv[0], "--hires", *sys.argv[1:]]  # 448x768, the published reel's size
import local  # noqa: E402
from local import (CAM, CAMERA, CRITIC, FPS, GAP, H, H3, H3_FRAMES, PLAN, SHEET, W, Graph, ask, comfy, fetch, geo,  # noqa: E402
                   klein, rows, run)
from cloud import reveal  # noqa: E402

FLEX_LORA = "pika_h3_flex_fl2va_v1_bf16.safetensors"
FLEX_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
MOVE = "dolly"
# the official turbo distill (lightx2v FL2V v1.3) at TURBO steps, instead of Pika Flex; the early turbo
# checkpoints are documented to bake in plastic skin
TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors"
TURBO = int(os.environ.get("TURBO", 0))
LOOK_LORAS = [x for x in os.environ.get("LOOK_LORAS", "").split(",") if x]  # e.g. h3_realism_people.safetensors:0.8
TRIGGER = os.environ.get("TRIGGER", "")  # a LoRA's trigger word, if it has one
STILL_TRIES = int(os.environ.get("STILL_TRIES", 6))


def still_graph(profile, post, plan, seed, prefix):
    g = Graph()
    who, shot = g.add("LoadImage", image=profile), g.add("LoadImage", image=post)
    # the person comes from their real profile picture, restored to a usable size (Klein Restore LoRA), not from a
    # generated turnaround sheet: a studio model sheet is what made everyone look like a game character
    sheet = klein(g, "Restore and enlarge this photo of a person: the same face, features, skin tone, hair and age, "
                     "as a sharp natural photograph with real skin texture. Change nothing about who they are.",
                  [who(0)], seed, width=768, height=768, steps=16, loras=[("klein_restore.safetensors", 1.0)])
    geom = g.add("DA3Inference", image=shot(0), resolution=504, resize_method="upper_bound_resize", mode="mono",
                 da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                 weight_dtype="default")())
    mesh = g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1, discontinuity_threshold=0.04,
                 confidence_threshold=0.1, use_sky_mask=True, texture=True)
    cam = g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", geo.START)},
                target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0, fov=geo.LENS,
                zoom=1.0, camera_type="perspective")
    layout = g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background=GAP, camera_info=cam())
    still = klein(g, "A real photograph of the place in ref_image_1, taken from exactly the camera position, angle, "
                     "lens and framing of ref_image_0, with everything where ref_image_0 has it. ref_image_0 is only a "
                     "rough 3D reprojection: redraw it as a clean real photograph, every object whole and properly "
                     "attached (poles reach the ground, signs sit on their posts, nothing floats), and fill its magenta "
                     "areas with more of the place. Light, colour and season as in ref_image_1. In the foreground, the "
                     "person in ref_image_2's photo, seen from directly behind, back to us, facing into "
                     f"the scene, {CAM['hold']}, aimed at the subject, taking the photo. Same face, hair and build "
                     "as that photo, in everyday clothes that suit the place. Exactly one photographer. " + plan["scene"] + " " + plan["photographer"]
                  + rows(geo.layout(camera=CAMERA)), [layout(0), shot(0), sheet()], seed + 1, steps=16,
                  loras=[("klein_skin_texture.safetensors", 0.7)])
    g.add("SaveImage", images=sheet(), filename_prefix=prefix + "_sheet")
    g.add("SaveImage", images=still(), filename_prefix=prefix + "_still")
    return g


def gate_still(name):
    g = Graph()
    picks = g.add("MomentoGateFrames", images=g.add("LoadImage", image=name)(0), move="still", camera=CAMERA)
    body = g.add("SAM3DBody_Predict", image=picks(0), bboxes=picks(1), run_hand_refinement=False, fov=geo.LENS,
                 batch_size=4, sam3d_body_model=g.add("SAM3DBody_Loader",
                                                      model_file="sam_3d_body_dinov3_bf16.safetensors")())
    g.add("MomentoGate", pose_data=body(), move="still", camera=CAMERA, frames=5, width=W, height=H)
    return run(g).get("gate")


def video_graph(still, plan, seed, prefix):
    g = Graph()
    end = g.add("LoadImage", image=still)
    distill = TURBO_LORA if TURBO else FLEX_LORA
    model = g.add("LoraLoaderModelOnly", lora_name=distill, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    for lora in LOOK_LORAS:  # realism LoRAs on top of Flex: name[:strength]
        name, _, strength = lora.partition(":")
        model = g.add("LoraLoaderModelOnly", lora_name=name, strength_model=float(strength or 1.0), model=model())
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    sound = plan["soundscape"].replace("phone camera shutter", "DSLR shutter" if CAMERA == "dslr" else
                                       "phone camera shutter")
    text = (f"integrated_multimodal_description: {TRIGGER + ' ' if TRIGGER else ''}One continuous shot. The camera makes {geo.MOVE_WORDS[MOVE]}, "
            f"through this real place, coming to rest exactly on the framing of <Picture 1>. The photographer in "
            f"the foreground, seen from behind, stands still in one spot and {CAM['act']}. At the start, "
            f"{plan.get('before', '')} The subject is in continuous motion the whole time and ends exactly in the "
            f"moment of <Picture 1>: {plan['action']} {plan['scene']}\noverall_soundscape: {sound}"
            f"\nnon_diegetic_music: N/A")
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=H3_FRAMES, last_frame=end(0),
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")(), vae=vae())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep" if TURBO else "lcm")(),
                sigmas=(g.add("BasicScheduler", model=model(), scheduler="simple", steps=TURBO, denoise=1.0)()
                        if TURBO else g.add("ManualSigmas", sigmas=FLEX_SIGMAS)()), latent_image=cond(1))
    video = g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                  audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())
    g.add("SaveVideo", video=video(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def main(profile, post, out="out"):
    name = Path(post).stem
    od = Path(out) / "steps"
    od.mkdir(parents=True, exist_ok=True)
    plan = ask(PLAN, "Left: the profile picture. Right: the photo they posted. Plan the moment.", [profile, post])
    print(f"[{name}] plan: {json.dumps(plan)[:300]}", flush=True)
    pu, su = comfy.upload(profile, f"momento_{Path(profile).name}"), comfy.upload(post, f"momento_{Path(post).name}")
    still = Path(os.environ["STILL"]) if os.environ.get("STILL") else None
    seed = int(os.environ.get("SEED", 0)) or random.randrange(2**31)
    for i in range(0 if still else STILL_TRIES):  # the still has to pass before any video is made
        seed = random.randrange(2**31)
        files = run(still_graph(pu, su, plan, seed, f"momento/{name}_single_{seed}"))
        path = od / f"{name}_still_{seed}.png"
        fetch(files["still"][0], path)
        report = gate_still(comfy.upload(str(path), f"momento_{name}_still_{seed}.png"))
        print(f"[{name}] still {i + 1} seed {seed}: {json.dumps(report)[:400]}", flush=True)
        if report and report["ok"]:
            still = path
            fetch(files["sheet"][0], od / f"{name}_sheet_{seed}.png")
            break
    if still is None:
        print(f"[{name}] no still passed in {STILL_TRIES} tries", flush=True)
        return None
    t = time.time()
    files = run(video_graph(comfy.upload(str(still), f"momento_{name}_end_{seed}.png"), plan, seed, f"momento/{name}_single_{seed}"))
    tag = os.environ.get("TAG", "single")
    raw, final = od / f"{name}_{tag}_raw.mp4", Path(out) / f"{name}_{tag}.mp4"
    fetch(files["final"][0], raw)
    reveal(str(raw), post, str(final))
    frames = Path(tempfile.mkdtemp())
    for i, s in enumerate((0.3, 1.5, 2.7, 3.6, 4.3)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s), "-i", str(raw), "-frames:v", "1",
                        "-vf", "scale=-2:512", str(frames / f"{i}.jpg")], check=True)
    verdict = ask(CRITIC, "Review this take.", sorted(map(str, frames.glob("*.jpg"))) + [post])
    print(f"[{name}] video ({time.time() - t:.0f}s) {final} review {verdict}", flush=True)
    return final if verdict.get("pass") else None


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]  # local.py consumed --hires
    for p in posts:
        main(profile, p)
