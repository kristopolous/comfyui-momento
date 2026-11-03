"""Animate a posed still with a pose sequence: the arms rise until the phone is at eye level.

    COMFY_URL=... python scripts/posedvideo.py still.png out.mp4 [seed]

MiniMax H3 (turbo v1.3, Realism People) from the still as the first frame, held to a DWPose sequence through the
Fun ControlNet-Union: the motion is the skeleton's, the model renders it.
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
import posed  # noqa: E402
from local import Graph, fetch, run  # noqa: E402

W, H, F, FPS = 384, 672, 56, 24
RAISED = {"r_elbow": (250, 760), "l_elbow": (560, 748), "r_wrist": (340, 560), "l_wrist": (462, 556)}
RISE = (0.3, 0.75)


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def pose_video(out):
    frames = []
    base = dict(posed.POSE)
    for i in range(F):
        k = ease((i / (F - 1) - RISE[0]) / (RISE[1] - RISE[0]))
        posed.POSE = {n: (tuple(a + (b - a) * k for a, b in zip(p, RAISED[n])) if n in RAISED else p)
                      for n, p in base.items()}
        frames.append(np.asarray(posed.pose_image().resize((W, H), Image.LANCZOS)))
    posed.POSE = base
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", out],
                           stdin=subprocess.PIPE)
    enc.stdin.write(np.stack(frames).tobytes())
    enc.stdin.close()
    enc.wait()


def graph(still, pose, seed, prefix):
    g = Graph()
    first = g.add("ImageScale", image=g.add("LoadImage", image=still)(0), upscale_method="lanczos", width=W, height=H,
                  crop="center")
    pv = g.add("GetVideoComponents", video=g.add("LoadVideo", file=pose)())
    vae = g.add("VAELoader", vae_name="minimax_h3_video_vae_int8_convrot.safetensors")
    avae = g.add("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    model = g.add("LoraLoaderModelOnly", lora_name="minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors",
                  strength_model=1.0, model=g.add("UNETLoader", unet_name="minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                                                  weight_dtype="default")())
    model = g.add("LoraLoaderModelOnly", lora_name="h3_realism_people.safetensors", strength_model=1.0, model=model())
    model = g.add("MiniMaxH3FunControlNetApply", model=model(), vae=vae(), strength=0.8, start_percent=0.0,
                  end_percent=0.8, control_video=pv(0),
                  model_patch=g.add("ModelPatchLoader", name="minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors")())
    text = ("integrated_multimodal_description: r34l1sm A handheld medium close-up on a city street at sunset. The "
            "man looks off to the left; something catches his eye, and he raises his phone in both hands up to eye "
            "level to take a photo of it. Natural, unhurried movement.\noverall_soundscape: city street ambience, "
            "then the click of a phone camera shutter\nnon_diegetic_music: N/A")
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=F, first_frame=first(), vae=vae(),
                 clip=g.add("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors", type="minimax",
                            device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=6, denoise=1.0)(),
                latent_image=cond(1))
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def main(still, out, seed=None):
    seed = int(seed) if seed else random.randrange(2**31)
    pv = Path(tempfile.mkdtemp()) / "pose.mp4"
    pose_video(str(pv))
    files = run(graph(comfy.upload(still, f"momento_{Path(still).name}"), comfy.upload(str(pv), f"momento_posevid_{seed}.mp4"),
                      seed, f"momento/posedvid_{seed}"))
    fetch(files["final"][0], out)
    subprocess.run(["cp", str(pv), str(out).replace(".mp4", "_pose.mp4")])
    print("seed", seed, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
