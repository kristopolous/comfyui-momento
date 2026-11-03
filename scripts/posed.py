"""Pose first, then the person: the face shot's keyframe from a chosen pose, not from the profile picture's pose.

    COMFY_URL=... python scripts/posed.py profile.jpg post.jpg out.png [seed]

Qwen-Image 2.1 (Turbo, edit) draws the frame with two references, the profile picture (who) and the post (where,
and its light), while the Fun ControlNet-Union holds it to a DWPose skeleton drawn here: a three-quarter close-up,
looking off to the left, both hands fully in frame holding the phone at chest height.
"""
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
from local import Graph, fetch, run  # noqa: E402

W, H = 768, 1344
SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
# OpenPose / DWPose body: index -> name, limbs and colours as the preprocessors draw them
NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
         "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
# a three-quarter close-up turned toward screen left; the person's right is screen left
POSE = {"nose": (352, 476), "r_eye": (322, 440), "l_eye": (398, 436), "l_ear": (470, 452),
        "neck": (400, 610), "r_shoulder": (226, 668), "l_shoulder": (578, 656),
        "r_elbow": (196, 960), "l_elbow": (612, 948), "r_wrist": (334, 1036), "l_wrist": (470, 1032),
        "r_hip": (290, 1330), "l_hip": (520, 1330)}


def pose_image():
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    stick = 8
    for (a, b), c in zip(LIMBS, COLORS):
        pa, pb = POSE.get(NAMES[a]), POSE.get(NAMES[b])
        if not pa or not pb:
            continue
        (x0, y0), (x1, y1) = pa, pb
        length, ang = math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts = [(cx + math.cos(r) * length / 2 * math.cos(ang) - math.sin(r) * stick * math.sin(ang),
                cy + math.cos(r) * length / 2 * math.sin(ang) + math.sin(r) * stick * math.cos(ang))
               for r in (math.radians(s) for s in range(0, 360, 15))]
        d.polygon(pts, fill=tuple(int(v * 0.6) for v in c))
    for i, n in enumerate(NAMES):
        if n in POSE:
            x, y = POSE[n]
            d.ellipse((x - stick, y - stick, x + stick, y + stick), fill=COLORS[i])
    return im


def graph(profile, post, pose, scene, seed, prefix):
    g = Graph()
    vae = g.add("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")
    model = g.add("QwenImage21Cache", device="auto", dtype="default",
                  model=g.add("UNETLoader", unet_name="qwen_image_2.1_turbo_int8_convrot.safetensors",
                              weight_dtype="default")())
    model = g.add("QwenImageDiffsynthControlnet", model=model(), vae=vae(), strength=0.9,
                  image=g.add("LoadImage", image=pose)(0),
                  model_patch=g.add("ModelPatchLoader", name="qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors")())
    clip = g.add("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")
    text = ("A real candid photograph of the person in Picture 1: exactly their face, hair, skin tone and age. Head and "
            "upper body, a three-quarter view, they look off to the left at something happening, holding a "
            "smartphone at chest height in both hands, both hands fully in frame. They stand in the place of "
            "Picture 2, lit by its light; behind them that place is far out of focus. 85mm lens, natural skin "
            "texture, real adult hands. " + scene)
    enc = g.add("TextEncodeQwenImage21", clip=clip(), vae=vae(), prompt=text, resolution=1024,
                negative_prompt="cartoon, 3d render, cgi, plastic skin, deformed hands, extra fingers, cropped hands",
                **{"images.image_1": g.add("LoadImage", image=profile)(0),
                   "images.image_2": g.add("LoadImage", image=post)(0)})
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=model(), positive=enc(0), negative=enc(1), cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler")(), sigmas=g.add("ManualSigmas", sigmas=SIGMAS)(),
                latent_image=g.add("EmptyLatentImage", width=W, height=H, batch_size=1)())
    g.add("SaveImage", images=g.add("VAEDecode", samples=out(0), vae=vae())(), filename_prefix=prefix + "_still")
    return g


def main(profile, post, out, seed=None):
    seed = int(seed) if seed else random.randrange(2**31)
    tmp = Path(out).with_suffix(".pose.png")
    pose_image().save(tmp)
    files = run(graph(comfy.upload(profile, f"momento_{Path(profile).name}"),
                      comfy.upload(post, f"momento_{Path(post).name}"), comfy.upload(str(tmp), "momento_pose_face.png"),
                      os.environ.get("SCENE", ""), seed, f"momento/posed_{Path(post).stem}_{seed}"))
    fetch(files["still"][0], out)
    print("seed", seed, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
