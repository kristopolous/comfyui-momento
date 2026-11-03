"""Extra angles for a moment, cut together with the from-behind take:

  profile     side-profile medium close-up: the photographer watching the subject, camera lowered, about to raise it
  viewfinder  over the shoulder onto the phone screen / camera back: the live view framing the shot

    COMFY_URL=... CAMERA=smartphone python scripts/cuts.py SHEET.png POST.jpg SCENE.txt OUTDIR [seed]

Each angle: a FLUX 3 Klein keyframe from the character sheet and the post, then MiniMax H3 (Flex + Realism People)
from that first frame. The edit (profile -> viewfinder -> from behind -> flash -> post) is done by edit().
"""
import os
import random
import subprocess
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--hires", *sys.argv[1:]]
from local import CAM, CAMERA, FPS, H, H3, W, Graph, comfy, fetch, klein, run  # noqa: E402

FLEX_LORA = "pika_h3_flex_fl2va_v1_bf16.safetensors"
REAL_LORA = "h3_realism_people.safetensors"
FLEX_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
FRAMES = 39  # 17n+5: about 1.6 s per angle


def angles(scene, who):
    device = "phone" if CAMERA != "dslr" else "camera"
    return {
        "profile": dict(
            still=(f"A real photograph, a side-profile medium close-up of the person from ref_image_0's character "
                   f"sheet ({who}), standing on the street, facing screen left and looking intently toward the left "
                   f"edge of the frame, {CAM.get('ready', 'holding the ' + ('phone' if CAMERA != 'dslr' else 'camera') + ' lowered at chest height, about to raise it')}. At the far left edge, soft and out of focus, the subject they "
                   f"are watching from ref_image_1. Behind them, softly out of focus, more of the same street in the "
                   f"same light. 50mm lens, shallow depth of field, natural daylight, real skin texture. {scene}"),
            video=(f"r34l1sm A side-profile medium close-up, the camera nearly still, handheld. The photographer "
                   f"watches the scene to the left, then starts to raise the {device} toward it. Ambient street "
                   f"sound. No speech.")),
        "viewfinder": dict(
            still=(f"A real photograph, an over-the-shoulder close-up from just behind and above the shoulder of the "
                   f"person from ref_image_0's character sheet ({who}): their shoulder and the back of their head "
                   f"soft in the foreground, both hands holding the {device} up at eye level. Its screen shows the "
                   f"camera's live view, framing exactly ref_image_1. Beyond it, out of focus, the real scene of "
                   f"ref_image_1. Natural daylight, real photograph. {scene}"),
            video=(f"r34l1sm An over-the-shoulder close-up of hands holding a {device} up, its screen showing the live "
                   f"view. The framing settles with small handheld movements and the screen stays lit. Ambient "
                   f"street sound. No speech.")),
    }


def graph(sheet, post, a, seed, prefix):
    g = Graph()
    refs = [g.add("LoadImage", image=sheet)(0), g.add("LoadImage", image=post)(0)]
    first = klein(g, a["still"], refs, seed, steps=20)
    g.add("SaveImage", images=first(), filename_prefix=prefix + "_still")
    model = g.add("LoraLoaderModelOnly", lora_name=FLEX_LORA, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    model = g.add("LoraLoaderModelOnly", lora_name=REAL_LORA, strength_model=1.0, model=model())
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    cond = g.add("MiniMaxH3ImageToVideo", prompt="integrated_multimodal_description: " + a["video"]
                 + "\noverall_soundscape: ambient street sound\nnon_diegetic_music: N/A", width=W, height=H,
                 length=FRAMES, first_frame=first(), vae=vae(),
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="lcm")(),
                sigmas=g.add("ManualSigmas", sigmas=FLEX_SIGMAS)(), latent_image=cond(1))
    video = g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                  audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())
    g.add("SaveVideo", video=video(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def main(sheet, post, scene, out, seed=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(seed) if seed else random.randrange(2**31)
    su, pu = comfy.upload(sheet, f"momento_cuts_{Path(sheet).name}"), comfy.upload(post, f"momento_cuts_{Path(post).name}")
    who = os.environ.get("WHO", "the person in the character sheet")
    for name, a in angles(Path(scene).read_text().strip(), who).items():
        files = run(graph(su, pu, a, seed, f"momento/cuts_{name}_{seed}"))
        fetch(files["still"][0], out / f"{name}_{seed}.png")
        fetch(files["final"][0], out / f"{name}_{seed}.mp4")
        print(name, seed, "done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
