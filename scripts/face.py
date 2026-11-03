"""The face shot: a close-up of the photographer as they see the moment and raise the camera.

    COMFY_URL=... CAMERA=smartphone python scripts/face.py profile.jpg post.jpg out.mp4 [seed]

The person comes from their real profile picture (restored by FLUX 3 Klein with the Restore LoRA), not a generated
model sheet; Klein stages the close-up in the post's light with the place behind them out of focus (skin-texture
LoRA); MiniMax H3 animates it from that first frame (turbo v1.3, Realism People).
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
from local import Graph, fetch, klein, run  # noqa: E402

W, H, F, FPS = 384, 672, 56, 24
CAMERA = os.environ.get("CAMERA", "smartphone")
DEVICE = "phone" if CAMERA != "dslr" else "camera"
H3U = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
TURBO = "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors"
REAL = "h3_realism_people.safetensors"
CLIP, VAE, AVAE = ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors",
                   "minimax_h3_audio_vae_fp32.safetensors")


def graph(profile, post, scene, seed, prefix):
    g = Graph()
    who = klein(g, "Restore and enlarge this photo of a person: the same face, features, skin tone, hair and age, as "
                   "a sharp natural photograph with real skin texture. Change nothing about who they are.",
                [g.add("LoadImage", image=profile)(0)], seed, width=768, height=768,
                loras=[("klein_restore.safetensors", 1.0)])
    still = klein(g, "A real photograph, a close-up of the person in ref_image_0: the same face, skin tone, hair and "
                     "age. Head and shoulders, three-quarter front view, they look past the camera toward screen left, "
                     f"at something happening, holding a {DEVICE} low at chest height, about to raise it. They stand "
                     "in the place of ref_image_1 with its exact light and colour on their face; behind them, far out "
                     "of focus, that place as soft bokeh. 85mm lens, shallow depth of field, natural skin texture, "
                     "candid, unposed. " + scene, [who(), g.add("LoadImage", image=post)(0)], seed + 1, width=W,
                  height=H, loras=[("klein_skin_texture.safetensors", 0.7)])
    g.add("SaveImage", images=still(), filename_prefix=prefix + "_still")
    model = g.add("LoraLoaderModelOnly", lora_name=TURBO, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3U, weight_dtype="default")())
    model = g.add("LoraLoaderModelOnly", lora_name=REAL, strength_model=1.0, model=model())
    vae, avae = g.add("VAELoader", vae_name=VAE), g.add("VAELoader", vae_name=AVAE)
    text = (f"integrated_multimodal_description: r34l1sm A handheld close-up, nearly still, shallow focus. The person "
            f"watches something off to the left; their eyes catch the moment, their expression sharpens, and they "
            f"lift the {DEVICE} up to their eye line to take the photo. Natural, small, real movements; they blink "
            f"once. {scene}\noverall_soundscape: the place's ambient sound, then the click of a camera shutter"
            f"\nnon_diegetic_music: N/A")
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=F, first_frame=still(), vae=vae(),
                 clip=g.add("CLIPLoader", clip_name=CLIP, type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=6, denoise=1.0)(),
                latent_image=cond(1))
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def main(profile, post, out, seed=None):
    seed = int(seed) if seed else random.randrange(2**31)
    files = run(graph(comfy.upload(profile, f"momento_{Path(profile).name}"),
                      comfy.upload(post, f"momento_{Path(post).name}"), os.environ.get("SCENE", ""), seed,
                      f"momento/face_{Path(post).stem}_{seed}"))
    fetch(files["final"][0], out)
    fetch(files["still"][0], str(out).replace(".mp4", "_still.png"))
    print("seed", seed, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
