"""The moment from the photographer's own eyes: the real photo, with only what's in it moving.

    COMFY_URL=... python scripts/pov.py post.jpg out.mp4 [seed]

  - The frame is the post itself, with room below it where the photographer's hands come up with the phone
    (FLUX 3 Klein fills only that margin; the photo is pasted back pixel for pixel).
  - SAM 3.1 marks what the photo shows as able to move (people, cars, scooters, umbrellas).
  - MiniMax H3 (Fun ControlNet-Union inpaint over the still frame, Realism People) animates only those regions and
    the hands; everything else is composited back from the photo, so nothing is invented in the scene.
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
from local import Graph, fetch, klein, run  # noqa: E402

W, H, F, FPS = 384, 672, 73, 24
TOP, BOTTOM = 96, 192  # the post sits in the middle (384x384); hands come up into the bottom margin
MOVERS = os.environ.get("MOVERS", "person:12, car:8, scooter:8, bus:3, umbrella:6")
H3U = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
TURBO = "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors"
REAL = "h3_realism_people.safetensors"
CN = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
CLIP, VAE, AVAE = ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors",
                   "minimax_h3_audio_vae_fp32.safetensors")


def graph(post, scene, action, seed, prefix):
    g = Graph()
    photo = g.add("ImageScale", image=g.add("LoadImage", image=post)(0), upscale_method="lanczos", width=W,
                  height=W, crop="center")
    pad = g.add("ImagePadForOutpaint", image=photo(), left=0, top=TOP, right=0, bottom=BOTTOM, feathering=0)
    filled = klein(g, "ref_image_0 is a real photo with empty grey margins above and below it. Fill the top margin "
                      "with more of the same place and sky. In the bottom margin, seen from the photographer's own "
                      "eyes: their two hands holding a smartphone up toward the scene in portrait orientation, "
                      "thumbs on the edges, the lower half of the phone and the hands visible. Real photograph, same "
                      "light. Keep the photo itself exactly as it is. " + scene, [pad(0)], seed, width=W, height=H,
                   loras=[("klein_skin_texture.safetensors", 0.6)])
    end = g.add("ImageCompositeMasked", destination=filled(), source=pad(0), x=0, y=0, resize_source=False,
                mask=g.add("InvertMask", mask=pad(1))())
    # the same frame a moment earlier, before the hands come up: the margin filled with only the place
    bare = klein(g, "ref_image_0 is a real photo with an empty grey margin below it. Fill the margin with more of "
                    "the same place: the ground in front, continuing the photo naturally, nothing else in it. Real "
                    "photograph, same light. Keep the photo itself exactly as it is. " + scene, [pad(0)], seed + 1,
                 width=W, height=H)
    g.add("SaveImage", images=end(), filename_prefix=prefix + "_end")
    # what moves: the photo's movers, and the band the hands rise through
    sam = g.add("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    movers = g.add("SAM3_Detect", model=sam(0), image=end(), threshold=0.25, refine_iterations=2,
                   individual_masks=False, conditioning=g.add("CLIPTextEncode", text=MOVERS, clip=sam(1))())
    # the hands and phone where Klein put them, smeared straight down to the frame edge: the path they rise along,
    # and nothing else (a full-width band gave the model room to drive a car in)
    held = g.add("SAM3_Detect", model=sam(0), image=end(), threshold=0.3, refine_iterations=2, individual_masks=False,
                 conditioning=g.add("CLIPTextEncode", text="hand:2, smartphone:1", clip=sam(1))())
    hands = g.add("MomentoRiseMask", mask=held(0))
    one = g.add("GrowMask", mask=g.add("MaskComposite", destination=movers(0), source=hands(), x=0, y=0,
                                       operation="add")(), expand=6, tapered_corners=True)
    mask = g.add("ImageToMask", channel="red",
                 image=g.add("RepeatImageBatch", image=g.add("MaskToImage", mask=one())(), amount=F)())
    still = g.add("RepeatImageBatch", image=end(), amount=F)  # outside the mask, every frame is the real photo
    model = g.add("LoraLoaderModelOnly", lora_name=TURBO, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3U, weight_dtype="default")())
    model = g.add("LoraLoaderModelOnly", lora_name=REAL, strength_model=1.0, model=model())
    vae, avae = g.add("VAELoader", vae_name=VAE), g.add("VAELoader", vae_name=AVAE)
    model = g.add("MiniMaxH3FunControlNetApply", model=model(), vae=vae(), strength=1.0, start_percent=0.0,
                  end_percent=1.0, model_patch=g.add("ModelPatchLoader", name=CN)(), mask=mask(), source_video=still())
    text = ("integrated_multimodal_description: r34l1sm Seen from the photographer's own eyes, handheld and nearly "
            f"still. {action} Their two hands bring a smartphone up from below into view and steady it, framing the "
            f"shot of <Picture 1>. {scene}\noverall_soundscape: the place's ambient sound, then the click of a phone "
            "camera shutter\nnon_diegetic_music: N/A")
    start = g.add("ImageCompositeMasked", destination=end(), source=bare(), x=0, y=0, resize_source=False,
                  mask=hands())  # the end frame with the hands and phone taken out
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=W, height=H, length=F, first_frame=start(),
                 last_frame=end(), vae=vae(),
                 clip=g.add("CLIPLoader", clip_name=CLIP, type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=6, denoise=1.0)(),
                latent_image=cond(1))
    gen = g.add("VAEDecode", samples=out(0), vae=vae())
    final = g.add("ImageCompositeMasked", destination=still(), source=gen(), x=0, y=0, resize_source=False,
                  mask=mask())
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=final(),
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_final", format="mp4", codec="h264")
    g.add("SaveImage", images=g.add("MaskToImage", mask=one())(), filename_prefix=prefix + "_mask")
    return g


def main(post, out, seed=None):
    seed = int(seed) if seed else random.randrange(2**31)
    pu = comfy.upload(post, f"momento_{Path(post).name}")
    files = run(graph(pu, os.environ.get("SCENE", ""), os.environ.get("ACTION", ""), seed,
                      f"momento/pov_{Path(post).stem}_{seed}"))
    fetch(files["final"][0], out)
    fetch(files["end"][0], str(out).replace(".mp4", "_end.png"))
    fetch(files["mask"][0], str(out).replace(".mp4", "_mask.png"))
    print("seed", seed, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
