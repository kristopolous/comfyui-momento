"""The real place, with only what's in it moving.

    COMFY_URL=... CAMERA=smartphone python scripts/motion.py profile.jpg post.jpg out.mp4 [seed]

  - The post becomes a 3D place (Depth Anything 3) and is rendered, textured, along the bystander's camera move.
    That render is kept as it is: roads stay roads, nothing new drives through anything.
  - SAM 3.1 finds what the photo shows as able to move (people, cars, scooters, umbrellas) in every frame.
  - Momento Photographer places the photographer from the 3D: a DWPose skeleton and a mask.
  - MiniMax H3 Ref2VA (Fun ControlNet-Union inpaint, Realism People) regenerates only those regions, plus the gaps
    the camera sees past the photo; the photographer's look comes from their restored profile picture.
  - Everything outside the mask is composited back from the render.
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
import geo  # noqa: E402
from local import Graph, fetch, klein, run  # noqa: E402

W, H, F, FPS = int(os.environ.get("MW", 256)), int(os.environ.get("MH", 512)), 73, 24
CAMERA = os.environ.get("CAMERA", "smartphone")
MOVE = os.environ.get("MOVE", "dolly")
PULL = float(os.environ.get("PULL_BACK", "0.35"))  # just over the shoulder: almost every pixel is the photo's
MOVERS = os.environ.get("MOVERS", "person:12, car:8, scooter:8, bus:3, umbrella:6")
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TURBO = "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
REAL = "h3_realism_people.safetensors"
CN = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
CLIP, VAE, AVAE = ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors",
                   "minimax_h3_audio_vae_fp32.safetensors")


def path():
    k = PULL / geo.PULL_BACK
    sc = lambda e: (e[0] * k, e[1], e[2] * k)  # noqa: E731
    a, b = sc(geo.MOVES.get(MOVE, geo.START)), sc(geo.START)
    ease = lambda t: t * t * (3 - 2 * t)  # noqa: E731
    return [tuple(p + (q - p) * ease(i / (F - 1)) for p, q in zip(a, b)) for i in range(F)]


def batch(g, items):
    out = items[0]
    for x in items[1:]:
        out = g.add("ImageBatch", image1=out, image2=x)()
    return out


def graph(profile, post, scene, action, seed, prefix):
    g = Graph()
    shot = g.add("LoadImage", image=post)
    who = klein(g, "Restore and enlarge this photo of a person: the same face, features, skin tone, hair and age, as "
                   "a sharp natural photograph with real skin texture. Change nothing about who they are.",
                [g.add("LoadImage", image=profile)(0)], seed, width=768, height=768,
                loras=[("klein_restore.safetensors", 1.0)])
    geom = g.add("DA3Inference", image=shot(0), resolution=504, resize_method="upper_bound_resize", mode="mono",
                 da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                 weight_dtype="default")())
    mesh = g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1, discontinuity_threshold=0.04,
                 confidence_threshold=0.1, use_sky_mask=True, texture=True)
    views, holes = [], []
    for eye in path():
        cam = g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", eye)},
                    target_x=geo.TARGET[0], target_y=geo.TARGET[1], target_z=geo.TARGET[2], roll=0.0, fov=geo.LENS,
                    zoom=1.0, camera_type="perspective")
        r = g.add("RenderMesh", mesh=mesh(), mode="texture", width=W, height=H, background="#808080",
                  camera_info=cam())
        views.append(r(0))
        holes.append(g.add("MaskToImage", mask=g.add("InvertMask", mask=r(1))())())
    place = batch(g, views)
    hole = g.add("ImageToMask", image=batch(g, holes), channel="red")
    # what in the photo can move, frame by frame
    sam = g.add("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    movers = g.add("SAM3_Detect", model=sam(0), image=place, threshold=0.4, refine_iterations=2, individual_masks=False,
                   conditioning=g.add("CLIPTextEncode", text=MOVERS, clip=sam(1))())
    who3d = g.add("MomentoPhotographer", move=MOVE, camera=CAMERA, frames=F, width=W, height=H, pull_back=PULL)
    mask = g.add("MaskComposite", destination=hole(), source=movers(0), x=0, y=0, operation="add")
    mask = g.add("MaskComposite", destination=mask(), source=who3d(1), x=0, y=0, operation="add")
    mask = g.add("GrowMask", mask=mask(), expand=4, tapered_corners=True)
    # H3 Ref2VA regenerates only the masked regions, the photographer under the projected skeleton
    model = g.add("LoraLoaderModelOnly", lora_name=TURBO, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=REF2VA, weight_dtype="default")())
    model = g.add("LoraLoaderModelOnly", lora_name=REAL, strength_model=1.0, model=model())
    model = g.add("MiniMaxH3SigmaShift", model=model(), shift_video=12.0, shift_audio=3.0)
    vae, avae = g.add("VAELoader", vae_name=VAE), g.add("VAELoader", vae_name=AVAE)
    model = g.add("MiniMaxH3FunControlNetApply", model=model(), vae=vae(), strength=1.0, start_percent=0.0,
                  end_percent=1.0, model_patch=g.add("ModelPatchLoader", name=CN)(), control_video=who3d(0),
                  mask=mask(), source_video=place)
    text = (f"integrated_multimodal_description: r34l1sm The same shot of the same real place, the camera moving "
            f"slowly sideways. Everyone and everything already in the scene keeps moving naturally where it is: "
            f"{action} In the foreground stands the person from <Picture 1>, seen from behind, raising a "
            f"{'phone' if CAMERA != 'dslr' else 'camera'} to take a photo. {scene}"
            f"\noverall_soundscape: the place's ambient sound\nnon_diegetic_music: N/A")
    cond = g.add("MiniMaxH3ReferenceToVideo", prompt=text, width=W, height=H, length=F, ref_image_size="match",
                 vae=vae(), audio_vae=avae(), clip=g.add("CLIPLoader", clip_name=CLIP, type="minimax",
                                                        device="default")(), **{"ref_images.ref_image_0": who()})
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    gen = g.add("VAEDecode", samples=out(0), vae=vae())
    final = g.add("ImageCompositeMasked", destination=place, source=gen(), x=0, y=0, resize_source=False, mask=mask())
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=final(),
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_final", format="mp4", codec="h264")
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=g.add("MaskToImage", mask=mask())())(),
          filename_prefix=prefix + "_mask", format="mp4", codec="h264")
    return g


def main(profile, post, out, seed=None):
    seed = int(seed) if seed else random.randrange(2**31)
    geo.W, geo.H = W, H
    scene = os.environ.get("SCENE", "")
    action = os.environ.get("ACTION", "people walk on the sidewalk, cars drive in their lanes.")
    pu = comfy.upload(profile, f"momento_{Path(profile).name}")
    su = comfy.upload(post, f"momento_{Path(post).name}")
    files = run(graph(pu, su, scene, action, seed, f"momento/motion_{Path(post).stem}_{seed}"))
    fetch(files["final"][0], out)
    fetch(files["mask"][0], str(out).replace(".mp4", "_mask.mp4"))
    print("seed", seed, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
