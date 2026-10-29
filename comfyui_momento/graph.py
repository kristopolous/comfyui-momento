"""The whole app as ONE API-format ComfyUI graph. Every default is the LOWEST resolution.

    python scripts/graph.py > workflows/momento_api.json
"""
import math

try:
    from .director import director as plan_prompt
except ImportError:  # imported as a plain module by scripts/
    from director import director as plan_prompt

KLEIN = "flux3-image-klein-int8-convrot.safetensors"
KLEIN_TE = "qwen3vl_4b_int8_convrot.safetensors"
FLUX_VAE = "flux2-vae.safetensors"
LTX = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # distilled, stage 1 only


class Graph(dict):
    def add(self, class_type, title=None, **inputs):
        nid = str(len(self) + 1)
        self[nid] = {"class_type": class_type, "inputs": inputs, **({"_meta": {"title": title}} if title else {})}
        return lambda slot=0: [nid, slot]


def momento(profile="profile.png", post="post.png", width=288, height=512, frames=97, fps=24, seed=1,
            angle="behind", prefix="momento/moment"):
    """angle: "behind" (over the shoulder, ending on the post itself as the last keyframe) or "face" (from the side,
    the photographer's face in view)."""
    g = Graph()
    who = g.add("LoadImage", "Profile picture", image=profile)
    shot = g.add("LoadImage", "The post", image=post)
    both = g.add("ImageStitch", image1=who(0), image2=shot(0), direction="right", match_image_size=True,
                 spacing_width=0, spacing_color="white")
    gemini = g.add("GeminiNode", "Director (Gemini)", prompt="Plan the moment this post was taken.",
                   model="gemini-3-1-pro", seed=seed, images=both(0), system_prompt=plan_prompt(angle))

    kclip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    lclip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    plan = g.add("MomentoPlan", plan=gemini(0), image_clip=kclip(), video_clip=lclip())

    # FLUX 3 Klein: the bystander's frame, built from the post's objects (ref 0) and the person (ref 1)
    kvae = g.add("VAELoader", vae_name=FLUX_VAE)
    mu = math.log(16.0)
    kmodel = g.add("ModelSamplingFlux", model=g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")(),
                   max_shift=mu, base_shift=mu, width=width, height=height)
    refs = g.add("ReferenceMultiLatent", positive=plan(0), negative=plan(1), vae=kvae(), resolution="resize",
                 **{"resolution.megapixels": 0.07, "images.image0": shot(0), "images.image1": who(0)})
    still = g.add("KSampler", model=kmodel(), seed=seed, steps=16, cfg=1.0, sampler_name="euler", scheduler="simple",
                  positive=refs(0), negative=refs(1), denoise=1.0,
                  latent_image=g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)())
    frame = g.add("VAEDecode", samples=still(), vae=kvae())

    # LTX-2.5: the shot being taken, with the place's sound
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    lat = g.add("LTXVImgToVideoInplace", vae=vae(), image=frame(), strength=1.0, bypass=False,
                latent=g.add("EmptyLTXVLatentVideo", width=width, height=height, length=frames, batch_size=1)())
    audio = g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=avae())
    cond = g.add("LTXVConditioning", positive=plan(2), negative=plan(3), frame_rate=float(fps))
    pos, neg, video_latent = cond(0), cond(1), lat()
    if angle == "behind":
        # last keyframe: the post itself, so the camera ends on exactly what the phone sees
        view = g.add("MomentoFit", image=shot(0), width=width, height=height)
        guide = g.add("LTXVAddGuide", positive=pos, negative=neg, vae=vae(), latent=video_latent, image=view(),
                      frame_idx=-1, strength=0.9)
        pos, neg, video_latent = guide(0), guide(1), guide(2)
    av = g.add("LTXVConcatAVLatent", video_latent=video_latent, audio_latent=audio())
    lmodel = g.add("UNETLoader", unet_name=LTX, weight_dtype="default")
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=lmodel(), positive=pos, negative=neg, cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    video_out = sep(0)
    if angle == "behind":  # drop the keyframe's extra latent before decoding
        video_out = g.add("LTXVCropGuides", positive=pos, negative=neg, latent=video_out)(2)
    moment = g.add("CreateVideo", images=g.add("VAEDecode", samples=video_out, vae=vae())(),
                   audio=g.add("LTXVAudioVAEDecode", samples=sep(1), audio_vae=avae())(), fps=float(fps))
    music = g.add("ComfyCloudMiniMaxMusic3TextToAudioNode", "Soundtrack (MiniMax Music 3)", prompt=plan(4), lyrics="",
                  seed=seed, max_duration=10.0, audio_quality="128k")
    final = g.add("MomentoReveal", moment=moment(), post=shot(0), hold=1.8, flash=0.25, music=music(0),
                  music_level=0.35)
    g.add("SaveVideo", "Momento", video=final(), filename_prefix=prefix, format="mp4",
          **{"format.codec": "h264"})
    return g
