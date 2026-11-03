"""LTX-2.5 Refine-Details: a finished take, re-rendered at twice the size with real texture in place of VAE softness.

    COMFY_URL=http://<gpu-host>:8188 python scripts/refine.py take.mp4 out.mp4 [scale]

Lightricks' recipe (ComfyUI-LTXVideo example_workflows/2.5/LTX-2.5_V2V_TiledFusion_Upscale.json): the take, scaled to
the output size, is the IC-LoRA guide at frame 0; the tiled fusion sampler runs the 8 distilled steps over
overlapping tiles; the take's own sound is carried through. LTX needs 8k+1 frames, so a 107-frame H3 take loses
its first two.
"""
import json
import subprocess
import sys
from pathlib import Path

import comfy
from local import Graph, fetch, run  # local sets up the import path graph.py needs

LTX = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE = "ltx-2.5-video-vae-bf16.safetensors"
REFINE = "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LOOK = ("Photographic, natural daylight colour, crisp fine detail and real texture, sharp edges, light natural film "
        "grain, true-to-life skin.")  # style only: anything named here would stamp into every tile


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s["nb_frames"]), float(num) / float(den)


def graph(take, w, h, frames, fps, scale, prefix, seed=42):
    W, H = w * scale // 32 * 32, h * scale // 32 * 32
    n = (frames - 1) // 8 * 8 + 1
    g = Graph()
    src = g.add("GetVideoComponents", video=g.add("LoadVideo", file=take)())
    clip = g.add("ImageFromBatch", image=src(0), batch_index=frames - n, length=n)  # keep the end: it's the instant
    big = g.add("ImageScale", image=clip(), upscale_method="lanczos", width=W, height=H, crop="disabled")
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    te = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    ic = g.add("LTXICLoRALoaderModelOnly", model=g.add("UNETLoader", unet_name=LTX, weight_dtype="default")(),
               lora_name=REFINE, strength_model=1.0)
    cond = g.add("LTXVConditioning", frame_rate=fps,
                 positive=g.add("CLIPTextEncode", text=LOOK, clip=te())(),
                 negative=g.add("CLIPTextEncode", text="blurry, soft, smeared, oversharpened halos, artifacts",
                                clip=te())())
    guide = g.add("LTXAddVideoICLoRAGuide", positive=cond(0), negative=cond(1), vae=vae(), image=big(),
                  latent=g.add("EmptyLTXVLatentVideo", width=W, height=H, length=n, batch_size=1)(), frame_idx=0,
                  strength=1.0, latent_downscale_factor=ic(1), crop="disabled", use_tiled_encode=False,
                  tile_size=256, tile_overlap=64, use_streaming=True, tile_frames=97)
    out = g.add("LTXVTiledFusionSampler", model=ic(0), positive=guide(0), negative=guide(1), latents=guide(2),
                sigmas=g.add("ManualSigmas", sigmas=SIGMAS)(), sampler=g.add("KSamplerSelect", sampler_name="euler")(),
                seed=seed, cfg=1.0, tile_width=w // 32 * 32, tile_height=h // 32 * 32, overlap_frac=0.5,
                blend_var=0.05, grid_cycle=1, tile_frames=97, vae=vae(), canvas_device="auto")
    images = g.add("VAEDecodeTiled", samples=out(0), vae=vae(), tile_size=512, overlap=64, temporal_size=128,
                   temporal_overlap=32)
    g.add("SaveVideo", video=g.add("CreateVideo", images=images(), audio=src(1), fps=fps)(),
          filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def main(src, dst, scale=2):
    w, h, frames, fps = probe(src)
    name = comfy.upload(src, f"momento_refine_{Path(src).name}")
    files = run(graph(name, w, h, frames, fps, scale, f"momento/refine_{Path(src).stem}"))
    fetch(files["final"][0], dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))
