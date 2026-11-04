"""The Comfy Cloud pipeline that made the approved v2 clips (scripts/moment.py), on your own GPU.

    COMFY_URL=... PLANNER_URL=... PLANNER_MODEL=... python scripts/moment_local.py profile.jpg post.jpg [more posts...]

Same steps as moment.py; only the hosted nodes are replaced: FLUX 3 Image -> FLUX 3 Klein, Gemini -> the local planner.
  0. Framing: the bystander's lens and tilt are the widest at which the post fills the frame (measured on the 3D render),
     so nothing below the photo has to be invented (that strip is where a sidewalk grew halfway up the bus).
  A. Plan; character sheet; the instant (the post's 3D place seen from the bystander, gaps filled seamlessly).
  B. MiniMax H3 films the scene with no photographer, camera nearly still, ending on the instant.
  C. Klein places the photographer into the scene's last frame, in the box projected from 3D; SAM 3 cuts them out
     onto the first frame, so both ends hold the same phone (two placements gave two phones, morphed mid-clip).
  D. LTX-2.5 + the in/outpainting IC-LoRA regenerates only the photographer's region (a mask projected from 3D)
     through the whole clip, pinned to the placed first and last frames; the rest stays H3's.
The camera dollies in (MOVE="dolly in", the default; MOVE=still holds it): it opens further back and higher, looking
level, and pushes in, tilting up to the framed end. The photographer is placed once, whole, in that wide opening and
carried to the close end by the 3D projection; H3 films the move between the two (no inpaint pass: the photographer
moves with the camera, which a fixed mask can't follow).
SEED, PLAN_JSON and TAG work as in local.py, for A/B tests.
"""
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import local  # noqa: E402  (reads --hires: 448x768)
from local import (CAMERA, FPS, H, H3, SHEET, W, Graph, ask, comfy, fetch, geo,  # noqa: E402
                   klein, rows, run)
from cloud import reveal  # noqa: E402

H3_FRAMES, LTX_FRAMES = 107, 105
LTX = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE, LTX_AUDIO_VAE = "ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors"
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
INPAINT = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
QUIET = " The only sounds are the place's own ambient sound; nobody speaks."
NEG = "blurry, distorted, extra limbs, extra fingers, warped hands, text, watermark, cartoon, speech"


MOVE = None if os.environ.get("MOVE", "dolly in") == "still" else os.environ.get("MOVE", "dolly in")
# the plan, with the place's sound only: the shutter belongs to the reverse shot's last beat, not the whole take
PLAN = local.PLAN.replace(", ending with the crisp click of a phone camera\n   shutter", "")
assert PLAN != local.PLAN
# the reverse shot cut in before the reveal, by what the photographer holds (REVERSE=none skips it):
#   smartphone: from the subject's side, the phone held sideways in both hands, its lenses aimed down ours, the eyes
#   over the top of it (the face stays in view; "phone front"/"phone front mid" hide it behind the phone)
#   dslr: Rear Window, the eye at the viewfinder, the long lens leaving the frame past us
CANDIDATES, FACE_MAX = 4, 3  # placements tried; face landmarks allowed (a stray dot or two) on someone seen from behind
PUSH_TO = 0.6  # when backing up would show what the photo doesn't have, the move pushes in from START to here (x START)
REVERSE = os.environ.get("REVERSE", "front") != "none"  # REVERSE=none: no reverse shot, the take ends on the push-in
# The reverse shot is built from the push-in's own photographer, so the clothes, the camera and the moment carry across
# the cut; only the face comes from the profile picture.
FRONT = {"smartphone": "both arms raised, holding an ordinary {shade} smartphone {held} in both hands at eye level in "
                       "front of the face, its back toward the camera, looking at its screen, lips closed, concentrating",
         "dslr": "holding a {shade} DSLR camera with a lens up to the eye in both hands, looking through the viewfinder, "
                 "lips closed, concentrating{held:.0}"}
LENS_OF = {"smartphone": "phone", "dslr": "camera"}  # what SAM 3 finds as the camera the shot pushes in on
FRAME_H = 0.69  # the reverse shot's framing: the top 69% of the waist-up still, head to mid-chest
REV_SCALE, REV_FRAMES = 1.5, 39  # rendered at 1.5x so the push-in on the camera stays sharp; H3's 17k+5, ~1.6 s
REV_KEEP = 1.33  # the reverse shot's beat: two thirds of the two seconds it had (it ran long)
PUNCH, PUNCH_TIME = 1.8, 0.6  # it ends pushing in 1.8x on the camera over its last 0.6 s, as the shutter fires


def cutout(hi, matte, path):
    """The push-in's photographer alone on a plain light square: the multiple-angles LoRA's own input format (it was
    trained on objects on a plain backdrop; given the whole street it drew studio objects)."""
    im = np.asarray(Image.open(hi).convert("RGB")).astype(float)
    a = np.asarray(Image.open(matte).convert("L").resize((im.shape[1], im.shape[0]))).astype(float)[..., None] / 255
    flat = Image.fromarray((im * a + 215 * (1 - a)).astype(np.uint8))
    ys, xs = np.nonzero(a[..., 0] > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * 1.15)
    sq = Image.new("RGB", (side, side), (215, 215, 215))
    sq.paste(flat.crop((x0, y0, x1, y1)), ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
    sq.save(path)


def held(cut, prefix):
    """How the push-in's photographer holds the camera, measured from SAM 3's mask of it in the cut-out: taller than
    wide is upright, else sideways; and its shade, from the camera's own pixels (black or silver). The reverse shot
    poses the same grip and the same camera. -> (way, shade)"""
    g = Graph()
    sam = g.add("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    found = g.add("SAM3_Detect", model=sam(0), image=g.add("LoadImage", image=cut)(0), threshold=0.3,
                  refine_iterations=2, individual_masks=False,
                  conditioning=g.add("CLIPTextEncode", text=LENS_OF[CAMERA], clip=sam(1))())
    g.add("SaveImage", images=g.add("MaskToImage", mask=found(0))(), filename_prefix=prefix + "_heldcam")
    f = run(g)["heldcam"][0]
    q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
    mask = np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{q}"))).convert("L")) > 127
    ys, xs = np.nonzero(mask)
    way = "upright" if len(xs) and np.ptp(ys) > np.ptp(xs) else "sideways"
    pixels = np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{urllib.parse.urlencode(dict(filename=cut, type='input'))}")))
                        .convert("L").resize(mask.shape[::-1]))
    rim = mask & ~ndimage.binary_erosion(mask, iterations=max(1, round(0.15 * min(np.ptp(ys), np.ptp(xs))))) \
        if len(xs) else mask  # its frame: from behind, the middle is the lit screen
    shade = "black" if not rim.any() or np.median(pixels[rim]) < 90 else "silver"
    print(f"the push-in's {LENS_OF[CAMERA]} is {shade}, held {way}", flush=True)
    return way, shade


def front_view(sheet, prefix, path):
    """The character sheet's front view, on a plain light square (the BFS body swap's reference format). BiRefNet's
    alpha separates the figures from the sheet's studio backdrop; the empty columns between them split them; the
    front view is the first, left to right."""
    g = Graph()
    matte = g.add("RemoveBackground", image=g.add("LoadImage", image=comfy.upload(str(sheet)))(0),
                  bg_removal_model=g.add("LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")())
    g.add("SaveImage", images=g.add("MaskToImage", mask=matte(0))(), filename_prefix=prefix + "_figures")
    f = run(g)["figures"][0]
    q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
    im = Image.open(sheet).convert("RGB")
    alpha = np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{q}"))).convert("L").resize(im.size)) / 255
    runs, n = ndimage.label(alpha.max(0) > 0.5)
    if not n:
        raise SystemExit(f"no figure found on the character sheet {sheet}")
    xs = np.nonzero(runs == 1)[0]
    alpha = alpha * (runs == 1)[None]  # only the first figure, none of its neighbour's arm
    a = np.asarray(im).astype(float)
    flat = Image.fromarray((a * alpha[..., None] + 225 * (1 - alpha[..., None])).astype(np.uint8))  # on light grey
    ys = np.nonzero(alpha[:, xs].max(1) > 0.5)[0]
    fig = flat.crop((max(0, xs[0] - 8), max(0, ys[0] - 8), min(im.width, xs[-1] + 9), min(im.height, ys[-1] + 9)))
    side = max(fig.size)
    sq = Image.new("RGB", (side, side), (225, 225, 225))
    sq.paste(fig, ((side - fig.width) // 2, (side - fig.height) // 2))
    sq.save(path)
    print(f"character sheet: {n} figures, the front view at x {xs[0]}-{xs[-1]}", flush=True)


def framed(size, skeleton, cam):
    """The reverse shot's framing in the waist-up still (posing a whole upright torso keeps them standing): the top
    FRAME_H, in the frame's aspect, centred between the face (the skeleton's face points, white) and the camera
    (SAM 3's mask), so both stay in. -> the crop box."""
    h = round(size[1] * FRAME_H)
    w = round(h * W / H)
    face = np.nonzero((np.asarray(skeleton.convert("RGB").resize(size))[: size[1] // 2] > 200).all(2))[1]
    lens = np.nonzero(np.asarray(cam.convert("L").resize(size)) > 127)[1]
    xs = [v.mean() for v in (face, lens) if len(v)]
    x0 = round(max(0, min(size[0] - w, (np.mean(xs) if xs else size[0] / 2) - w / 2)))
    return x0, 0, x0 + w, h


# the pose ControlNet is built for the Qwen-Image 2.1 base model at ~40 steps; on Turbo it barely holds the skeleton
QWEN_BASE, QWEN_STEPS = "qwen_image_2.1_int8_convrot.safetensors", 40
HEAD_SWAP = "bfs_head_v1.1_qwen_2.1.safetensors"  # Alissonerdx/BFS-Best-Face-Swap, MIT
TILTS = [round(-0.5 + 0.2 * i, 1) for i in range(14)]  # look-at heights 4 m out, a little down to ~20 degrees up
LENSES = (62.0, 56.0, 52.0, 48.0, 44.0, 40.0)  # the bystander's vertical fov: the widest one the photo fills
COVER_FRAME = 0.85  # what's left is thin slivers behind near things (the view is off to the side), filled


def mesh(g, shot, sky=False):
    """The post's 3D place. `sky`: keep the photo's own sky in it (a far backdrop), so every view of the move sees
    the same real clouds; without it each end's sky is invented separately and H3 races clouds between the two."""
    geom = g.add("DA3Inference", image=shot, resolution=504, resize_method="upper_bound_resize", mode="mono",
                 da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                 weight_dtype="default")())
    return g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1, discontinuity_threshold=0.04,
                 confidence_threshold=0.1, use_sky_mask=not sky, texture=True)


def camera(g, eye=None, target=None):
    t = target or geo.TARGET
    return g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", eye or geo.START)},
                 target_x=t[0], target_y=t[1], target_z=t[2], roll=0.0, fov=geo.LENS, zoom=1.0,
                 camera_type="perspective")


def framing(post, prefix, eyes=None):
    """Sets geo.LENS and geo.TARGET: the widest lens (of LENSES) and its tilt at which the post covers at least
    COVER_FRAME of the bystander's whole frame (the render's coverage mask, the photo's own sky included) from every
    one of `eyes` (default: START). A lens wider than the photo sees past its edges, where there is nothing real:
    that is where the signs floated, their poles outside the photo."""
    g, level, eyes = Graph(), geo.TARGET, eyes or [geo.START]
    m = mesh(g, g.add("LoadImage", image=post)(0), sky=True)
    for a, lens in enumerate(LENSES):
        geo.LENS = lens
        for i, ty in enumerate(TILTS):
            for j, eye in enumerate(eyes):
                view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                             camera_info=camera(g, eye, (level[0], ty, level[2]))())
                g.add("SaveImage", images=g.add("MaskToImage", mask=view(1))(),
                      filename_prefix=f"{prefix}_cover{a}t{i}e{j}")
    files = run(g)

    def cover(a, i):
        out = []
        for j in range(len(eyes)):
            f = files[f"cover{a}t{i}e{j}"][0]
            q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
            out.append((np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{q}"))).convert("L")) > 127).mean())
        return min(out)

    best = None  # the widest lens that reaches COVER_FRAME, else the best covered of all
    for a, lens in enumerate(LENSES):
        here = max((cover(a, i), lens, ty) for i, ty in enumerate(TILTS))
        if here[0] >= COVER_FRAME:
            best = here
            break
        best = max(best or here, here)
    cover_, geo.LENS, ty = best
    geo.TARGET = (level[0], ty, level[2])
    print(f"framing: lens {geo.LENS:.0f} deg, look-at height {ty} m, frame covered {cover_:.2f}", flush=True)


def reach(post, prefix, opening):
    """How far back the camera move may start: along the line from START to `opening`, the farthest point whose
    view the photo covers on at least COVER_FRAME of the whole frame, as framing() asks of START. Further back, the
    view is mostly invented: signs float, a car grows under the bus. None if no point is: then the move can't back
    up at all."""
    g = Graph()
    m = mesh(g, g.add("LoadImage", image=post)(0), sky=True)
    steps = (1.0, 0.75, 0.5, 0.3)
    eyes = [tuple(a + (b - a) * f for a, b in zip(geo.START, opening)) for f in steps]
    for i, eye in enumerate(eyes):
        view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                     camera_info=camera(g, eye)())
        g.add("SaveImage", images=g.add("MaskToImage", mask=view(1))(), filename_prefix=f"{prefix}_reach{i}")
    files = run(g)
    for i, (f, eye) in enumerate(zip(steps, eyes)):
        q = urllib.parse.urlencode({k: files[f"reach{i}"][0][k] for k in ("filename", "subfolder", "type")})
        cover = (np.asarray(Image.open(io.BytesIO(comfy._req(f"/view?{q}"))).convert("L")) > 127).mean()
        print(f"camera move at {f:.2f} of its length: frame covered {cover:.2f}", flush=True)
        if cover >= COVER_FRAME:
            return eye
    return None


def h3(g, last, prompt, seed, first=None):
    model = g.add("LoraLoaderModelOnly", lora_name=H3["lora"], strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    cond = g.add("MiniMaxH3ImageToVideo", prompt=prompt, width=W, height=H, length=H3_FRAMES, last_frame=last, vae=vae(),
                 **({"first_frame": first} if first is not None else {}),
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    return g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                 audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())


def stage_ab(profile, post, plan, seed, prefix):
    g = Graph()
    who, shot = g.add("LoadImage", image=profile), g.add("LoadImage", image=post)
    sheet = klein(g, SHEET + plan["photographer"], [who(0)], seed, width=768, height=448)
    g.add("SaveImage", images=sheet(), filename_prefix=prefix + "_sheet")
    m = mesh(g, shot(0))
    view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                 camera_info=camera(g)())
    fill = ("ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around things: "
            "fill them seamlessly with more of the same place, same light, same lens. Keep everything else in "
            "ref_image_0 exactly as it is. No people in the foreground. " + plan["scene"])
    instant = klein(g, fill, [view(0)], seed)
    g.add("SaveImage", images=instant(), filename_prefix=prefix + "_keys")
    text = ("integrated_multimodal_description: Handheld bystander camera, nearly still. The subject is in motion and "
            f"ends exactly in the moment of <Picture 1>: {plan['action']} {plan['scene']}"
            f"\noverall_soundscape: {plan['soundscape']}\nnon_diegetic_music: N/A")
    g.add("SaveVideo", video=h3(g, instant(), text, seed)(), filename_prefix=prefix + "_scene", format="mp4",
          codec="h264")
    return g


BODY = [(x, y, z) for x in (-0.26, 0.26) for y in (0.18, -1.7) for z in (-0.5, 0.2)]


def mask_video(path, pad=None):
    """The photographer's region in every frame, projected from 3D (the bystander holds still), as a mask video."""
    tmp = tempfile.TemporaryDirectory()
    d = Path(tmp.name)
    pad = pad if pad is not None else round(W / 25)
    y0, x0, y1, x1 = geo.box(BODY, geo.START)
    for i in range(LTX_FRAMES):
        im = Image.new("RGB", (W, H), "black")
        ImageDraw.Draw(im).rectangle((x0 * W / 1000 - pad, y0 * H / 1000 - pad, x1 * W / 1000 + pad,
                                      y1 * H / 1000 + pad), fill="white")
        im.save(d / f"{i:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "10", str(path)], check=True)
    tmp.cleanup()


PLACE = ("Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the foreground, "
         "seen from behind and slightly to the side, turned toward the subject, {hold}, aimed at the subject, taking "
         "a photo. Same hair and clothes as the sheet. Exactly one photographer. Change nothing else in ref_image_0. ")


def opening_layout(opening):
    """The move holds framing()'s tilt from end to end: looking level at the opening showed the ground below the
    photo's bottom edge, which isn't in it."""
    return geo.layout(opening, CAMERA), geo.box(BODY, opening)


def move_keys(profile, post, box, plan, seed, prefix, opening):
    """A camera move's two ends, with the photographer placed once. The 3D place seen from `opening` (further back,
    looking level) and from START (framed); gaps filled; Klein puts the photographer, whole, into the wide opening at
    twice the size; BiRefNet mattes them out (a real alpha), only inside their 3D box. compose() carries them to the close end."""
    g = Graph()
    who, shot = g.add("LoadImage", image=profile), g.add("LoadImage", image=post)
    sheet = klein(g, SHEET + plan["photographer"], [who(0)], seed, width=768, height=448)
    g.add("SaveImage", images=sheet(), filename_prefix=prefix + "_sheet")
    m = mesh(g, shot(0), sky=True)
    fill = ("ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around things: "
            "fill them seamlessly with more of the same place, same light, same lens. Keep everything else in "
            "ref_image_0 exactly as it is. No people in the foreground. " + plan["scene"])
    ends = {}
    for name, eye, target, s_ in (("open", opening, geo.TARGET, seed + 2), ("end", geo.START, geo.TARGET, seed)):
        view = g.add("RenderMesh", mesh=m(), mode="texture", width=W, height=H, background="#000000",
                     camera_info=camera(g, eye, target)())
        ends[name] = klein(g, fill, [view(0)], s_)
        g.add("SaveImage", images=ends[name](), filename_prefix=f"{prefix}_{name}fill")
    sam = g.add("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    bg = g.add("LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")
    pose = g.add("CheckpointLoaderSimple", ckpt_name="sdpose_wholebody_fp16.safetensors")
    for k in range(CANDIDATES):  # several placements; measure() keeps one that faces the right way, whole
        hi = klein(g, PLACE.format(hold=local.CAM["hold"]) + plan["photographer"] + rows(opening_layout(opening)[0]),
                   [ends["open"](), sheet()], seed + 1 + 10 * k, width=2 * W, height=2 * H)
        # which person: SAM 3 prompted with the concept and their 3D box (body, raised arms and phone); their
        # edges: BiRefNet's alpha, inside that instance (grown a little so hair and fingers keep their soft edge)
        found = g.add("SAM3_Detect", model=sam(0), image=hi(), threshold=0.3, refine_iterations=2,
                      individual_masks=False, bboxes=g.add("PrimitiveBoundingBox", **box)(),
                      conditioning=g.add("CLIPTextEncode", text="person holding a phone or camera", clip=sam(1))())
        matte = g.add("RemoveBackground", image=hi(), bg_removal_model=bg())
        mine = g.add("MaskComposite", destination=matte(0), operation="multiply", x=0, y=0,
                     source=g.add("GrowMask", mask=found(0), expand=12, tapered_corners=True)())
        # the camera itself: SAM 3's instance, solid. BiRefNet reads a phone screen showing the street as background,
        # and the push-in's phone went see-through
        held_cam = g.add("SAM3_Detect", model=sam(0), image=hi(), threshold=0.3, refine_iterations=2,
                         individual_masks=False, bboxes=g.add("PrimitiveBoundingBox", **box)(),
                         conditioning=g.add("CLIPTextEncode", text=LENS_OF[CAMERA], clip=sam(1))())
        mine = g.add("MaskComposite", destination=mine(), source=held_cam(0), operation="add", x=0, y=0)
        skel = g.add("SDPoseDrawKeypoints", keypoints=g.add("SDPoseKeypointExtractor", model=pose(0), vae=pose(2),
                                                            image=hi(), batch_size=1)(),
                     draw_body=True, draw_hands=False, draw_face=True, draw_feet=False, stick_width=4,
                     face_point_size=3, score_threshold=0.3, draw_head=True)
        g.add("SaveImage", images=hi(), filename_prefix=f"{prefix}_openhi{k}")
        g.add("SaveImage", images=g.add("MaskToImage", mask=mine())(), filename_prefix=f"{prefix}_openmask{k}")
        g.add("SaveImage", images=skel(), filename_prefix=f"{prefix}_openpose{k}")
    return g


def face_points(skeleton):
    """How many face landmarks SDPose drew (white dots): seen from behind, facing into the scene, there are none."""
    a = np.asarray(Image.open(skeleton).convert("RGB")).astype(int)
    return int(ndimage.label((a > 235).all(2))[1])  # one white disc per landmark


def pick(od, tag):
    """The placement to use: facing into the scene (no face landmarks) and the most of the person in the matte.
    None when no candidate faces the right way."""
    best = None
    for k in range(CANDIDATES):
        faces = face_points(od / f"{tag}_openpose{k}.png")
        area = float((np.asarray(Image.open(od / f"{tag}_openmask{k}.png").convert("L")) > 127).mean())
        print(f"placement {k}: face landmarks {faces}, person {area:.3f} of frame", flush=True)
        if faces <= FACE_MAX and area > 0.01 and (best is None or area > best[1]):
            best = (k, area)
    return None if best is None else best[0]


def compose(open_fill, end_fill, hi, cut, opening, first_path, last_path):
    """First keyframe: the opening with the photographer cut from `hi`. Last keyframe: the framed end with the same
    cut-out photographer, moved and scaled to where the 3D projection puts them from START."""
    f0, f1 = Image.open(open_fill).convert("RGB"), Image.open(end_fill).convert("RGB")
    hi, cut = Image.open(hi).convert("RGB"), Image.open(cut).convert("L")  # the alpha matte, as is
    f0.paste(hi.resize((W, H), Image.LANCZOS), (0, 0), cut.resize((W, H), Image.LANCZOS))
    head, chest = (0.0, 0.18, -0.15), (0.0, -0.6, -0.15)
    (ax, ay), (bx, by) = (geo.project(p, opening) for p in (head, chest))
    (ex, ey), (fx, fy) = (geo.project(p, geo.START) for p in (head, chest))
    k = math.hypot(fx - ex, fy - ey) / math.hypot(bx - ax, by - ay) / 2  # hi is twice the frame
    size = (round(hi.width * k), round(hi.height * k))
    f1.paste(hi.resize(size, Image.LANCZOS), (round(ex - 2 * ax * k), round(ey - 2 * ay * k)),
             cut.resize(size, Image.LANCZOS))
    f0.save(first_path)
    f1.save(last_path)


def move_film(first, last, plan, seed, prefix):
    """H3 films the move between the two keyframes, photographer and all: the same person at both ends."""
    g = Graph()
    text = (f"integrated_multimodal_description: The camera makes {geo.MOVE_WORDS[MOVE]}, moving "
            f"through the real space of the place. The photographer in the foreground, seen from behind, "
            f"{local.CAM['act']}. There is exactly one photographer. The subject is in motion and ends exactly in the "
            f"moment of <Picture 2>: {plan['action']} {plan['scene']}"
            f"\noverall_soundscape: {plan['soundscape']}\nnon_diegetic_music: N/A")
    g.add("SaveVideo", video=h3(g, g.add("LoadImage", image=last)(0), text, seed,
                                g.add("LoadImage", image=first)(0))(),
          filename_prefix=prefix + "_raw", format="mp4", codec="h264")
    return g


# Reverse shots of the photographer taking the picture, after the films the user pointed to. Their eyes are the origin,
# looking down -Z at the subject; each shot is a bystander position (angle off their line of sight, distance, height),
# a lens, and how the photographer stands, so the face and camera boxes come from the same 3D geometry.
SHOTS = {
    # Rear Window: from the front side, long lens; the lens leaves the frame's edge past us, eyes over the camera
    "rear window": dict(angle=35.0, dist=1.1, height=-0.03, fov=24.0, crouch=0.0,
                        words="A tight cinematic close-up on a long lens, shallow depth of field"),
    # Peeping Tom: almost down the lens; the lens fills the foreground, the face just behind the camera
    "peeping tom": dict(angle=10.0, dist=0.9, height=0.0, fov=28.0, crouch=0.0,
                        words="A cinematic close-up almost straight down the camera's lens, the lens huge in the "
                              "foreground, shallow depth of field"),
    # Blow-Up: low and wider, the photographer crouched and leaning in, shooting down at the subject
    "blow up": dict(angle=50.0, dist=1.6, height=-1.0, fov=50.0, crouch=0.55,
                    words="A low-angle 1960s fashion-shoot shot on a wider lens: the photographer crouched, leaning in, "
                          "shooting down at the subject with energy"),
}


def shot_eye(shot):
    a = math.radians(shot["angle"])
    return (shot["dist"] * math.sin(a), shot["height"] - shot["crouch"], -shot["dist"] * math.cos(a))


def close_layout(shot, camera=CAMERA):
    """Face and camera boxes for a reverse shot, projected from 3D (the photographer's own geometry)."""
    drop = shot["crouch"]
    eye, target = shot_eye(shot), (0.0, -0.03 - drop, -0.2)

    def bbox(points):
        px = [geo.project((x, y - drop, z), eye, target=target, fov=shot["fov"]) for x, y, z in points]
        clip = lambda v, m: max(0, min(1000, round(v / m * 1000)))  # noqa: E731
        return [clip(min(p[1] for p in px), H), clip(min(p[0] for p in px), W),
                clip(max(p[1] for p in px), H), clip(max(p[0] for p in px), W)]

    face = [(x, y, z) for x in (-0.08, 0.08) for y in (0.13, -0.13) for z in (-0.1, 0.08)]
    if camera == "dslr":
        held = [(x, y, z) for x in (-0.09, 0.09) for y in (-0.07, 0.07) for z in (-0.06, -0.38)]  # body and lens
        what = "a DSLR with a long lens at their eye, the lens pointing out past the viewer"
    else:
        held = [(x, y, -0.42) for x in (-0.08, 0.12) for y in (-0.06, 0.16)]
        what = "a phone held up in both hands, its back to the viewer, aimed past the viewer"
    return [{"type": "obj", "bbox": bbox(face), "desc": "face: image 2's person, eyes on the subject beyond the "
                                                          "camera"},
            {"type": "obj", "bbox": bbox(held), "desc": f"camera: {what}"}]


def close_still(profile, post, sheet, plan, seed, prefix, name="rear window"):
    """A reverse shot's still: the photographer from the profile picture and sheet, face and camera where the 3D
    layout puts them, in front of a soft blur of the post's own colours and light (no place to invent)."""
    shot = SHOTS[name]
    g = Graph()
    plate = g.add("ImageBlur", image=g.add("ImageScale", image=g.add("LoadImage", image=post)(0),
                                           upscale_method="lanczos", width=W, height=H, crop="center")(),
                  blur_radius=31, sigma=10.0)
    still = klein(g, shot["words"] + ": the photographer, the person from ref_image_2 (face) and ref_image_1 (clothes "
                     "and hair), in the act of taking a photo. Background: ref_image_0, out of focus. "
                     + plan["photographer"] + rows(close_layout(shot)),
                  [plate(), g.add("LoadImage", image=sheet)(0), g.add("LoadImage", image=profile)(0)], seed + 5)
    g.add("SaveImage", images=still(), filename_prefix=f"{prefix}_{name.replace(' ', '')}")
    return g


def reverse_still(profile, front, held_as, seed, prefix):
    """The reverse shot's still, each tool doing one job, everything but the face carried from the push-in:
    1. the clothes: the character sheet's front view (`front`), the same sheet the push-in's photographer was dressed
       from. Turning the push-in's back view around invented a front: an apron, a tie, sunglasses;
    2. the moment: Qwen-Image 2.1's edit re-poses that front view with the push-in's grip, at the front
       three-quarter angle, keeping all they wear. Posing a stranger and BFS-body-swapping the sheet's clothes on
       carried the shirt but dropped the backpack on most seeds;
    3. the BFS head swap puts the profile picture's head on them.
    With it: SDPose's skeleton of the still (H3 holds that pose) and SAM 3's mask of the camera, which the shot
    pushes in on."""
    g = Graph()
    vae = g.add("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")
    clip = g.add("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")
    base = g.add("UNETLoader", unet_name=QWEN_BASE, weight_dtype="default")

    def qwen(model, prompt, seed_, images=(), latent=None, resolution=1024, negative=""):
        enc = g.add("TextEncodeQwenImage21", clip=clip(), vae=vae(), resolution=resolution, negative_prompt=negative,
                    prompt=prompt, **{f"images.image_{i + 1}": im for i, im in enumerate(images)})
        return g.add("VAEDecode", vae=vae(), samples=g.add(
            "KSampler", model=model(), seed=seed_, steps=QWEN_STEPS, cfg=1.0, sampler_name="euler", scheduler="simple",
            positive=enc(0), negative=enc(1), denoise=1.0,
            latent_image=latent or g.add("EmptyLatentImage", width=768, height=1312, batch_size=1)())())

    def lora(name):
        return g.add("LoraLoaderModelOnly", lora_name=name, strength_model=1.0, model=base())

    def skeleton(im):
        return g.add("SDPoseDrawKeypoints", keypoints=g.add("SDPoseKeypointExtractor", model=sdpose(0), vae=sdpose(2),
                                                             image=im, batch_size=1)(),
                     draw_body=True, draw_hands=False, draw_face=True, draw_feet=False, stick_width=4,
                     face_point_size=3, score_threshold=0.3, draw_head=True)

    sdpose = g.add("CheckpointLoaderSimple", ckpt_name="sdpose_wholebody_fp16.safetensors")
    dressed = qwen(base, "The same person from <image1>, wearing exactly the same clothes and everything they wear, "
                         "now seen waist-up from the front three-quarter view, "
                         f"{FRONT[CAMERA].format(held=held_as[0], shade=held_as[1])}, taking a photo of something "
                         "across the street. Plain light grey background, soft daylight.", seed + 7,
                   [g.add("LoadImage", image=front)(0)])
    who = qwen(lora(HEAD_SWAP),
               "head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. "
               "remove the head from <image1> completely and replace it with the head from <image2>, strictly "
               "preserving the hair, eye color, nose structure from <image2>. copy the direction of the eye, head "
               "rotation, micro expressions from <image1>, high quality, sharp details, 4k", seed + 9,
               [dressed(), g.add("LoadImage", image=profile)(0)])
    sam = g.add("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    cam = g.add("SAM3_Detect", model=sam(0), image=who(), threshold=0.3, refine_iterations=2, individual_masks=False,
                conditioning=g.add("CLIPTextEncode", text=LENS_OF[CAMERA], clip=sam(1))())
    for k, im in (("revdressed", dressed), ("revwho", who)):
        g.add("SaveImage", images=im(), filename_prefix=f"{prefix}_{k}")
    g.add("SaveImage", images=skeleton(who())(), filename_prefix=prefix + "_revpose")
    g.add("SaveImage", images=g.add("MaskToImage", mask=cam(0))(), filename_prefix=prefix + "_revcam")
    return g


def defocused(g, post, width, height):
    """The real place behind a long-lens close-up: the post scaled to the frame, blurred hard (ImageBlur tops out
    at radius 31, so it is repeated)."""
    plate = g.add("ImageScale", image=g.add("LoadImage", image=post)(0), upscale_method="lanczos", width=width,
                  height=height, crop="center")
    for _ in range(3):
        plate = g.add("ImageBlur", image=plate(), blur_radius=31, sigma=10.0)
    return plate


def matted(g, images, plate, frames=1):
    """`images` (a batch) matted with BiRefNet (a real alpha) over `plate`, every frame."""
    matte = g.add("RemoveBackground", image=images, bg_removal_model=g.add(
        "LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")())
    behind = g.add("RepeatImageBatch", image=plate(), amount=frames) if frames > 1 else plate
    return g.add("ImageCompositeMasked", destination=behind(), source=images, x=0, y=0, resize_source=False,
                 mask=matte(0))


def reverse_film(still, pose, plan, seed, prefix, post=None):
    """H3 animates the reverse shot, the camera locked off. It continues the cut: the push-in ends on the placed
    keyframe, camera already up, so the pose is held there (the Fun ControlNet-Union on the same skeleton, every
    frame), and the shutter fires at the end. Rendered at REV_SCALE, for the push-in on the camera."""
    g, w, h = Graph(), round(W * REV_SCALE / 32) * 32, round(H * REV_SCALE / 32) * 32
    text = ("integrated_multimodal_description: A tight cinematic close-up, the camera locked off on a long lens. The "
            f"photographer, {local.CAM['hold']}, holds it steady, intent behind the camera, and presses the shutter "
            "at the very end. Shallow depth of field, the background soft."
            f"\noverall_soundscape: {plan['soundscape']}"
            "\nnon_diegetic_music: N/A")
    vae, avae = g.add("VAELoader", vae_name=H3["vae"]), g.add("VAELoader", vae_name=H3["audio_vae"])
    model = g.add("LoraLoaderModelOnly", lora_name=H3["lora"], strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=H3["unet"], weight_dtype="default")())
    model = g.add("MiniMaxH3FunControlNetApply", model=model(), vae=vae(), strength=1.0, start_percent=0.0,
                  end_percent=1.0, model_patch=g.add(
                      "ModelPatchLoader", name="minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors")(),
                  control_video=g.add("RepeatImageBatch", image=g.add("LoadImage", image=pose)(0),
                                      amount=REV_FRAMES)())
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=w, height=h, length=REV_FRAMES, vae=vae(),
                 first_frame=g.add("LoadImage", image=still)(0),
                 clip=g.add("CLIPLoader", clip_name=H3["clip"], type="minimax", device="default")())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    frames = g.add("VAEDecode", samples=out(0), vae=vae())()
    if post:  # H3 only pins the first frame's background: hold the defocused place behind every frame
        frames = matted(g, frames, defocused(g, post, w, h), REV_FRAMES)()
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=frames,
                                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())(),
          filename_prefix=prefix + "_reverse", format="mp4", codec="h264")
    return g


SHUTTER = Path(__file__).parent / "sfx" / "dslr_shutter.wav"  # a real DSLR (see sfx/CREDITS.md)


def punch_in(video, center, out):
    """The camera pushes in on the photographer's camera over the reverse shot's last PUNCH_TIME (eased), ending as
    the shutter fires: crops of the REV_SCALE render, so the end of the push stays sharp. `center`: the camera, in
    that render's pixels."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), f"{d}/%04d.png"], check=True)
        for i, f in enumerate(sorted(Path(d).glob("*.png"))):
            u = min(1.0, max(0.0, (i / FPS - (REV_KEEP - PUNCH_TIME)) / PUNCH_TIME))
            e = u * u * (3 - 2 * u)
            im = Image.open(f)
            cw, ch = im.width / (1 + (PUNCH - 1) * e), im.height / (1 + (PUNCH - 1) * e)
            cx = im.width / 2 + (center[0] - im.width / 2) * e
            cy = im.height / 2 + (center[1] - im.height / 2) * e
            x0, y0 = min(max(0.0, cx - cw / 2), im.width - cw), min(max(0.0, cy - ch / 2), im.height - ch)
            im.convert("RGB").resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch)).save(f)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{d}/%04d.png", "-i", str(video),
                        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14",
                        "-c:a", "copy", str(out)], check=True)


def cut(take, reverse, out, keep=REV_KEEP):
    """The take, then the reverse shot's first `keep` seconds (it starts where the take ends: the same pose), and
    the shutter on its last beat: a recorded shutter laid in like foley, not whatever H3 may or may not voice.
    The place keeps sounding across the cut: a stretch of the take's own ambience (from its middle, so nothing
    audibly repeats) runs under the reverse shot, the way a cut between two angles of one place is mixed."""
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   str(take)], capture_output=True, text=True, check=True).stdout)
    bed = max(0.0, length / 2 - keep / 2)
    click_at = keep - 0.25  # the shutter, then a beat of it before the reveal's flash
    stereo = "aresample=48000,aformat=channel_layouts=stereo"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(take), "-i", str(reverse), "-i", str(SHUTTER),
                    "-filter_complex",
                    f"[1:v]trim=0:{keep:.3f},setpts=PTS-STARTPTS[r];"
                    f"[0:a]{stereo},asplit=2[ta][tb];"
                    f"[tb]atrim={bed:.3f}:{bed + keep:.3f},asetpts=PTS-STARTPTS,afade=t=out:st={keep - 0.3:.3f}:d=0.3[amb];"
                    f"[1:a]atrim=0:{keep:.3f},asetpts=PTS-STARTPTS,{stereo}[ra];"
                    f"[2:a]adelay={int(click_at * 1000)}:all=1,apad,atrim=0:{keep:.3f}[c];"
                    "[ra][amb][c]amix=inputs=3:normalize=0[rc];"
                    "[0:v][ta][r][rc]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-c:a", "aac", str(out)], check=True)


def stage_cd(sheet, scene, mask, plan, seed, prefix):
    g = Graph()
    frames = g.add("GetVideoComponents", video=g.add("LoadVideo", file=scene)())
    src = g.add("ImageFromBatch", image=frames(0), batch_index=0, length=LTX_FRAMES)
    hole_img = g.add("ImageFromBatch", batch_index=0, length=LTX_FRAMES,
                     image=g.add("GetVideoComponents", video=g.add("LoadVideo", file=mask)())(0))
    hole = g.add("ImageToMask", channel="red", image=hole_img())
    sheet_img = g.add("LoadImage", image=sheet)

    def place(index, s_):
        img = klein(g, "Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the "
                       "foreground, seen from behind and slightly to the side, turned toward the subject, "
                       f"{local.CAM['hold']}, aimed at the subject, taking a photo. Same hair and clothes as the "
                       "sheet. Exactly one photographer. Change nothing else in ref_image_0. " + plan["photographer"]
                    + rows(geo.layout(camera=CAMERA)),
                    [g.add("ImageFromBatch", image=frames(0), batch_index=index, length=1)(), sheet_img(0)], s_)
        return img

    last = place(LTX_FRAMES - 1, seed + 1)
    if os.environ.get("TWO_PLACES"):  # the old way, for A/B tests: two placements, two different phones to morph between
        first = place(0, seed)
    else:  # one placement: the same photographer, phone and grip cut out of the last frame onto the first
        matte = g.add("RemoveBackground", image=last(), bg_removal_model=g.add(
            "LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")())  # an alpha matte
        cut = g.add("MaskComposite", destination=matte(0), operation="multiply", x=0, y=0,  # only inside their 3D box
                    source=g.add("ImageToMask", channel="red", image=g.add("ImageFromBatch", batch_index=0, length=1,
                                                                           image=hole_img())())())
        first = g.add("ImageCompositeMasked", destination=g.add("ImageFromBatch", image=frames(0), batch_index=0,
                                                                 length=1)(), source=last(), x=0, y=0,
                      resize_source=False, mask=cut())
    g.add("SaveImage", images=g.add("ImageBatch", image1=first(), image2=last())(), filename_prefix=prefix + "_placed")
    vae, avae = g.add("VAELoader", vae_name=LTX_VAE), g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    ic = g.add("LTXICLoRALoaderModelOnly", model=g.add("UNETLoader", unet_name=LTX, weight_dtype="default")(),
               lora_name=INPAINT, strength_model=1.0)
    text = (f"{plan['photographer']}, seen from behind in the foreground, {local.CAM['act']}. {plan['action']} "
            f"{plan['scene']}{QUIET}")
    cond = g.add("LTXVConditioning", frame_rate=float(FPS), positive=g.add("CLIPTextEncode", text=text, clip=clip())(),
                 negative=g.add("CLIPTextEncode", text=NEG, clip=clip())())
    lat = g.add("LTXVImgToVideoInplace", vae=vae(), image=first(), strength=1.0, bypass=False,
                latent=g.add("EmptyLTXVLatentVideo", width=W, height=H, length=LTX_FRAMES, batch_size=1)())
    guide = g.add("LTXAddVideoICLoRAGuideAdvanced", positive=cond(0), negative=cond(1), vae=vae(), latent=lat(),
                  image=g.add("LTXVInpaintPreprocess", images=src(), mask=hole())(), frame_idx=0, strength=1.0,
                  latent_downscale_factor=ic(1), crop="disabled", use_tiled_encode=False, tile_size=256,
                  tile_overlap=64, attention_strength=1.0, use_streaming=False, tile_frames=97)
    guide = g.add("LTXVAddGuide", positive=guide(0), negative=guide(1), vae=vae(), latent=guide(2), image=last(),
                  frame_idx=-1, strength=1.0)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=LTX_FRAMES, frame_rate=FPS, batch_size=1, audio_vae=avae())
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("CFGGuider", model=ic(0), positive=guide(0), negative=guide(1), cfg=1.0)(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(),
                latent_image=g.add("LTXVConcatAVLatent", video_latent=guide(2), audio_latent=audio())())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    video = g.add("LTXVCropGuides", positive=guide(0), negative=guide(1), latent=sep(0))(2)
    g.add("SaveVideo", video=g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=video, vae=vae())(),
                                   audio=frames(1))(), filename_prefix=prefix + "_final", format="mp4", codec="h264")
    return g


def reverse(od, tag, pu, su, plan, seed, raw):
    """The reverse shot, from the push-in's own photographer (its steps in `od`), cut after the push-in `raw`.
    -> the cut."""
    cutout(od / f"{tag}_openhi.png", od / f"{tag}_openmask.png", od / f"{tag}_revcut.png")
    cu = comfy.upload(str(od / f"{tag}_revcut.png"), f"momento_{tag}_revcut.png")
    front_view(od / f"{tag}_sheet.png", f"momento/{tag}", od / f"{tag}_revfront.png")
    files = run(reverse_still(pu, comfy.upload(str(od / f"{tag}_revfront.png"), f"momento_{tag}_revfront.png"),
                              held(cu, f"momento/{tag}"), seed, f"momento/{tag}"))
    for k in ("revdressed", "revwho", "revpose", "revcam"):
        fetch(files[k][0], od / f"{tag}_{k}.png")
    who, skel, cam = (Image.open(od / f"{tag}_{k}.png") for k in ("revwho", "revpose", "revcam"))
    box = framed(who.size, skel, cam)
    size = (round(W * REV_SCALE / 32) * 32, round(H * REV_SCALE / 32) * 32)
    for k, im in (("revkey", who), ("revposekey", skel), ("revcamkey", cam)):
        im.convert("RGB").resize(who.size).crop(box).resize(size, Image.LANCZOS).save(od / f"{tag}_{k}.png")
    ys, xs = np.nonzero(np.asarray(Image.open(od / f"{tag}_revcamkey.png").convert("L")) > 127)
    center = (xs.mean(), ys.mean()) if len(xs) else (size[0] / 2, size[1] / 3)
    files = run(reverse_film(comfy.upload(str(od / f"{tag}_revkey.png"), f"momento_{tag}_revkey.png"),
                             comfy.upload(str(od / f"{tag}_revposekey.png"), f"momento_{tag}_revposekey.png"),
                             plan, seed, f"momento/{tag}", post=su))
    fetch(files["reverse"][0], od / f"{tag}_reverse.mp4")
    punch_in(od / f"{tag}_reverse.mp4", center, od / f"{tag}_punched.mp4")
    cut(raw, od / f"{tag}_punched.mp4", od / f"{tag}_cut.mp4")
    return od / f"{tag}_cut.mp4"


def main(profile, post, out="out"):
    name, t = Path(post).stem, time.time()
    seed = int(os.environ.get("SEED") or random.randrange(2**31))
    tag = f"{name}_{os.environ.get('TAG', 'mol')}"
    saved = os.environ.get("PLAN_JSON")
    if saved and Path(saved).exists():
        plan = json.loads(Path(saved).read_text())
    else:
        plan = ask(PLAN, "Left: the profile picture. Right: the photo they posted. Plan the moment.", [profile, post])
        if saved:
            Path(saved).write_text(json.dumps(plan))
    od = Path(out) / "steps"
    od.mkdir(parents=True, exist_ok=True)
    pu, su = comfy.upload(profile, f"momento_{Path(profile).name}"), comfy.upload(post, f"momento_{Path(post).name}")
    opening = geo.MOVES[MOVE] if MOVE else None
    if not os.environ.get("LEVEL"):  # LEVEL=1 keeps the old level camera, for A/B tests
        framing(su, f"momento/{tag}")
    if opening:  # the camera moves: H3 films it between two keyframes holding the one placement
        opening = reach(su, f"momento/{tag}", opening)
        if opening is None:  # backing up shows what the photo doesn't have: push in from START instead
            opening, geo.START = geo.START, tuple(PUSH_TO * v for v in geo.START)
            framing(su, f"momento/{tag}", eyes=[opening, geo.START])
            print(f"the move can't back up inside the photo: pushing in from {opening} to {geo.START}", flush=True)
        y0, x0, y1, x1 = opening_layout(opening)[1]  # SAM's prompt: their 3D box, widened for raised arms and phone
        wide = 0.35 * (x1 - x0)
        box = dict(x=round(max(0, x0 - wide) * W / 500), y=0, width=round((min(1000, x1 + wide) - max(0, x0 - wide)) * W / 500),
                   height=round(y1 * H / 500))
        files = run(move_keys(pu, su, box, plan, seed, f"momento/{tag}", opening))
        for k in ("sheet", "openfill", "endfill"):
            fetch(files[k][0], od / f"{tag}_{k}.png")
        for i in range(CANDIDATES):
            for k in ("openhi", "openmask", "openpose"):
                fetch(files[f"{k}{i}"][0], od / f"{tag}_{k}{i}.png")
        chosen = pick(od, tag)
        if chosen is None:
            raise SystemExit(f"[{tag}] no placement faces into the scene; not rendering a take that would")
        for k in ("openhi", "openmask", "openpose"):
            Image.open(od / f"{tag}_{k}{chosen}.png").save(od / f"{tag}_{k}.png")
        compose(*(od / f"{tag}_{k}.png" for k in ("openfill", "endfill", "openhi", "openmask")), opening,
                od / f"{tag}_first.png", od / f"{tag}_last.png")
        files = run(move_film(*(comfy.upload(str(od / f"{tag}_{k}.png"), f"momento_{tag}_{k}.png")
                                for k in ("first", "last")), plan, seed, f"momento/{tag}"))
        raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
        fetch(files["raw"][0], raw)
        if REVERSE:  # cut to the photographer taking the picture; the push-in on their camera and the shutter end it
            raw = reverse(od, tag, pu, su, plan, seed, raw)
        reveal(str(raw), post, str(final), click=not REVERSE)
        print(f"[{tag}] ({time.time() - t:.0f}s) {final}", flush=True)
        return
    files = run(stage_ab(pu, su, plan, seed, f"momento/{tag}"))
    fetch(files["sheet"][0], od / f"{tag}_sheet.png")
    fetch(files["scene"][0], od / f"{tag}_scene.mp4")
    mask_video(od / f"{tag}_mask.mp4")
    files = run(stage_cd(comfy.upload(str(od / f"{tag}_sheet.png"), f"momento_{tag}_sheet.png"),
                         comfy.upload(str(od / f"{tag}_scene.mp4"), f"momento_{tag}_scene.mp4"),
                         comfy.upload(str(od / f"{tag}_mask.mp4"), f"momento_{tag}_mask.mp4"), plan, seed,
                         f"momento/{tag}"))
    raw, final = od / f"{tag}_raw.mp4", Path(out) / f"{tag}.mp4"
    fetch(files["final"][0], raw)
    reveal(str(raw), post, str(final))
    print(f"[{tag}] ({time.time() - t:.0f}s) {final}", flush=True)


if __name__ == "__main__":
    profile, *posts = sys.argv[1:]
    for p in posts:
        main(profile, p)
