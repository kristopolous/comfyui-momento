"""The page's pipeline: geometry, not prompts (the same steps as scripts/local.py, inside the node pack).

A vision LLM plans the moment (outside the graph); then one graph:
  FLUX 3 Klein draws a character sheet of the account owner -> Depth Anything 3 rebuilds the post as a 3D place
  (the post's camera is where the photographer stood) -> RenderMesh renders the bystander's view, and the opening
  view of a camera move -> Klein fills them in and places the photographer at boxes projected from 3D -> MiniMax H3
  films the move into the instant -> Momento Reveal cuts to the real post. The planner then reviews the take.
"""
import json
import math

try:
    from .graph import Graph
except ImportError:  # imported as a plain module
    from graph import Graph

KLEIN, KLEIN_TE, FLUX_VAE = ("flux3-image-klein-int8-convrot.safetensors", "qwen3vl_4b_int8_convrot.safetensors",
                             "flux2-vae.safetensors")
H3_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
H3_VAE, H3_AUDIO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
H3_UNETS = ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors")
H3_FRAMES, FPS = 107, 24

PLAN = """Left image: someone's profile picture. Right image: a photo they posted. Reply with JSON only:
{"scene": "<one sentence: the place, the light, and the subject of the photo (who or what, what they wear)>",
 "action": "<one sentence: what the subject does over four seconds, ending in exactly the pose or instant the photo
   froze. Active, never posing.>",
 "photographer": "<one short phrase: the person in the profile picture (or, if it isn't a person, an anonymous
   everyday person whose look fits the place): hair, build, and clothes that fit the place of the photo>",
 "soundscape": "<1-4 sentences of the place's real sound only: ambient sound, physical action sounds and non-verbal
   human sounds, ending with the click of the camera shutter. No speech.>",
 "closeup": <true if the photo is a flat or near subject filling the frame (an object on a table, a laptop lid, food,
   a document), false if it is a place with depth (a street, a stage, a building, a landscape)>}"""

CRITIC = """These are frames, in order, from a generated video of a photographer taking a photo, seen by a bystander:
the photographer is in the foreground (often from behind), aiming a phone or camera at the scene. The LAST image is the
real photo they took. The video is NOT supposed to end on that photo: it ends on the bystander's view of the same
scene, photographer included, and then cuts to the photo. Judge the video frames as a short film of that moment.
Reject only for failures a viewer would notice at a glance: a person melting, morphing, multiplying or popping in and
out; extra limbs; hands and camera fusing into a blob; the subject frozen instead of moving; the scene not being the
real photo's place and subject; text or watermarks. Score how good a take it is overall. Reply with JSON only:
{"pass": true or false, "score": 1-10, "problem": "<the worst problem in one short sentence, or empty>"}"""

# the bystander's camera, in the post's 3D space (metric; the photographer stands at the origin, facing -z)
LENS, TARGET, START = 62.0, (0.0, -0.1, -4.0), (-0.4, 0.1, 1.7)


def orbit(eye, degrees):
    a = math.radians(degrees)
    x, y, z = eye
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


# camera moves: where the take starts; it always ends at START, behind the photographer
MOVES = {"dolly in": (-0.55, 0.15, 3.0), "orbit": orbit(START, 28), "crane down": (-0.35, 1.3, 2.1)}
MOVE_WORDS = {"dolly in": "a slow dolly in from further back", "orbit": "a slow arc around the photographer",
              "crane down": "a slow crane down from above"}

# what the photographer holds (the page's Phone / DSLR toggle)
CAMERAS = {
    "smartphone": dict(hold="holding a phone up in both hands", sheet="hold a phone up at eye level in both hands",
                       act="holds the phone up, aimed at the subject, and taps the shutter at the end",
                       box="phone: held up in both hands, aimed at the subject"),
    "dslr": dict(hold="holding a DSLR camera with a lens up to their eye", sheet="hold a DSLR camera with a zoom lens "
                 "up to their eye", act="raises the DSLR to their eye, frames the subject through the viewfinder and "
                 "presses the shutter at the end", box="camera: a DSLR with a lens, held up to the eye, aimed at the "
                 "subject"),
}


def project(point, eye, width, height, target=TARGET, fov=LENS):
    """World point -> pixel, for RenderMesh's look-at camera (Y up)."""
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    r = [-f[2], 0.0, f[0]]  # f x up
    n = math.sqrt(sum(v * v for v in r))
    r = [v / n for v in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    p = [a - b for a, b in zip(point, eye)]
    z = sum(a * b for a, b in zip(p, f))
    k = 1 / math.tan(math.radians(fov) / 2)
    x = sum(a * b for a, b in zip(p, r)) / z * k * height / width
    y = sum(a * b for a, b in zip(p, u)) / z * k
    return (x + 1) / 2 * width, (1 - y) / 2 * height


def layout(eye, camera, width, height):
    """FLUX 3 layout rows: where the photographer and their camera are, seen from `eye`."""
    def box(points):
        px = [project(p, eye, width, height) for p in points]
        clip = lambda v, m: max(0, min(1000, round(v / m * 1000)))  # noqa: E731
        return [clip(min(p[1] for p in px), height), clip(min(p[0] for p in px), width),
                clip(max(p[1] for p in px), height), clip(max(p[0] for p in px), width)]
    body = [(x, y, z) for x in (-0.24, 0.24) for y in (0.14, -1.6) for z in (-0.1, 0.15)]
    held = ([(x, y, z) for x in (-0.1, 0.1) for y in (-0.06, 0.08) for z in (-0.12, -0.3)] if camera == "dslr" else
            [(x, y, -0.42) for x in (-0.08, 0.12) for y in (-0.02, 0.16)])
    return [{"id": "photographer", "bbox": box(body), "desc": "the person from the character sheet, seen from behind "
                                                              "and a little from the side, facing into the scene"},
            {"id": CAMERAS[camera]["box"].split(":")[0], "bbox": box(held), "desc": CAMERAS[camera]["box"]}]


def build(profile, post, plan, seed, prefix, move, camera="smartphone", h3_unet=H3_UNETS[0], width=448, height=768,
          megapixels=0.4):
    """The whole take as one API-format graph. plan: the planner's JSON; move: a key of MOVES (or None).
    Open weights only; the server reviews the finished take with the planner model (CRITIC)."""
    g = Graph()
    cam = CAMERAS[camera]
    who = g.add("LoadImage", "Profile picture", image=profile)
    shot = g.add("LoadImage", "The post", image=post)
    kclip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    kvae = g.add("VAELoader", vae_name=FLUX_VAE)

    def klein(prompt, refs, s, w=width, h=height):
        model = g.add("ModelSamplingFlux", model=g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")(),
                      max_shift=2.77, base_shift=2.77, width=w, height=h)
        refd = g.add("ReferenceMultiLatent", vae=kvae(), resolution="resize", **{"resolution.megapixels": megapixels},
                     positive=g.add("CLIPTextEncode", text=prompt, clip=kclip())(),
                     negative=g.add("CLIPTextEncode", text="", clip=kclip())(),
                     **{f"images.image{i}": r for i, r in enumerate(refs)})
        out = g.add("KSampler", model=model(), seed=s, steps=16, cfg=1.0, sampler_name="euler", scheduler="simple",
                    positive=refd(0), negative=refd(1), denoise=1.0,
                    latent_image=g.add("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)())
        return g.add("VAEDecode", samples=out(), vae=kvae())

    sheet = klein("Character turnaround sheet, four full-body views side by side on a plain light grey studio "
                  "background: front, three-quarter front, side profile, and back. Same face, hair, build and clothes "
                  f"in every view. In the side and back views they {cam['sheet']}, taking a photo. Photographic, even "
                  "lighting. If ref_image_0 shows a person, it is them, exactly; if not, the character is the "
                  "anonymous everyday person described here. The character: " + plan["photographer"],
                  [who(0)], seed, w=768, h=448)
    fill = ("ref_image_0 is a bystander's view of a real place, with black gaps where the camera sees around things: "
            "fill them seamlessly with more of the same place, same light, same lens. Keep everything else exactly as "
            "it is. No people in the foreground. " + plan["scene"])

    def place(img, eye, s):
        return klein("Add the person from ref_image_1's character sheet into ref_image_0 as the photographer: in the "
                     f"foreground, seen from behind and slightly to the side, turned toward the subject, {cam['hold']}, "
                     "aimed at the subject, taking a photo. Same hair and clothes as the sheet. Exactly one "
                     "photographer, no other copies of them. Change nothing else in ref_image_0. "
                     + plan["photographer"] + " " + json.dumps(layout(eye, camera, width, height)), [img, sheet()], s)

    if plan.get("closeup"):  # no depth worth rebuilding: zoom out from the post itself, no camera move
        move = None
        instant = klein("Zoom out from ref_image_0: the same instant seen by a bystander a couple of steps back. "
                        "ref_image_0's view, unchanged, fills the middle of the frame, with the surroundings it cropped "
                        "out all around it, same light and lens. No people in the foreground. " + plan["scene"],
                        [shot(0)], seed)
    else:  # the post as a 3D place
        geom = g.add("DA3Inference", image=shot(0), resolution=504, resize_method="upper_bound_resize", mode="mono",
                     da3_model=g.add("LoadDA3Model", model_name="depth_anything_3_metric_large.safetensors",
                                     weight_dtype="default")())
        mesh = g.add("DA3GeometryToMesh", da3_geometry=geom(), batch_index=0, decimation=1,
                     discontinuity_threshold=0.04, confidence_threshold=0.1, use_sky_mask=True, texture=True)

        def view(eye):
            c = g.add("CreateCameraInfo", mode="look_at", **{f"mode.position_{a}": v for a, v in zip("xyz", eye)},
                      target_x=TARGET[0], target_y=TARGET[1], target_z=TARGET[2], roll=0.0, fov=LENS, zoom=1.0,
                      camera_type="perspective")
            return g.add("RenderMesh", mesh=mesh(), mode="texture", width=width, height=height,
                         background="#000000", camera_info=c())(0)

        instant = klein(fill, [view(START)], seed)
    if plan.get("closeup"):
        view = None
    placed = place(instant(), START, seed + 1)
    opening = place(klein(fill, [view(MOVES[move])], seed + 2)(), MOVES[move], seed + 3) if move else None

    # MiniMax H3: the camera move into the instant; soundscape only (no tagged speakers, so nobody talks)
    model = g.add("LoraLoaderModelOnly", lora_name=H3_LORA, strength_model=1.0,
                  model=g.add("UNETLoader", unet_name=h3_unet, weight_dtype="default")())
    vae = g.add("VAELoader", vae_name=H3_VAE)
    avae = g.add("VAELoader", vae_name=H3_AUDIO_VAE)
    camera_text = (f"The camera makes {MOVE_WORDS[move]}, moving through the real space of <Picture 1> to "
                   f"<Picture 2>." if move else "Handheld bystander camera, nearly still.")
    text = (f"integrated_multimodal_description: {camera_text} The photographer in the foreground, seen from behind, "
            f"{cam['act']}. There is exactly one photographer. The subject is in motion and ends exactly in the moment "
            f"of the last frame: {plan['action']} {plan['scene']}\noverall_soundscape: {plan['soundscape']}"
            "\nnon_diegetic_music: N/A")
    ends = {"first_frame": opening(), "last_frame": placed()} if opening else {"last_frame": placed()}
    cond = g.add("MiniMaxH3ImageToVideo", prompt=text, width=width, height=height, length=H3_FRAMES, vae=vae(),
                 clip=g.add("CLIPLoader", clip_name=H3_CLIP, type="minimax", device="default")(), **ends)
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(),
                guider=g.add("BasicGuider", model=model(), conditioning=cond(0))(),
                sampler=g.add("KSamplerSelect", sampler_name="res_multistep")(),
                sigmas=g.add("BasicScheduler", model=model(), scheduler="simple", steps=8, denoise=1.0)(),
                latent_image=cond(1))
    moment = g.add("CreateVideo", fps=float(FPS), images=g.add("VAEDecode", samples=out(0), vae=vae())(),
                   audio=g.add("VAEDecodeAudio", samples=out(0), vae=avae())())
    final = g.add("MomentoReveal", moment=moment(), post=shot(0), hold=1.8, flash=0.25, music_level=0.0)
    g.add("SaveVideo", "Momento", video=final(), filename_prefix=prefix, format="mp4", **{"format.codec": "h264"})
    return g
