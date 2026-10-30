# Momento

**Type an Instagram name. Tap a post. Watch the moment the photo was taken.**

Live: **https://9ol.es/momento-demo/** · Demo video: **https://9ol.es/momento-video**

Momento opens a public Instagram profile and shows its 12 most recent posts as a grid. Tap one and it films the
moment that photo was taken, from a bystander's point of view. The person from the profile picture is there with
their phone up, in the same place, under the same light, with whatever the post cropped out. Then the video arrives
at the post itself. The soundtrack is written for each photo: a punk show at night gets punk, a quiet park gets
something gentle.

Every run is different: a new seed, and a camera that is either over the photographer's shoulder or out in front
where you can see their face. If the profile picture doesn't show a person, an anthropomorphic rabbit takes the photo.

Built for Comfy's **Open Source Higgsfield** build night. It's one ComfyUI graph. It runs as a Comfy App and behind
its own page on the same ComfyUI server, and it uses the Comfy API for the two models that aren't open weights.

## How it works

```
profile picture ─┐
                 ├─ Gemini 3.1 Pro (Comfy API partner node) looks at both and plans the moment:
the post ────────┘   the bystander's view of the scene, and the motion and sound of the shot being taken
                          │
                          ▼
                 Momento Plan: encodes the plan for both models, and passes Gemini's music prompt on
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
 FLUX 3 Klein builds the first frame    LTX-2.5 animates it with native audio:
 from the post's objects (ref 0) and    the person takes the shot; ambient sound
 the person (ref 1), from a new     ──▶ and a lens shutter, no speech. Over the
 camera position                        shoulder, the post is the LAST keyframe,
                                        so the camera ends on what the phone sees
                                             │
                                             ▼
 MiniMax Music 3 (Comfy Cloud) ────▶ Momento Reveal: mixes the soundtrack under the moment,
 scores it from Gemini's prompt      white shutter flash, then the real post
```

By default everything runs at the lowest resolution: 288×512 vertical, 4 seconds. That takes about 2.5 minutes on an
RTX 4090.

## Setup

You need ComfyUI v0.43 or newer, a GPU with 24 GB, and a Comfy account for the Gemini and MiniMax Music 3 nodes.

**1. Install the node pack**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kristopolous/comfyui-momento
```

It has no Python dependencies beyond ComfyUI itself.

**2. Download the models.** All of them are public on Hugging Face:

```bash
cd ComfyUI/models
K=https://huggingface.co/ml-resources/flux-3-image-klein-comfy/resolve/main
L=https://huggingface.co/Lightricks/LTX-2.5/resolve/main
wget -P diffusion_models $K/diffusion_models/flux3-image-klein-int8-convrot.safetensors
wget -P text_encoders    $K/text_encoders/qwen3vl_4b_int8_convrot.safetensors
wget -P vae              $K/vae/flux2-vae.safetensors
wget -P diffusion_models $L/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
wget -P text_encoders    $L/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
wget -P vae              $L/vae/ltx-2.5-video-vae-bf16.safetensors
wget -P vae              $L/vae/ltx-2.5-audio-vae-bf16.safetensors
```

**3. Sign in to Comfy.** In ComfyUI, sign in with your Comfy account, or use an API key from
[platform.comfy.org](https://platform.comfy.org). Gemini and MiniMax Music 3 run through the Comfy API and use your credits.

## Run

**The page:** with the node pack installed, open `http://<comfy-host>:8188/momento`. Type an Instagram name; the
server opens the public profile in headless Chromium (Instagram doesn't answer plain server requests), saves the
profile picture and posts, and shows the grid. Runs use the server's Comfy API key: set `COMFY_API_KEY`, or put it in
a `.env` file next to the ComfyUI folder. It needs `chromium` or `google-chrome` on the server.

**In ComfyUI:** open `workflows/momento.json` (App Mode), pick a profile picture and a photo, and press Run. The output is saved to `output/momento/`.

**From a script:**

```bash
COMFY_URL=http://<gpu-host>:8188 COMFY_API_KEY=comfyui-... python scripts/run.py [--angle behind|face] profile.jpg post1.jpg post2.jpg
```

To rebuild the graph, run `python scripts/graph.py > workflows/momento_api.json`.

Use photos the person actually took, with nothing in front of the camera except the scene. Posts where they appear
themselves don't fit the idea.

## The full pipeline: geometry, not prompts

The page above runs the single-graph version. The demo video's shots come from a stepped pipeline where the
relationship between the photographer and the scene is computed, not described:

1. **Plan.** A vision LLM looks at the profile picture and the post: the scene, the action, the photographer, the
   soundscape. This is Gemini 3.1 Pro on Comfy Cloud, or any local Ollama-API vision model.
2. **Character sheet.** FLUX 3 draws the account owner as a turnaround (front, side, back, camera raised). If the
   profile picture isn't a person, it draws an anonymous person instead.
3. **The place in 3D.** Depth Anything 3 (metric) rebuilds the post as a mesh. The post's camera is where the
   photographer stood. `RenderMesh` renders the bystander's view from behind them, and FLUX 3 fills the gaps.
4. **Placement.** FLUX 3 puts the photographer from the sheet into that view. The layout boxes are projected from
   their 3D position, so they face the subject with the phone or camera aimed at it.
5. **Film it.** MiniMax H3 films the seconds before the instant, ending on that frame. Its `overall_soundscape`
   prompt structure keeps it free of generated speech. Camera moves (dolly in, orbit, crane down) start from a
   second camera position in the same 3D scene.
6. **Review.** The LLM watches each take and the best one is kept. Shot-type routing sends close-ups (a laptop on a
   desk) through outpainting instead of geometry.

Options: `CAMERA=smartphone|dslr` sets what the photographer holds (also a toggle on the page). `FORCE_MOVE=1`
always uses a camera move. `RES=576x1024` sets the size on a bigger GPU.

    python scripts/moment.py profile.jpg post.jpg ...      # Comfy Cloud: FLUX 3 + H3 + LTX-2.5 inpaint, Gemini
    COMFY_URL=http://<gpu>:8188 PLANNER_URL=http://<ollama>:11434/api/chat PLANNER_MODEL=<vision model> \
        python scripts/local.py --hires profile.jpg post.jpg ...   # your GPU: FLUX 3 Klein + H3, open weights

`scripts/demo.py` cuts the demo video. It shows marching ants on the real profile, scene text from each post's
location tag and date, the narration, and the music.

## License

MIT. See [LICENSE](LICENSE).
