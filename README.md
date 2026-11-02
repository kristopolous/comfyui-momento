# Momento

**Type an Instagram name. Tap a post. Watch the moment the photo was taken.**

Live: **https://9ol.es/momento-demo/** · Demo video: **https://9ol.es/momento-video**

![Chris Burkard's real Instagram profile zooms into his helicopter post, then the moment he took it: he raises a DSLR as the surfers walk off from the Mi-8, the shutter fires, and it cuts to the real photo](docs/hero.gif)

Momento opens a public Instagram profile and shows its most recent posts as a grid. Tap one and it films the moment
that photo was taken, from a bystander's point of view. The person in the profile picture stands where the camera
stood, phone or camera raised. The subject moves into the exact instant the photo froze, the shutter fires, and the
video cuts to the real post.

Built for Comfy's **Open Source Higgsfield** build night, entirely in ComfyUI: FLUX 3, MiniMax H3, LTX-2.5, Depth
Anything 3 and Gemini or a local vision model. It runs on Comfy Cloud or on your own GPU, with open weights only.

## What it makes

Each strip shows three frames of the generated moment, then the real post.

![Chris Burkard, Kamchatka: an orbit around him as he raises a DSLR at the surfers](docs/shot_burkard_kamchatka.jpg)
![Rachel Bennett: The Last Dinner Party moving into the pose she caught](docs/shot_bennett_neon.jpg)
![Joel Meyerowitz, St. Louis, 1977](docs/shot_meyerowitz_stlouis.jpg)
![@kristopolous at L.A. City Hall as the bus pulls in](docs/shot_kristopolous_la.jpg)

## How it works: geometry, not prompts

Where the photographer stands, and which way they face, isn't left to a prompt. It's computed from the post itself.

![The profile picture, the character sheet, the bystander's view rebuilt in 3D, the photographer placed](docs/steps.jpg)

1. **Plan.** A vision LLM looks at the profile picture and the post and describes the scene, the action, the
   photographer and the place's sound. This is Gemini 3.1 Pro on Comfy Cloud, or any Ollama-API vision model locally.
2. **Character sheet.** FLUX 3 draws the account owner as a turnaround (front, side, back, camera raised), so they
   stay the same person from every angle. If the profile picture isn't a person, it draws an anonymous person.
3. **The place in 3D.** Depth Anything 3 (metric) rebuilds the post as a mesh. The post's camera is where the
   photographer stood. `RenderMesh` renders the view of a bystander standing behind them, and FLUX 3 fills the gaps.
4. **Placement.** FLUX 3 puts the person from the sheet into that view. Its layout boxes are projected from their
   real 3D position, so they face the subject with the phone or camera aimed at it.
5. **Film it.** MiniMax H3 films the seconds before the instant, ending on that frame, and the subject moves into the
   photo's pose. Camera moves (dolly in, orbit, crane down) start from a second camera position in the same 3D scene.
   H3's `overall_soundscape` prompt structure gives real ambient sound and a shutter click, with no generated speech.
6. **Review.** The LLM watches every take and only the best one is kept. Close-ups (a laptop on a desk) have no
   useful depth, so they're routed through FLUX 3 outpainting instead.

## Setup

You need **ComfyUI v0.43 or newer**. There are three ways to run it:

| | What runs where | Needs |
|---|---|---|
| **A. Your own GPU** (open weights) | everything on your ComfyUI | a 24 GB GPU, about 65 GB of models, a local vision LLM |
| **B. Comfy Cloud** | everything on Comfy Cloud | a paid Comfy Cloud plan and API key |
| **C. The page / Comfy App** | the single-graph app on your ComfyUI | a 24 GB GPU, a Comfy account for Gemini and MiniMax Music 3 |

### 1. Get the code

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kristopolous/comfyui-momento
python -m pip install -r comfyui-momento/scripts/requirements.txt   # in ComfyUI's venv; only for scripts/
```

The node pack has no Python dependencies beyond ComfyUI. The page also needs `chromium` (or `google-chrome`) on the
server to read public profiles: `sudo apt install chromium`.

### 2. Download the models

FLUX 3 Klein (used by A and C) and LTX-2.5 (used by C) are gated: accept their licenses at
https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy and https://huggingface.co/Lightricks/LTX-2.5,
then log in with `hf auth login`. Run these from the ComfyUI folder's parent (`cd ../..` after step 1).

**For A (your own GPU).**

```bash
cd ComfyUI/models
K=https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main
H=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main
D=https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main
TOKEN=$(cat ~/.cache/huggingface/token)
wget --header="Authorization: Bearer $TOKEN" -P diffusion_models $K/diffusion_models/flux3-image-klein-int8-convrot.safetensors
wget --header="Authorization: Bearer $TOKEN" -P text_encoders    $K/text_encoders/qwen3vl_4b_int8_convrot.safetensors
wget --header="Authorization: Bearer $TOKEN" -P vae              $K/vae/flux2-vae.safetensors
wget -P diffusion_models    $H/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
wget -P text_encoders       $H/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
wget -P vae                 $H/vae/minimax_h3_video_vae_int8_convrot.safetensors
wget -P vae                 $H/vae/minimax_h3_audio_vae_fp32.safetensors
wget -P loras               $H/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
wget -P geometry_estimation $D/geometry_estimation/depth_anything_3_metric_large.safetensors
```

For the planner and critic, run any vision chat model that speaks the Ollama API, for example `ollama pull qwen3-vl`.

**For C (the page).** C needs only the three FLUX 3 Klein files from A, plus LTX-2.5. It doesn't need H3 or Depth
Anything 3; Gemini and MiniMax Music 3 run through the Comfy API.

```bash
cd ComfyUI/models
L=https://huggingface.co/Lightricks/LTX-2.5/resolve/main
TOKEN=$(cat ~/.cache/huggingface/token)
wget --header="Authorization: Bearer $TOKEN" -P diffusion_models $L/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
wget --header="Authorization: Bearer $TOKEN" -P text_encoders    $L/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
wget --header="Authorization: Bearer $TOKEN" -P vae              $L/vae/ltx-2.5-video-vae-bf16.safetensors
wget --header="Authorization: Bearer $TOKEN" -P vae              $L/vae/ltx-2.5-audio-vae-bf16.safetensors
```

**For B (Comfy Cloud).** Nothing to download.

### 3. Keys

For **B** and **C**, set `COMFY_API_KEY` (from [platform.comfy.org](https://platform.comfy.org)) in the environment,
or put `COMFY_API_KEY=...` in a `.env` file next to the ComfyUI folder. **A** needs no keys.

## Run

**A. Your own GPU:**

```bash
COMFY_URL=http://<gpu-host>:8188 \
PLANNER_URL=http://<ollama-host>:11434/api/chat PLANNER_MODEL=qwen3-vl \
CAMERA=dslr \
python scripts/local.py --hires profile.jpg post1.jpg post2.jpg
```

The output for each post is `out/<post>_local<N>.mp4`, with every step's images in `out/steps/`.

- `CAMERA=smartphone|dslr` sets what the photographer holds.
- Every take has a camera move through the 3D scene (dolly in, orbit or crane down). `ALLOW_STILL=1` also allows a
  still camera.
- `TAKES=2` sets how many takes the critic picks from.
- `PULL_BACK=1.5` (the default) is how far back the bystander stands, as a multiple of the old 1.7 m: the photographer is head to toe in the frame with the place around them. Where the camera sees past the post, the gaps render magenta and FLUX 3 Klein outpaints them.
- `FLEX=1` films with [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) (Comfy's 6-step recipe) instead of the 8-step turbo LoRA. Put `pika_h3_flex_fl2va_v1_bf16.safetensors` from Comfy-Org/MiniMax-H3 in `loras/`.
- `SEED=` and `MOVE="dolly in"|orbit|"crane down"` repeat a take exactly, for A/B tests.
- `--hires` gives 448×768, and `RES=576x1024` goes bigger on a 48 GB GPU.
- If only another machine can reach the planner, `PLANNER=<ssh host>` calls it through that machine.

**B. Comfy Cloud:**

```bash
COMFY_API_KEY=... python scripts/moment.py [--hires] profile.jpg post1.jpg post2.jpg
```

This uses Gemini, FLUX 3 Image, MiniMax H3 and LTX-2.5's in/outpainting IC-LoRA. It runs three takes per post side
by side, and Gemini keeps the best one.

**C. The page:** open `http://<comfy-host>:8188/momento`, type an Instagram name, tap a post, choose
📱 Phone or 📷 DSLR, ▯ portrait (the default) or ▭ landscape, and press **Make the moment**. With `PLANNER_URL` and
`PLANNER_MODEL` set (any Ollama-API vision model, e.g. Qwen3-VL), that model plans the moment, reviews each take
against the real post, and reshoots rejected ones; the best-scoring take wins.
In ComfyUI itself, open `workflows/momento.json` (App Mode).

![The page: type a name, the public profile opens, tap a post, choose phone or DSLR](docs/app.png)

Use photos the person actually took. Posts they appear in themselves don't fit the idea.

`scripts/demo.py` cuts the demo video: marching ants on the real profile, scene text from each post's location and
date, then narration and music.

## License

MIT. See [LICENSE](LICENSE).
