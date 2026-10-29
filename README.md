# Momento

**Give it a profile picture and a photo someone posted. Get back a video of the moment they took it.**

Momento rebuilds the scene from a bystander's point of view. Everything in the post is still there, now seen from a
few meters behind the photographer, along with whatever the post cropped out. The person from the profile picture
raises their phone, frames the shot and takes it. Soft jazz plays under the sounds of the place, and a lens shutter
clicks as the photo is taken. Then the video cuts to the real post.

If the profile picture doesn't show a person, an anthropomorphic rabbit takes the photo instead.

Built for Comfy's **Open Source Higgsfield** build night. It's one ComfyUI graph, it runs as an App, and it uses the
Comfy API for the model that plans the shot.

## How it works

```
profile picture ─┐
                 ├─ Gemini 3.1 Pro (Comfy API partner node) looks at both and plans the moment:
the post ────────┘   the bystander's view of the scene, and the motion and sound of the shot being taken
                          │
                          ▼
                 Momento Plan: encodes the plan for both models
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
 FLUX 3 Klein builds the frame from     LTX-2.5 animates it with native audio:
 the post's objects (ref 0) and the     the person takes the shot; ambient sound,
 person (ref 1), from a new camera  ──▶ soft jazz, a lens shutter. No speech.
 position, not the post as a start frame
                                             │
                                             ▼
                              Momento Reveal: white shutter flash, then the real post
```

By default everything runs at the lowest resolution: 288×512 vertical, 4 seconds. That takes about 2.5 minutes on an
RTX 4090.

## Setup

You need ComfyUI v0.43 or newer, a GPU with 24 GB, and a Comfy account for the Gemini partner node.

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
[platform.comfy.org](https://platform.comfy.org). Gemini runs through the Comfy API and uses your credits.

## Run

**In ComfyUI:** open `workflows/momento_api.json`, upload a profile picture to "Profile picture" and a photo to
"The post", and press Run. The output is saved to `output/momento/`.

**From a script:**

```bash
COMFY_URL=http://<gpu-host>:8188 COMFY_API_KEY=comfyui-... python scripts/run.py profile.jpg post1.jpg post2.jpg
```

To rebuild the graph, run `python scripts/graph.py > workflows/momento_api.json`.

Use photos the person actually took, with nothing in front of the camera except the scene. Posts where they appear
themselves don't fit the idea.

## License

MIT. See [LICENSE](LICENSE).
