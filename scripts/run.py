"""Run the app graph on a profile picture and posts, like pressing Run in the app.

    COMFY_URL=http://<gpu-host>:8188 python scripts/run.py profile.jpg post1.jpg [post2.jpg ...]
The Gemini node needs a Comfy API key: COMFY_API_KEY in the environment or in .env.
"""
import json
import os
import random
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comfyui_momento"))  # the pack's graph.py, not this folder's
from graph import momento  # noqa: E402


def key():
    if os.environ.get("COMFY_API_KEY"):
        return os.environ["COMFY_API_KEY"]
    for env in (Path.cwd() / ".env", Path.home() / "hacks/comfy/.env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("COMFY_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("Set COMFY_API_KEY")


TAKES = 3  # the critic can send a take back this many times


def submit(profile, post, out):
    pic, img = comfy.upload(profile, f"momento_profile_{Path(profile).stem}.png"), comfy.upload(post, f"momento_{Path(post).stem}.png")
    for take in range(1, TAKES + 1):
        verdict = shoot(pic, img, Path(post).stem, out, seed=random.randrange(2**31), take=take)
        if verdict.get("pass", True):
            return
        print(f"[{Path(post).stem}] take {take} rejected: {verdict.get('problem')}", flush=True)


def shoot(pic, img, name, out, seed, take):
    g = momento(pic, img, angle=ANGLE, seed=seed, prefix=f"momento/{name}_{ANGLE}")
    body = json.dumps({"prompt": g, "client_id": comfy.CLIENT_ID, "extra_data": {"api_key_comfy_org": key()}}).encode()
    res = json.loads(comfy._req("/prompt", body, {"Content-Type": "application/json"}))
    if res.get("node_errors"):
        raise RuntimeError(json.dumps(res["node_errors"])[:3000])
    pid, t = res["prompt_id"], time.time()
    while not (h := json.loads(comfy._req(f"/history/{pid}")).get(pid)) or h["status"].get("completed") is None:
        time.sleep(2)
    if h["status"]["status_str"] == "error":
        raise RuntimeError(json.dumps([m for m in h["status"]["messages"] if m[0] == "execution_error"])[:3000])
    verdict = {}
    for node in h["outputs"].values():
        for text in node.get("text", []):
            print(f"[{name}] {text}\n", flush=True)
            if text.startswith("REVIEW "):
                verdict = json.loads(text[7:])
        for f in node.get("images", []) + node.get("videos", []):
            q = urllib.parse.urlencode({k: f[k] for k in ("filename", "subfolder", "type")})
            dest = Path(out) / f"{name}_{ANGLE}_take{take}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(comfy._req(f"/view?{q}"))
            print(f"[{name}] {dest}  ({time.time() - t:.0f}s)", flush=True)
    return verdict


if __name__ == "__main__":
    args = sys.argv[1:]
    ANGLE = "behind"
    if "--angle" in args:
        i = args.index("--angle")
        ANGLE = args[i + 1]
        del args[i:i + 2]
    profile, *posts = args
    with ThreadPoolExecutor(len(posts)) as pool:  # queued together; the server runs them one after another
        list(pool.map(lambda p: submit(profile, p, "out"), posts))
