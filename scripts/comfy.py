"""Minimal client for a ComfyUI server (API-format prompts).

Set COMFY_URL for a remote server, or tunnel to it:
    ssh -f -N -L 8188:127.0.0.1:8188 <gpu-host>
"""
import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid
from pathlib import Path

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
CLIENT_ID = str(uuid.uuid4())


def _req(path, data=None, headers=None, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL + path, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except (ConnectionError, urllib.error.URLError):
            if data is not None or i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def upload(path, name=None):
    """Upload an image into ComfyUI's input folder; returns the name LoadImage wants."""
    path = Path(path)
    name = name or path.name
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + \
           (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
            f"\r\n--{boundary}--\r\n").encode()
    out = json.loads(_req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    return out["name"]


def run(prompt, out_dir, timeout=1800):
    """Queue a prompt, wait for it, download every output file into out_dir. Returns (paths, seconds)."""
    t = time.time()
    body = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode()
    res = json.loads(_req("/prompt", body, {"Content-Type": "application/json"}))
    if res.get("node_errors"):
        raise RuntimeError(json.dumps(res["node_errors"], indent=2))
    pid = res["prompt_id"]
    while True:
        hist = json.loads(_req(f"/history/{pid}")).get(pid)
        if hist and hist.get("status", {}).get("completed") is not None:
            break
        if time.time() - t > timeout:
            raise TimeoutError(pid)
        time.sleep(1)
    status = hist["status"]
    if status.get("status_str") == "error":
        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError(json.dumps(msgs, indent=2)[:4000])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for node in hist["outputs"].values():
        for kind in ("images", "videos", "gifs"):
            for f in node.get(kind, []):
                q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
                dest = out_dir / f["filename"]
                dest.write_bytes(_req(f"/view?{q}"))
                paths.append(dest)
    return paths, round(time.time() - t, 1)
