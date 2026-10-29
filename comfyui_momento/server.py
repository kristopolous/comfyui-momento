"""The Momento page's backend, on ComfyUI's own server.

    GET  /momento                      the Instagram-style page
    GET  /momento/profile/<name>       a saved public profile: {"name", "picture", "posts": [urls]}
    POST /momento/run                  form: name + post index -> {"id"}
    GET  /momento/status/<id>          {"state": queued|running|done|error, "node", "video", "plan", "error"}

A name is looked up by opening its public Instagram profile in headless Chromium, the same page anyone sees in a
browser, and reading the profile picture and posts off it. They're saved under input/momento/profiles/<name>/.
Runs use the server's Comfy API key (COMFY_API_KEY) for Gemini and MiniMax Music 3.
"""
import asyncio
import html.parser
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

import folder_paths
from server import PromptServer

from .graph import momento

PAGE = Path(__file__).parent / "web" / "app.html"
MAX_WAITING = 4  # runs queued at once; more gets "busy"
NAME = re.compile(r"^[A-Za-z0-9._]{1,30}$")


def _inbox():
    d = Path(folder_paths.get_input_directory()) / "momento"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profile(name):
    d = _inbox() / "profiles" / name.lower()
    pic = next(iter(sorted(d.glob("picture.*"))), None)
    posts = sorted(p for p in d.glob("post*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    return (d, pic, posts) if pic and posts else None


class _Images(html.parser.HTMLParser):
    """Profile picture and post images from a rendered Instagram profile page."""

    def __init__(self):
        super().__init__()
        self.picture, self.posts, self._link = None, [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and "/p/" in (a.get("href") or ""):
            self._link = a["href"]
        elif tag == "img" and a.get("src", "").startswith("https://"):
            alt = a.get("alt") or ""
            if "profile picture" in alt and not self.picture:
                self.picture = a["src"]
            elif self._link and a["src"] not in self.posts:
                self.posts.append(a["src"])

    def handle_endtag(self, tag):
        if tag == "a":
            self._link = None


def _browser():
    return next((shutil.which(b) for b in ("chromium", "google-chrome", "chromium-browser") if shutil.which(b)), None)


async def _fetch_profile(name):
    """Open the public profile in headless Chromium and save its picture and posts. Returns _profile() or None."""
    browser = _browser()
    if not browser:
        return None
    proc = await asyncio.create_subprocess_exec(
        browser, "--headless=new", "--disable-gpu", "--virtual-time-budget=8000", "--dump-dom",
        f"https://www.instagram.com/{name}/", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        dom, _ = await asyncio.wait_for(proc.communicate(), 60)
    except asyncio.TimeoutError:
        proc.kill()
        return None
    page = _Images()
    page.feed(dom.decode("utf-8", "replace"))
    if not page.picture or not page.posts:
        return None
    d = _inbox() / "profiles" / name.lower()
    d.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as s:
        for target, url in [("picture", page.picture)] + [(f"post{i + 1:02d}", u) for i, u in enumerate(page.posts[:12])]:
            async with s.get(url) as r:
                if r.status == 200:
                    (d / f"{target}.jpg").write_bytes(await r.read())
    return _profile(name)


def _key():
    if os.environ.get("COMFY_API_KEY"):
        return os.environ["COMFY_API_KEY"]
    env = Path(folder_paths.base_path).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("COMFY_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _url(path):
    rel = Path(path).relative_to(folder_paths.get_input_directory())
    return f"view?type=input&subfolder={rel.parent.as_posix()}&filename={rel.name}"


routes = PromptServer.instance.routes

# which node each run is on: ComfyUI sends "executing" {node, prompt_id} as it goes; keep the latest per run
_executing = {}
_send_sync = PromptServer.instance.send_sync


def _track(event, data, sid=None):
    if event == "executing" and isinstance(data, dict) and data.get("prompt_id"):
        _executing[data["prompt_id"]] = data.get("node")
    return _send_sync(event, data, sid)


PromptServer.instance.send_sync = _track


@routes.get("/momento")
@routes.get("/momento/")
async def page(request):
    return web.FileResponse(PAGE)


@routes.get("/momento/profile/{name}")
async def profile(request):
    name = request.match_info["name"].strip().lstrip("@").lower()
    if not NAME.match(name):
        return web.json_response({"name": name, "found": False})
    found = None if request.query.get("refresh") else _profile(name)
    found = found or await _fetch_profile(name)
    if not found:
        return web.json_response({"name": name, "found": False})
    _, pic, posts = found
    return web.json_response({"name": name, "found": True, "picture": _url(pic), "posts": [_url(p) for p in posts]})


async def _queue_length():
    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{PromptServer.instance.port}/queue") as r:
            q = await r.json()
    return len(q["queue_running"]) + len(q["queue_pending"])


@routes.post("/momento/run")
async def run(request):
    if await _queue_length() >= MAX_WAITING:
        return web.json_response({"error": "Busy: a few moments are already being made. Try again in a few minutes."},
                                 status=503)
    key = _key()
    if not key:
        return web.json_response({"error": "This server has no Comfy API key set."}, status=500)
    form = await request.post()
    job = uuid.uuid4().hex[:12]
    name = str(form.get("name", "")).strip().lstrip("@").lower()
    found = _profile(name) if NAME.match(name) else None
    if not found or not str(form.get("post", "")).isdigit():
        return web.json_response({"error": "Open a profile and pick a post first."}, status=400)
    _, picture, posts = found
    i = int(form["post"])
    if not 0 <= i < len(posts):
        return web.json_response({"error": "No such post."}, status=400)
    post = posts[i]
    rel = lambda p: Path(p).relative_to(folder_paths.get_input_directory()).as_posix()  # noqa: E731
    graph = momento(rel(picture), rel(post), prefix=f"momento/{job}")
    body = {"prompt": graph, "extra_data": {"api_key_comfy_org": key}}
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{PromptServer.instance.port}/prompt", json=body) as r:
            res = await r.json()
    if "prompt_id" not in res:
        return web.json_response({"error": json.dumps(res.get("node_errors") or res)[:500]}, status=500)
    return web.json_response({"id": res["prompt_id"]})


@routes.get("/momento/status/{id}")
async def status(request):
    pid = request.match_info["id"]
    port = PromptServer.instance.port
    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{port}/history/{pid}") as r:
            hist = (await r.json()).get(pid)
        if not hist:
            async with s.get(f"http://127.0.0.1:{port}/queue") as r:
                q = await r.json()
            running = next((item for item in q["queue_running"] if item[1] == pid), None)
            node = _executing.get(pid) if running else None
            kind = running[2].get(str(node), {}).get("class_type") if running and node else None
            return web.json_response({"state": "running" if running else "queued", "node": kind})
    st = hist["status"]
    if st.get("status_str") == "error":
        msg = next((m[1].get("exception_message") for m in st.get("messages", []) if m[0] == "execution_error"), "")
        return web.json_response({"state": "error", "error": (msg or "failed")[:400]})
    video, plan = None, None
    for out in hist["outputs"].values():
        for f in out.get("images", []) + out.get("videos", []):
            if f["filename"].endswith(".mp4"):
                video = f"view?type={f['type']}&subfolder={f['subfolder']}&filename={f['filename']}"
        if out.get("text"):
            plan = out["text"][0]
    return web.json_response({"state": "done", "video": video, "plan": plan})
