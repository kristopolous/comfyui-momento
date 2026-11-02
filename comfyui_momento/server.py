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
import base64
import html.parser
import json
import logging
import os
import random
import re
import shutil
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

import folder_paths
from server import PromptServer

from . import pipeline
from .graph import momento

PAGE = Path(__file__).parent / "web" / "app.html"
MAX_WAITING = 4  # runs queued at once; more gets "busy"
TAKES = 3  # the critic (Gemini) can send a take back this many times
JOBS = {}  # first prompt id -> {"pid": the current take's prompt id, "take", "notes", "args"}
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


def _env(name):
    """A setting from the environment, or from the .env file next to the ComfyUI folder."""
    if os.environ.get(name):
        return os.environ[name]
    env = Path(folder_paths.base_path).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return None


async def _ask(system, text, images):
    """One call to the planner model (an Ollama-API vision model). -> parsed JSON, or None."""
    url, model = _env("PLANNER_URL"), _env("PLANNER_MODEL")
    if not url or not model:
        return None
    body = {"model": model, "stream": False, "format": "json", "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": text, "images": [base64.b64encode(b).decode() for b in images]}]}
    for _ in range(3):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as r:
                    reply = (await r.json())["message"]["content"]
            return json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
            pass
    return None


async def _review(video, post):
    """The critic: the planner model looks at four frames of the take and the real photo. -> verdict dict"""
    frames = []
    for t in (0.5, 2.0, 3.5, 4.3):
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-vf", "scale=-2:384",
            "-f", "image2", "-c:v", "mjpeg", "-", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        if out:
            frames.append(out)
    if not frames:
        return {"pass": True, "score": None, "problem": ""}
    verdict = await _ask(pipeline.CRITIC, "Review this take.", frames + [Path(post).read_bytes()])
    return {"pass": bool((verdict or {}).get("pass", True)), "score": (verdict or {}).get("score"),
            "problem": str((verdict or {}).get("problem") or "")}


async def _plan(picture, post):
    """The planner plans the moment (PLANNER_URL, PLANNER_MODEL). None if unset or it fails."""
    root = Path(folder_paths.get_input_directory())
    plan = await _ask(pipeline.PLAN, "Left: the profile picture. Right: the photo they posted. Plan the moment.",
                      [(root / p).read_bytes() for p in (picture, post)])
    if plan and all(plan.get(k) for k in ("scene", "action", "photographer", "soundscape")):
        return plan
    return None


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
    if not found and not _browser():
        msg = "This server can't open Instagram profiles: install chromium (e.g. apt install chromium)."
        logging.warning("Momento: %s", msg)
        return web.json_response({"name": name, "found": False, "error": msg})
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
    rel = lambda p: Path(p).relative_to(folder_paths.get_input_directory()).as_posix()  # noqa: E731
    camera = "dslr" if form.get("camera") == "dslr" else "smartphone"
    wide = form.get("frame") == "landscape"
    args = (rel(picture), rel(posts[i]), job, key, camera, wide)
    pid, err = await _shoot(*args)
    if err:
        return web.json_response({"error": err}, status=500)
    JOBS[pid] = {"pid": pid, "take": 1, "notes": [], "args": args}
    return web.json_response({"id": pid})


async def _shoot(picture, post, job, key, camera="smartphone", wide=False):
    """Queue one take; every take is different (a new seed). -> (prompt id, error)"""
    plan = await _plan(picture, post)
    if plan:  # the 3D pipeline, with a camera move every take
        have = folder_paths.get_filename_list("diffusion_models")
        graph = pipeline.build(picture, post, plan, seed=random.randrange(2**31), prefix=f"momento/{job}",
                               move=random.choice(list(pipeline.MOVES)), camera=camera,
                               h3_unet=next((u for u in pipeline.H3_UNETS if u in have), pipeline.H3_UNETS[0]),
                               **(dict(width=768, height=448) if wide else {}))
    else:  # no planner configured: the original single graph
        graph = momento(picture, post, seed=random.randrange(2**31), angle="wide", prefix=f"momento/{job}",
                        camera=camera)
    # a client_id makes ComfyUI track the running node (last_node_id), which /momento/status reports
    body = {"prompt": graph, "client_id": f"momento-{job}", "extra_data": {"api_key_comfy_org": key}}
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{PromptServer.instance.port}/prompt", json=body) as r:
            res = await r.json()
    if "prompt_id" not in res:
        return None, json.dumps(res.get("node_errors") or res)[:500]
    return res["prompt_id"], None


@routes.get("/momento/status/{id}")
async def status(request):
    job = JOBS.get(request.match_info["id"]) or {"pid": request.match_info["id"], "take": 1, "notes": []}
    pid = job["pid"]
    port = PromptServer.instance.port
    takes = {"take": job["take"], "notes": job["notes"], "rejected": job.get("rejected", [])}
    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{port}/history/{pid}") as r:
            hist = (await r.json()).get(pid)
        if not hist:
            async with s.get(f"http://127.0.0.1:{port}/queue") as r:
                q = await r.json()
            running = next((item for item in q["queue_running"] if item[1] == pid), None)
            srv = PromptServer.instance  # ComfyUI keeps the running prompt and node here
            node = srv.last_node_id if running and getattr(srv, "last_prompt_id", None) == pid else None
            spec = running[2].get(str(node), {}) if running and node else {}
            kind = "Critic" if spec.get("_meta", {}).get("title", "").startswith("Critic") else spec.get("class_type")
            return web.json_response({"state": "running" if running else "queued", "node": kind, **takes})
    st = hist["status"]
    if st.get("status_str") == "error":
        msg = next((m[1].get("exception_message") for m in st.get("messages", []) if m[0] == "execution_error"), "")
        return web.json_response({"state": "error", "error": (msg or "failed")[:400]})
    video, plan, verdict = None, None, {}
    for out in hist["outputs"].values():
        for f in out.get("images", []) + out.get("videos", []):
            if f["filename"].endswith(".mp4"):
                video = f"view?type={f['type']}&subfolder={f['subfolder']}&filename={f['filename']}"
        for text in out.get("text", []):
            if text.startswith("REVIEW "):
                verdict = json.loads(text[7:])
            elif text.startswith("THE SHOT"):
                plan = text
    if not verdict and video and job.get("args"):  # the 3D pipeline has no review node: the planner reviews it
        seen = job.setdefault("seen", {}).get(pid)
        if seen:
            verdict = seen[3]
        else:
            q = dict(p.split("=", 1) for p in video.split("?", 1)[1].split("&"))
            path = Path(folder_paths.get_output_directory()) / q["subfolder"] / q["filename"]
            verdict = await _review(path, Path(folder_paths.get_input_directory()) / job["args"][1])
    # every finished take, by prompt id (the page polls this more than once per take)
    rank = (verdict.get("score") or 0) + (10 if verdict.get("pass", True) else 0)
    job.setdefault("seen", {})[pid] = (rank, video, plan, verdict)
    if not verdict.get("pass", True) and job.get("args") and job["take"] < TAKES:
        # the critic sent it back: reshoot with a new seed, and say why
        new, err = await _shoot(*job["args"])
        if not err:
            job["notes"].append(verdict.get("problem") or "not good enough")
            job.setdefault("rejected", []).append(video)  # still watchable: "eh, good enough" is allowed
            job["take"] += 1
            job["pid"] = new
            return web.json_response({"state": "running", "node": "Reshoot", "take": job["take"],
                                      "notes": job["notes"], "rejected": job["rejected"]})
    # done: show the highest-rated take, which isn't always the last one
    _, video, plan, verdict = max(job["seen"].values(), key=lambda t: t[0])
    return web.json_response({"state": "done", "video": video, "plan": plan, "review": verdict, **takes})
