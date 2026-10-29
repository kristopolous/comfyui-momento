"""Turn an API-format graph into a frontend workflow (editable in ComfyUI, listed under Workflows),
using comfy-cli's own edit ops so node layouts and widgets match the live catalog.

    COMFY_HOST=<gpu-host> python scripts/ui.py workflows/momento_api.json workflows/momento.json --app
"""
import json
import os
import subprocess
import sys

EMPTY = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [],
         "config": {}, "extra": {}, "version": 0.4}
CATALOG = ["--where", "local", "--host", os.environ.get("COMFY_HOST", "127.0.0.1"), "--port", "8188"]


def op(*args):
    out = subprocess.run(["comfy", "workflow", *args, *CATALOG], capture_output=True, text=True)
    env = json.loads(out.stdout.splitlines()[0])
    if not env["ok"]:
        raise RuntimeError(f"{args}: {env['error']['message']}")
    return env["data"]


def convert(api, path):
    json.dump(EMPTY, open(path, "w"))
    ids, pos = {}, {}
    # lay nodes out left to right by dependency depth
    def depth(nid, seen=()):
        links = [v for v in api[nid]["inputs"].values() if isinstance(v, list)]
        return 0 if not links else 1 + max(depth(v[0], seen + (nid,)) for v in links)
    rows = {}
    for nid in api:
        d = depth(nid)
        pos[nid] = f"{40 + d * 360},{60 + rows.get(d, 0) * 220}"
        rows[d] = rows.get(d, 0) + 1
    for nid, node in api.items():
        data = op("add-node", path, node["class_type"], "--at", pos[nid], "--allow-deprecated")
        ids[nid] = data["op"]["node_id"]
    for nid, node in api.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                op("connect", path, f"{ids[value[0]]}.{value[1]}", f"{ids[nid]}.{name}")
            else:
                op("set-widget", path, f"{ids[nid]}.{name}", json.dumps(value))
    return path


APP_INPUTS = [("LoadImage", "image")]
APP_OUTPUTS = ["MomentoPlan", "SaveVideo"]


def app_mode(path, inputs=APP_INPUTS, outputs=APP_OUTPUTS):
    """Make the workflow open in App Mode with these widgets as the app's inputs and these nodes as its outputs."""
    d = json.load(open(path))
    d.setdefault("extra", {})
    d["extra"]["linearMode"] = True
    d["extra"]["linearData"] = {"inputs": [[n["id"], w, *rest] for t, w, *rest in inputs for n in d["nodes"]
                                           if n["type"] == t],
                                "outputs": [n["id"] for t in outputs for n in d["nodes"] if n["type"] == t]}
    json.dump(d, open(path, "w"))


if __name__ == "__main__":
    convert(json.load(open(sys.argv[1])), sys.argv[2])
    if "--app" in sys.argv:
        app_mode(sys.argv[2])
    print("wrote", sys.argv[2])
