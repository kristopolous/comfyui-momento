"""How much the photographer's phone (and body) changes through a take, measured, not asked.

    python scripts/drift.py take_raw.mp4 [TILT]

The boxes are the ones the take was made with, projected from 3D (geo.layout at that look-at height), so the crop is
the phone in every frame. Each crop is embedded with DINOv2; a phone that morphs jumps between neighbouring frames.
Prints, per box: the worst frame-to-frame jump (cosine distance) and where it is, and the mean jump.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent))
import geo  # noqa: E402

MODEL = "facebook/dinov2-small"


def frames(video):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), str(Path(d) / "%04d.png")], check=True)
        return [Image.open(p).convert("RGB") for p in sorted(Path(d).glob("*.png"))]


def drift(video, tilt, camera="smartphone"):
    ims = frames(video)
    geo.W, geo.H = ims[0].size
    geo.TARGET = (geo.TARGET[0], tilt, geo.TARGET[2])
    proc, model = AutoImageProcessor.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    out = {}
    for name, o in zip(("body", "phone"), geo.layout(camera=camera)):
        y0, x0, y1, x1 = o["bbox"]
        pad = 0.15 * max(x1 - x0, y1 - y0)  # a little context: the hands around the phone
        crop = (max(0, (x0 - pad) * geo.W / 1000), max(0, (y0 - pad) * geo.H / 1000),
                min(geo.W, (x1 + pad) * geo.W / 1000), min(geo.H, (y1 + pad) * geo.H / 1000))
        with torch.no_grad():
            e = torch.cat([model(**proc(images=[im.crop(crop) for im in ims[i:i + 16]], return_tensors="pt"))
                           .pooler_output for i in range(0, len(ims), 16)])
        e = torch.nn.functional.normalize(e, dim=1)
        jump = (1 - (e[1:] * e[:-1]).sum(1)).numpy()
        out[name] = dict(worst=float(jump.max()), at=int(jump.argmax()) + 1, mean=float(jump.mean()),
                         p95=float(np.percentile(jump, 95)))
    return out


if __name__ == "__main__":
    video, tilt = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else geo.TARGET[1]
    for name, r in drift(video, tilt).items():
        print(f"{Path(video).name} {name:5s} worst jump {r['worst']:.4f} at frame {r['at']}, "
              f"p95 {r['p95']:.4f}, mean {r['mean']:.4f}")
