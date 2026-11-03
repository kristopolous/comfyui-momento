"""Geometry as conditioning, in the graph.

The post's camera is where the photographer stood: the origin of Depth Anything 3's metric space, Y up, looking down
-Z. Where the bystander's camera goes and where the photographer stands, which way they face and where their hands
are all follow from that, so these nodes compute them and hand them to the models as control, then measure the result:

  Momento Camera Path    the post's mesh plus a ground plane at the photographer's feet, rendered as a depth video
                         along the move (MiniMax H3 Fun ControlNet-Union follows it)
  Momento Photographer   the photographer's 3D skeleton through the same path: a DWPose body video (left and right
                         limbs keep their colours, the face is hidden, so it says "back to us") and an inpaint mask
  Momento Gate Frames    frames to measure, with the box the 3D puts the photographer in (for SAM 3D Body)
  Momento Gate           SAM 3D Body's 3D pose against the 3D: facing, hands at the camera, standing where projected
"""
import json
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from comfy_api.latest import Types, io, ui

from . import pipeline as P

MOVES = [*P.MOVES, "still"]
CAMERAS = ["smartphone", "dslr"]
FEET = -1.63  # the ground under the photographer, metres below the post camera (eye height)

# OpenPose COCO-18 body keypoints, in DWPose's drawing order and colours
NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
         "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

# The photographer in metres, eyes at the origin, facing -Z; their right hand is +X. Seen from behind the face is
# hidden: no nose or eyes, which is what tells the model it's their back.
BODY = {"neck": (0, -0.22, 0.03), "r_shoulder": (0.19, -0.26, 0.03), "l_shoulder": (-0.19, -0.26, 0.03),
        "r_hip": (0.11, -0.80, 0.04), "l_hip": (-0.11, -0.80, 0.04), "r_knee": (0.12, -1.24, 0.0),
        "l_knee": (-0.12, -1.24, 0.02), "r_ankle": (0.12, FEET, 0.06), "l_ankle": (-0.12, FEET, 0.07),
        "r_ear": (0.08, -0.02, 0.02), "l_ear": (-0.08, -0.02, 0.02)}
ARMS = {  # lowered (about to shoot), then raised to shoot along the post's axis
    "low": {"r_elbow": (0.24, -0.52, 0.02), "r_wrist": (0.22, -0.76, -0.08),
            "l_elbow": (-0.24, -0.52, 0.03), "l_wrist": (-0.2, -0.78, -0.02)},
    "dslr": {"r_elbow": (0.23, -0.30, -0.12), "r_wrist": (0.07, -0.07, -0.20),
             "l_elbow": (-0.20, -0.34, -0.20), "l_wrist": (-0.04, -0.12, -0.28)},
    "smartphone": {"r_elbow": (0.22, -0.20, -0.22), "r_wrist": (0.06, 0.0, -0.42),
                   "l_elbow": (-0.22, -0.21, -0.22), "l_wrist": (-0.05, 0.0, -0.42)},
}
HELD = {"dslr": (0.0, -0.06, -0.16), "smartphone": (0.0, 0.02, -0.44)}  # the camera itself, when raised
RAISE = (0.25, 0.6)  # the arms come up over this part of the clip, then hold


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def camera_path(move, frames):
    """The bystander's eye for each frame: from the move's start into the end framing (still: no move)."""
    a = P.MOVES.get(move, P.START)
    return [tuple(p + (q - p) * ease(i / max(1, frames - 1)) for p, q in zip(a, P.START)) for i in range(frames)]


def skeleton(t, camera):
    k = ease((t - RAISE[0]) / (RAISE[1] - RAISE[0]))
    joints = dict(BODY)
    for name, lo in ARMS["low"].items():
        joints[name] = tuple(p + (q - p) * k for p, q in zip(lo, ARMS[camera][name]))
    held = tuple(p + (q - p) * k for p, q in zip(ARMS["low"]["r_wrist"], HELD[camera]))
    return joints, held


def project(point, eye, w, h):
    return P.project(point, eye, w, h, fov=P.lens(w, h))


def frame_joints(move, camera, frames, w, h):
    out = []
    for i, eye in enumerate(camera_path(move, frames)):
        joints, held = skeleton(i / max(1, frames - 1), camera)
        out.append(({n: project(p, eye, w, h) for n, p in joints.items()}, project(held, eye, w, h)))
    return out


def draw_pose(j2, w, h):
    """One DWPose-style body frame on black, as the DWPose / SDPose preprocessors draw it."""
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    stick = max(3, round(4 * h / 768))
    for (a, b), color in zip(LIMBS, COLORS):
        pa, pb = j2.get(NAMES[a]), j2.get(NAMES[b])
        if pa is None or pb is None:
            continue
        (x0, y0), (x1, y1) = pa, pb
        length, ang = math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts = [(cx + math.cos(r) * length / 2 * math.cos(ang) - math.sin(r) * stick * math.sin(ang),
                cy + math.cos(r) * length / 2 * math.sin(ang) + math.sin(r) * stick * math.cos(ang))
               for r in (math.radians(s) for s in range(0, 360, 15))]
        d.polygon(pts, fill=tuple(int(c * 0.6) for c in color))
    for i, name in enumerate(NAMES):
        if name in j2:
            x, y = j2[name]
            d.ellipse((x - stick, y - stick, x + stick, y + stick), fill=COLORS[i])
    return im


def draw_mask(j2, held, w, h, grow=0.06):
    """White where the photographer goes: torso, thick limbs, head, the camera and their feet, dilated."""
    m = Image.new("L", (w, h))
    d = ImageDraw.Draw(m)
    body_h = abs(j2["neck"][1] - (j2["r_ankle"][1] + j2["l_ankle"][1]) / 2) or h / 3
    width = max(6, int(body_h * 0.13))
    d.polygon([j2["r_shoulder"], j2["l_shoulder"], j2["l_hip"], j2["r_hip"]], fill=255)
    for a, b in LIMBS:
        if NAMES[a] in j2 and NAMES[b] in j2:
            d.line([j2[NAMES[a]], j2[NAMES[b]]], fill=255, width=width)
    hx, hy = (j2["r_ear"][0] + j2["l_ear"][0]) / 2, (j2["r_ear"][1] + j2["l_ear"][1]) / 2
    r = body_h * 0.075
    d.ellipse((hx - r, hy - r * 1.3, hx + r, hy + r * 1.1), fill=255)
    r = body_h * 0.06
    d.ellipse((held[0] - r, held[1] - r, held[0] + r, held[1] + r), fill=255)
    for name in ("r_ankle", "l_ankle"):
        x, y = j2[name]
        d.ellipse((x - width, y - width * 0.5, x + width, y + width * 0.8), fill=255)
    size = max(3, int(body_h * grow) | 1)
    return m.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.GaussianBlur(size / 4))


def ground_mesh(y=FEET, near=1.0, far=80.0, half=60.0):
    """A ground plane under the photographer, so a camera that sees past the photo still sees the ground."""
    v = torch.tensor([[-half, y, near * 4], [half, y, near * 4], [half, y, -far], [-half, y, -far]],
                     dtype=torch.float32)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    return Types.MESH(vertices=v[None], faces=f[None])


def _images(pils):
    return torch.from_numpy(np.stack([np.asarray(p, dtype=np.float32) / 255.0 for p in pils]))


class MomentoCameraPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoCameraPath",
            display_name="Momento Camera Path",
            category="Momento",
            description="The post's 3D place (plus the ground under the photographer) rendered as a depth video "
                        "along the bystander's camera move. Feed it to MiniMax H3 Fun ControlNet-Union.",
            inputs=[
                io.Mesh.Input("mesh"),
                io.Combo.Input("move", options=MOVES, default=MOVES[0]),
                io.Int.Input("frames", default=107, min=5, max=1000),
                io.Int.Input("width", default=448, min=64, max=4096, step=32),
                io.Int.Input("height", default=768, min=64, max=4096, step=32),
                io.Boolean.Input("ground", default=True),
            ],
            outputs=[io.Image.Output(display_name="depth")],
        )

    @classmethod
    def execute(cls, mesh, move, frames, width, height, ground=True) -> io.NodeOutput:
        from comfy_extras.nodes_gaussian_splat import _lookat_camera_info
        from comfy_extras.nodes_mesh_postprocess import RenderMesh
        from comfy_extras.nodes_save_3d import MergeMeshes
        if ground:
            mesh = MergeMeshes.execute(meshes={"mesh_0": mesh, "mesh_1": ground_mesh()}).args[0]
        fov = P.lens(width, height)
        out = []
        for eye in camera_path(move, frames):
            cam = _lookat_camera_info(list(eye), list(P.TARGET), fov, "cpu")
            out.append(RenderMesh.execute(mesh, "depth", width, height, "#000000", None, cam).args[0])
        return io.NodeOutput(torch.cat(out, dim=0))


class MomentoPhotographer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoPhotographer",
            display_name="Momento Photographer",
            category="Momento",
            description="The photographer from 3D: standing at the post's camera, facing its axis, raising their "
                        "camera, projected through the bystander's move. A DWPose body video and an inpaint mask "
                        "for MiniMax H3 Fun ControlNet-Union.",
            inputs=[
                io.Combo.Input("move", options=MOVES, default=MOVES[0]),
                io.Combo.Input("camera", options=CAMERAS, default=CAMERAS[0]),
                io.Int.Input("frames", default=107, min=5, max=1000),
                io.Int.Input("width", default=448, min=64, max=4096, step=32),
                io.Int.Input("height", default=768, min=64, max=4096, step=32),
            ],
            outputs=[io.Image.Output(display_name="pose"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, move, camera, frames, width, height) -> io.NodeOutput:
        per = frame_joints(move, camera, frames, width, height)
        pose = _images([draw_pose(j2, width, height) for j2, _ in per])
        mask = torch.from_numpy(np.stack([np.asarray(draw_mask(j2, held, width, height), dtype=np.float32) / 255.0
                                          for j2, held in per]))
        return io.NodeOutput(pose, mask)


GATE_AT = (0.7, 0.85, 0.98)


class MomentoGateFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoGateFrames",
            display_name="Momento Gate Frames",
            category="Momento",
            description="Frames of a take to measure, after the camera is raised, each with the box the 3D puts "
                        "the photographer in, so SAM 3D Body measures them and nobody else.",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("move", options=MOVES, default=MOVES[0]),
                io.Combo.Input("camera", options=CAMERAS, default=CAMERAS[0]),
            ],
            outputs=[io.Image.Output(display_name="frames"), io.BoundingBox.Output(display_name="bboxes")],
        )

    @classmethod
    def execute(cls, images, move, camera) -> io.NodeOutput:
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        path = camera_path(move, n)
        picks, boxes = [], []
        for t in GATE_AT:
            i = min(n - 1, round(t * (n - 1)))
            joints, _ = skeleton(t, camera)
            pts = [project(p, path[i], w, h) for p in joints.values()]
            x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
            y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
            mx, my = (x1 - x0) * 0.35 + 8, (y1 - y0) * 0.12 + 8
            x0, y0, x1, y1 = max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx)), min(h, int(y1 + my))
            picks.append(images[i:i + 1])
            boxes.append([{"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)}])
        return io.NodeOutput(torch.cat(picks, dim=0), boxes)


def _expected_forward(eye):
    """The post's axis (world -Z) in the bystander camera's frame: x right, y down, z forward."""
    f = [t - e for t, e in zip(P.TARGET, eye)]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    r = [-f[2], 0.0, f[0]]
    n = math.sqrt(sum(v * v for v in r))
    r = [v / n for v in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    return [-axis[2] for axis in (r, [-v for v in u], f)]


def measure(k3d, eye):
    """SAM 3D Body MHR70 keypoints (camera space, metres) -> facing error in degrees, how high the hands are above
    the shoulders, and how far to the side of the head they are. MHR70: 3/4 ears, 5/6 left/right shoulder,
    21-62 hands."""
    k3d = np.asarray(k3d, dtype=np.float64)
    ls, rs = k3d[5], k3d[6]
    right = rs - ls
    right /= np.linalg.norm(right) or 1.0
    fwd = np.cross([0.0, -1.0, 0.0], right)
    fwd /= np.linalg.norm(fwd) or 1.0
    deg = math.degrees(math.acos(float(np.clip(fwd @ np.asarray(_expected_forward(eye)), -1, 1))))
    hands = k3d[21:63]
    hands_up = float((ls[1] + rs[1]) / 2 - hands[:, 1].mean())
    side = float((hands.mean(0) - (k3d[3] + k3d[4]) / 2) @ right)
    return deg, hands_up, side


class MomentoGate(io.ComfyNode):
    MAX_DEG, MIN_HANDS, MAX_SIDE, MAX_OFF = 35.0, 0.05, 0.07, 0.12

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MomentoGate",
            display_name="Momento Gate",
            category="Momento",
            description="Measures the photographer SAM 3D Body found against the 3D they were built from: facing "
                        "within 35 degrees of the post's axis, hands at the eye (above the shoulders, centred in "
                        "front of the face) and standing where projected. Fails the take otherwise.",
            inputs=[
                io.Custom("MHR_POSE_DATA").Input("pose_data"),
                io.Combo.Input("move", options=MOVES, default=MOVES[0]),
                io.Combo.Input("camera", options=CAMERAS, default=CAMERAS[0]),
                io.Int.Input("frames", default=107, min=5, max=1000),
                io.Int.Input("width", default=448, min=64, max=4096, step=32),
                io.Int.Input("height", default=768, min=64, max=4096, step=32),
            ],
            outputs=[io.Boolean.Output(display_name="ok"), io.String.Output(display_name="report")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, pose_data, move, camera, frames, width, height) -> io.NodeOutput:
        path = camera_path(move, frames)
        report, ok = [], True
        for t, people in zip(GATE_AT, pose_data.get("frames", [])):
            eye = path[min(frames - 1, round(t * (frames - 1)))]
            want = project(skeleton(t, camera)[0]["neck"], eye, width, height)
            best = None
            for p in people or []:
                k2 = np.asarray(p.get("pred_keypoints_2d"))
                off = math.dist(((k2[5][0] + k2[6][0]) / 2, (k2[5][1] + k2[6][1]) / 2), want) / width
                if best is None or off < best[0]:
                    best = (off, p.get("pred_keypoints_3d"))
            if best is None:
                report.append({"t": t, "found": False})
                ok = False
                continue
            deg, hands_up, side = measure(best[1], eye)
            r = {"t": t, "deg": round(deg, 1), "hands_up_m": round(hands_up, 2), "hands_side_m": round(side, 2),
                 "off": round(best[0], 3)}
            r["ok"] = (deg <= cls.MAX_DEG and best[0] <= cls.MAX_OFF and
                       (t < RAISE[1] or (hands_up >= cls.MIN_HANDS and abs(side) <= cls.MAX_SIDE)))
            ok = ok and r["ok"]
            report.append(r)
        text = json.dumps({"ok": ok, "frames": report})
        return io.NodeOutput(ok, text, ui=ui.PreviewText("GATE " + text))


NODES = [MomentoCameraPath, MomentoPhotographer, MomentoGateFrames, MomentoGate]
