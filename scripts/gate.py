"""Gate an existing take: SAM 3D Body measures the photographer against the 3D it was built from (Momento Gate).

    COMFY_URL=http://<gpu-host>:8188 python scripts/gate.py take_raw.mp4 MOVE CAMERA
prints the gate's report and exits 0 if the take passes, 1 if it doesn't.
"""
import sys

import comfy
from local import Graph, run


def gate(take, move, camera, frames=107, width=576, height=1024):
    g = Graph()
    video = g.add("GetVideoComponents", video=g.add("LoadVideo", file=take)())
    picks = g.add("MomentoGateFrames", images=video(0), move=move, camera=camera)
    body = g.add("SAM3DBody_Predict", image=picks(0), bboxes=picks(1), run_hand_refinement=False, fov=62.0,
                 batch_size=4, sam3d_body_model=g.add("SAM3DBody_Loader",
                                                      model_file="sam_3d_body_dinov3_bf16.safetensors")())
    g.add("MomentoGate", pose_data=body(), move=move, camera=camera, frames=frames, width=width, height=height)
    return run(g).get("gate")


if __name__ == "__main__":
    path, move, camera = sys.argv[1:4]
    report = gate(comfy.upload(path, f"momento_gate_{path.rsplit('/', 1)[-1]}"), move, camera)
    print(report)
    sys.exit(0 if report and report["ok"] else 1)
