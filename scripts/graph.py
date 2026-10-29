"""Write the app graph in API format.

    python scripts/graph.py > workflows/momento_api.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comfyui_momento"))
from graph import Graph, momento  # noqa: E402,F401

if __name__ == "__main__":
    json.dump(momento(), sys.stdout, indent=1, ensure_ascii=False)
