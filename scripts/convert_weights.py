"""Rebuild a YOLO checkpoint with the local Ultralytics version.

Weights trained on Colab (newer PyTorch/Ultralytics) can crash older versions
on the Jetson. This copies only the learned weights into a fresh model built
locally, then saves a clean checkpoint.

Example:
    python scripts/convert_weights.py weights/best.pt weights/best_jetson.pt
"""
import sys

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel


def match_head_width(model: DetectionModel, state: dict) -> None:
    """Newer Ultralytics can build a wider class branch (cv3). Rebuild ours to match."""
    det = model.model[-1]
    key = f"model.{len(model.model) - 1}.cv3.0.0.conv.weight"
    c3 = state[key].shape[0]
    if det.cv3[0][0].conv.out_channels == c3:
        return
    ch = [seq[0].conv.in_channels for seq in det.cv3]
    det.cv3 = nn.ModuleList(
        nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, det.nc, 1)) for x in ch
    )
    print(f"head width adjusted to {c3}")


def main(src_path: str, dst_path: str, cfg: str = "yolov8n.yaml") -> None:
    ck = torch.load(src_path, map_location="cpu", weights_only=False)  # your own trusted file
    src = ck.get("ema") or ck["model"]
    names = dict(src.names)
    state = {k: v.detach().float().clone().contiguous() for k, v in src.state_dict().items()}
    del ck, src

    model = DetectionModel(cfg, nc=len(names), verbose=False)
    match_head_width(model, state)
    model.load_state_dict(state, strict=True)
    model.names = names
    model.eval()

    torch.save({
        "model": model.half(),
        "train_args": {"task": "detect", "imgsz": 640, "single_cls": False, "data": None},
        "epoch": -1,
    }, dst_path)
    print(f"saved {dst_path}  classes={names}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
