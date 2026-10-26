"""Glue: frames in, status + annotated frame out, alerts saved."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import store
from .detector import draw, frame_ok
from .rules import AlertEngine, Config, Detection, Status

IMG_EXT = {".jpg", ".jpeg", ".png"}


def iter_frames(source: str, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) from a video file, a folder of images, or a camera index."""
    p = Path(source)
    if p.is_dir():  # e.g. a FIgLib sequence, frames sorted by file name
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        for i, f in enumerate(files):
            if i % step == 0:
                img = cv2.imread(str(f))
                if img is not None:
                    yield i, img
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i, img
            i += 1
    finally:
        cap.release()


class SmokeWatch:
    """detector: any object with .detect(frame) -> list[Detection]."""

    def __init__(self, detector, config: Config | None = None,
                 source_name: str = "unknown", snapshot_dir: str = "outputs/snapshots",
                 save_alerts: bool = True):
        self.detector = detector
        self.engine = AlertEngine(config)
        self.source_name = source_name
        self.snapshot_dir = Path(snapshot_dir)
        self.save_alerts = save_alerts
        self._prev: np.ndarray | None = None

    def process(self, frame: np.ndarray, index: int = 0,
                timestamp: float | None = None) -> tuple[Status, list[Detection], np.ndarray]:
        ok = frame_ok(frame, self._prev)
        self._prev = frame
        dets = self.detector.detect(frame) if ok else []
        status = self.engine.update(dets, frame_ok=ok, timestamp=timestamp)
        annotated = draw(frame, dets, status)
        if status.escalated and self.save_alerts:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            snap = self.snapshot_dir / f"{self.source_name}_{index:06d}.jpg"
            cv2.imwrite(str(snap), annotated)
            store.add_alert(self.source_name, index, status.level.label,
                            status.reason, str(snap))
        return status, dets, annotated
