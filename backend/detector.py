"""YOLO wrapper. ultralytics is imported lazily so the rules can be tested without it."""
from __future__ import annotations

import cv2
import numpy as np

from .rules import Detection, Level, Status

COLORS = {"smoke": (200, 200, 200), "fire": (0, 80, 255)}
LEVEL_COLORS = {
    Level.CLEAR: (60, 160, 60),
    Level.WATCH: (0, 200, 220),
    Level.POSSIBLE_FIRE: (0, 140, 255),
    Level.CONFIRMED_FIRE: (0, 0, 220),
    Level.FEED_ISSUE: (150, 150, 150),
}


class YoloDetector:
    def __init__(self, weights: str = "weights/best.pt", imgsz: int = 640,
                 conf: float = 0.3, device: str | None = None):
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(weights)  # .pt, or a TensorRT .engine file
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        names = res.names
        out = []
        for b in res.boxes:
            cls = str(names[int(b.cls)]).lower()
            out.append(Detection(cls, float(b.conf), tuple(map(float, b.xyxy[0]))))
        return out


def frame_ok(frame: np.ndarray, prev: np.ndarray | None = None,
             min_brightness: float = 15.0) -> bool:
    """False for very dark frames or an exact repeat of the previous frame."""
    if float(frame.mean()) < min_brightness:
        return False
    if prev is not None and prev.shape == frame.shape and np.array_equal(frame, prev):
        return False
    return True


def draw(frame: np.ndarray, dets: list[Detection], status: Status) -> np.ndarray:
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        col = COLORS.get(d.cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    col = LEVEL_COLORS[status.level]
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), col, -1)
    cv2.putText(img, status.level.label.upper(), (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img
