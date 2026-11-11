"""YOLO wrapper. ultralytics is imported lazily so the rules can be tested without it."""
from __future__ import annotations

import cv2
import numpy as np

from .rules import Detection, Level, Status

# Box colors (BGR). Smoke boxes take the color of the current alert level.
LEVEL_COLORS = {
    Level.CLEAR: (60, 160, 60),
    Level.WATCH: (0, 200, 230),
    Level.POSSIBLE_FIRE: (0, 140, 255),
    Level.CONFIRMED_FIRE: (40, 40, 220),
    Level.FEED_ISSUE: (150, 150, 150),
}
FIRE_COLOR = (0, 0, 255)


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
    """Draw thick, labelled boxes. Line width and text size scale with the image."""
    img = frame.copy()
    scale = max(img.shape[1] / 1280, 0.5)
    thick = max(2, int(3 * scale))
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        col = FIRE_COLOR if d.cls == "fire" else LEVEL_COLORS[status.level]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        label = f"{d.cls} {d.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, thick)
        y_top = max(0, y1 - th - 2 * thick)
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 2 * thick, y_top + th + 2 * thick), col, -1)
        cv2.putText(img, label, (x1 + thick, y_top + th + thick), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img
