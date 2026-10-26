"""Decision logic for SmokeWatch.

Turns raw detections (boxes) into an alert level per camera:
CLEAR, WATCH, POSSIBLE_FIRE, CONFIRMED_FIRE or FEED_ISSUE.

Pure Python, no model or GPU needed, so it is easy to unit test.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


class Level(IntEnum):
    CLEAR = 0
    WATCH = 1
    POSSIBLE_FIRE = 2
    CONFIRMED_FIRE = 3
    FEED_ISSUE = 4  # special state, not part of the escalation order

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class Detection:
    cls: str  # "smoke" or "fire"
    conf: float
    box: Box


@dataclass
class Config:
    conf_alert: float = 0.5  # at or above: counts toward alerts
    conf_watch: float = 0.3  # between watch and alert: Watch only
    persistence_frames: int = 3  # Rule 1
    growth_frames: int = 5  # Rule 2
    growth_ratio: float = 1.2  # Rule 2 (20% growth)
    max_missed: int = 2  # tracks survive this many missed frames
    match_iou: float = 0.3
    exclusion_zones: list[Box] = field(default_factory=list)  # Rule 3


@dataclass
class Track:
    id: int
    box: Box
    hits: int = 1
    missed: int = 0
    areas: deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class Status:
    level: Level
    reason: str
    escalated: bool = False  # True on the frame an alert level is first reached
    first_detection_ts: Optional[float] = None
    tracks: list[Track] = field(default_factory=list)


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def center_in(box: Box, zone: Box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


class AlertEngine:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self._next_id = 1
        self._fire_streak = 0
        self._prev_level = Level.CLEAR
        self.first_detection_ts: Optional[float] = None
        self.log: list[str] = []  # smoke seen inside exclusion zones (Rule 3)

    # ---- helpers -------------------------------------------------------
    def _excluded(self, box: Box) -> bool:
        return any(center_in(box, z) for z in self.cfg.exclusion_zones)

    def _match(self, smoke: list[Detection]) -> None:
        """Greedy IoU matching of confident smoke boxes to existing tracks."""
        free = list(self.tracks)
        matched: set[int] = set()
        for d in sorted(smoke, key=lambda d: -d.conf):
            best, best_iou = None, self.cfg.match_iou
            for t in free:
                if t.id in matched:
                    continue
                v = iou(t.box, d.box)
                if v >= best_iou:
                    best, best_iou = t, v
            if best is None:
                t = Track(self._next_id, d.box)
                t.areas = deque([area(d.box)], maxlen=self.cfg.growth_frames)
                self._next_id += 1
                self.tracks.append(t)
                matched.add(t.id)
            else:
                best.box = d.box
                best.hits += 1
                best.missed = 0
                best.areas.append(area(d.box))
                matched.add(best.id)
        for t in list(self.tracks):
            if t.id not in matched:
                t.missed += 1
                if t.missed > self.cfg.max_missed:
                    self.tracks.remove(t)

    # ---- main entry ----------------------------------------------------
    def update(self, detections: list[Detection], frame_ok: bool = True,
               timestamp: Optional[float] = None) -> Status:
        c = self.cfg
        if not frame_ok:
            # Dark or frozen frame: say so instead of reporting "no smoke".
            return Status(Level.FEED_ISSUE, "Frame too dark or frozen",
                          first_detection_ts=self.first_detection_ts,
                          tracks=list(self.tracks))

        smoke_hi: list[Detection] = []
        smoke_lo = False
        fire: list[Detection] = []
        for d in detections:
            if self._excluded(d.box):
                self.log.append(f"ignored {d.cls} in exclusion zone")
                continue
            if d.cls == "smoke":
                if d.conf >= c.conf_alert:
                    smoke_hi.append(d)
                elif d.conf >= c.conf_watch:
                    smoke_lo = True
            elif d.cls == "fire" and d.conf >= c.conf_alert:
                fire.append(d)

        self._match(smoke_hi)
        self._fire_streak = self._fire_streak + 1 if fire else 0

        level, reason = Level.CLEAR, "No smoke detected"
        if smoke_lo:
            level, reason = Level.WATCH, "Low-confidence smoke, logged only"

        persistent = [t for t in self.tracks if t.hits >= c.persistence_frames]
        if persistent:
            level = Level.POSSIBLE_FIRE
            reason = f"Smoke seen in {c.persistence_frames}+ frames"
        elif self._fire_streak >= c.persistence_frames:
            level = Level.POSSIBLE_FIRE
            reason = "Fire seen without smoke track"

        # Rule 2: growth over growth_frames, or fire inside a persistent smoke box
        for t in persistent:
            grown = (t.hits >= c.growth_frames and len(t.areas) >= c.growth_frames
                     and t.areas[0] > 0
                     and t.areas[-1] / t.areas[0] >= c.growth_ratio)
            fire_inside = any(inter(t.box, f.box) > 0 for f in fire)
            if grown or fire_inside:
                level = Level.CONFIRMED_FIRE
                reason = ("Fire inside smoke" if fire_inside
                          else f"Smoke grew {int((c.growth_ratio - 1) * 100)}%+")
                break

        escalated = (level in (Level.POSSIBLE_FIRE, Level.CONFIRMED_FIRE)
                     and level > self._prev_level)
        self._prev_level = level
        if level >= Level.POSSIBLE_FIRE and self.first_detection_ts is None:
            self.first_detection_ts = timestamp
        return Status(level, reason, escalated, self.first_detection_ts,
                      list(self.tracks))
