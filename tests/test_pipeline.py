import numpy as np

from backend import store
from backend.detector import frame_ok
from backend.pipeline import SmokeWatch, iter_frames
from backend.rules import Detection, Level


class FakeDetector:
    def detect(self, frame):
        return [Detection("smoke", 0.9, (10, 10, 60, 60))]


def frame(i, value=100):
    return np.full((120, 160, 3), value + i, dtype=np.uint8)


def test_frame_ok_rejects_dark_and_frozen():
    assert frame_ok(frame(0))
    assert not frame_ok(frame(0, value=2))
    assert not frame_ok(frame(0), frame(0))
    assert frame_ok(frame(1), frame(0))


def test_pipeline_reaches_possible_fire():
    sw = SmokeWatch(FakeDetector(), save_alerts=False)
    for i in range(3):
        status, dets, img = sw.process(frame(i), i)
    assert status.level == Level.POSSIBLE_FIRE
    assert len(dets) == 1 and img.shape == (120, 160, 3)


def test_pipeline_dark_frame_gives_feed_issue():
    sw = SmokeWatch(FakeDetector(), save_alerts=False)
    status, dets, _ = sw.process(frame(0, value=1), 0)
    assert status.level == Level.FEED_ISSUE and dets == []


def test_store_roundtrip(tmp_path):
    db = tmp_path / "a.db"
    aid = store.add_alert("cam1", 5, "Possible Fire", "test", "x.jpg", db=db)
    store.set_feedback(aid, "false_alarm", db=db)
    rows = store.list_alerts(db=db)
    assert rows[0]["feedback"] == "false_alarm" and rows[0]["frame_index"] == 5


def test_iter_frames_reads_image_folder(tmp_path):
    import cv2

    for i in range(4):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), frame(i * 10))
    assert [i for i, _ in iter_frames(str(tmp_path))] == [0, 1, 2, 3]
    assert [i for i, _ in iter_frames(str(tmp_path), step=2)] == [0, 2]
