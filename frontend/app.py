"""SmokeWatch dashboard (Streamlit).

Run:  streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
Then open http://<jetson-ip>:8501 in a browser on your laptop.
"""
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import store  # noqa: E402
from backend.detector import YoloDetector  # noqa: E402
from backend.pipeline import IMG_EXT, SmokeWatch  # noqa: E402
from backend.rules import Config, Level  # noqa: E402

WEIGHTS = ROOT / "weights" / "best.pt"
FIGLIB = ROOT / "data" / "figlib"
ALERT_LEVELS = (Level.POSSIBLE_FIRE, Level.CONFIRMED_FIRE)
BADGE = {
    Level.CLEAR: "#2e9e4f",
    Level.WATCH: "#d4a600",
    Level.POSSIBLE_FIRE: "#e8710a",
    Level.CONFIRMED_FIRE: "#c62828",
    Level.FEED_ISSUE: "#757575",
}


# ---------------- helpers ----------------
@st.cache_resource
def load_detector(weights: str):
    return YoloDetector(weights, conf=Config().conf_watch)


def offset_seconds(name: str):
    """FIgLib file names end in seconds from ignition, e.g. 1465063268_-02400.jpg."""
    m = re.search(r"_([+-]?\d+)\.\w+$", name)
    return int(m.group(1)) if m else None


def list_sequences() -> list[str]:
    if not FIGLIB.exists():
        return []
    return sorted(d.name for d in FIGLIB.iterdir()
                  if d.is_dir() and any(f.suffix.lower() in IMG_EXT for f in d.iterdir()))


def sequence_frames(folder: Path):
    """Yield (frame, seconds from ignition) for every image in a FIgLib folder."""
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is not None:
            off = offset_seconds(f.name)
            yield img, float(off if off is not None else i * 60)


def video_frames(path: str, per_second: int = 2):
    """Yield (frame, seconds into the video), about 2 frames per second of video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / per_second))
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield img, i / fps
            i += 1
    finally:
        cap.release()


def shrink(img, width: int = 1280):
    h, w = img.shape[:2]
    return img if w <= width else cv2.resize(img, (width, int(h * width / w)))


def badge(level: Level, reason: str) -> str:
    return (f"<div style='background:{BADGE[level]};color:white;padding:14px 18px;"
            f"border-radius:8px;font-size:24px;font-weight:700'>{level.label.upper()}"
            f"<span style='font-size:15px;font-weight:400;margin-left:14px'>{reason}</span></div>")


def fmt_time(t, is_sequence: bool) -> str:
    if t is None:
        return "none yet"
    return f"{t / 60:+.0f} min" if is_sequence else f"{t:.0f} s"


# ---------------- page ----------------
st.set_page_config(page_title="SmokeWatch", layout="wide")
st.title("SmokeWatch")
st.caption("Wildfire smoke early warning, running on an NVIDIA Jetson")

with st.sidebar:
    st.header("1. Choose input")
    source = st.radio("Source", ["Real fire sequence", "Upload video"])
    choice = upload = None
    if source == "Real fire sequence":
        seqs = list_sequences()
        if seqs:
            choice = st.selectbox("Sequence", seqs)
        else:
            st.warning("No sequences found in data/figlib.")
    else:
        upload = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

    st.header("2. Sensitivity")
    conf = st.slider("Alert confidence", 0.3, 0.9, 0.5, 0.05,
                     help="Lower = alerts sooner, but more false alarms.")

    start = st.button("Start", type="primary", width="stretch")

status_box = st.empty()
m1, m2, m3 = st.columns(3)
time_box, first_box, speed_box = m1.empty(), m2.empty(), m3.empty()
image_box = st.empty()
chart_title = st.empty()
chart_box = st.empty()
chart_note = st.empty()

if start:
    if not WEIGHTS.exists():
        st.error(f"Model not found: {WEIGHTS.relative_to(ROOT)}")
        st.stop()

    is_sequence = source == "Real fire sequence"
    if is_sequence:
        if not choice:
            st.error("Pick a sequence first.")
            st.stop()
        frames, name = sequence_frames(FIGLIB / choice), choice
        time_label, axis = "Minutes since fire started", "Minutes since fire started"
    else:
        if upload is None:
            st.error("Upload a video first.")
            st.stop()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
        tmp.write(upload.read())
        tmp.close()
        frames, name = video_frames(tmp.name), Path(upload.name).stem
        time_label, axis = "Video time", "Seconds into video"

    sw = SmokeWatch(load_detector(str(WEIGHTS)), Config(conf_alert=conf), source_name=name)
    rows, first_alert, speeds = [], None, []
    chart_title.markdown("**Smoke confidence over time**")

    for n, (frame, t) in enumerate(frames):
        t0 = time.perf_counter()
        status, dets, annotated = sw.process(frame, n, timestamp=t)
        speeds.append(time.perf_counter() - t0)

        if status.level in ALERT_LEVELS and first_alert is None:
            first_alert = t
        smoke = max((d.conf for d in dets if d.cls == "smoke"), default=0.0)
        rows.append({axis: t / 60 if is_sequence else t,
                     "Smoke confidence": smoke, "Alert threshold": conf})

        status_box.markdown(badge(status.level, status.reason), unsafe_allow_html=True)
        time_box.metric(time_label, fmt_time(t, is_sequence))
        first_box.metric("First alert", fmt_time(first_alert, is_sequence))
        recent = speeds[-10:]
        speed_box.metric("Speed", f"{len(recent) / sum(recent):.1f} frames/sec")
        image_box.image(cv2.cvtColor(shrink(annotated), cv2.COLOR_BGR2RGB),
                        width="stretch")
        if n % 3 == 0:
            chart_box.line_chart(pd.DataFrame(rows).set_index(axis))

    if rows:
        chart_box.line_chart(pd.DataFrame(rows).set_index(axis))
        if is_sequence:
            chart_note.caption("0 on the time axis is the moment the fire started.")
        st.success(f"Done. Processed {len(rows)} frames.")
    else:
        st.warning("No frames found in this input.")

# ---------------- alert log ----------------
st.divider()
head, clear = st.columns([5, 1])
head.subheader("Alert log")
if clear.button("Clear log"):
    store.clear_alerts()
    st.rerun()

alerts = store.list_alerts()[:10]
if not alerts:
    st.info("No alerts yet.")
for a in alerts:
    c1, c2, c3 = st.columns([2, 4, 2])
    if a["snapshot"] and Path(a["snapshot"]).exists():
        c1.image(a["snapshot"], width="stretch")
    c2.markdown(f"**{a['level']}**  \n{a['reason']}  \nFrame {a['frame_index']} of {a['source']}")
    c3.write(f"Feedback: {a['feedback'] or 'none'}")
    if c3.button("Confirm", key=f"c{a['id']}"):
        store.set_feedback(a["id"], "confirmed")
        st.rerun()
    if c3.button("False alarm", key=f"f{a['id']}"):
        store.set_feedback(a["id"], "false_alarm")
        st.rerun()
