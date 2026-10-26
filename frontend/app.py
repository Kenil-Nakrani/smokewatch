"""SmokeWatch dashboard (Streamlit).

Run:  streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
Then open http://<jetson-ip>:8501 in a browser on your laptop.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import store  # noqa: E402
from backend.detector import YoloDetector  # noqa: E402
from backend.pipeline import SmokeWatch, iter_frames  # noqa: E402
from backend.rules import Config, Level  # noqa: E402

BADGE = {
    Level.CLEAR: "#2e9e4f",
    Level.WATCH: "#d4a600",
    Level.POSSIBLE_FIRE: "#e8710a",
    Level.CONFIRMED_FIRE: "#c62828",
    Level.FEED_ISSUE: "#757575",
}

st.set_page_config(page_title="SmokeWatch", layout="wide")
st.title("SmokeWatch")
st.caption("Edge AI wildfire smoke early warning")


@st.cache_resource
def load_detector(weights: str, conf: float):
    return YoloDetector(weights, conf=conf)


def badge(level: Level, reason: str) -> str:
    return (f"<div style='background:{BADGE[level]};color:white;padding:14px 18px;"
            f"border-radius:8px;font-size:22px;font-weight:700'>{level.label.upper()}"
            f"<span style='font-size:14px;font-weight:400;margin-left:12px'>{reason}</span></div>")


def parse_zones(text: str):
    zones = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            x1, y1, x2, y2 = (float(v) for v in line.split(","))
            zones.append((x1, y1, x2, y2))
    return zones


# ---------------- sidebar controls ----------------
with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Image folder (e.g. FIgLib)", "Upload video", "Live camera"])
    weights = st.text_input("Model weights", "weights/best.pt")
    folder = upload = None
    if mode.startswith("Image"):
        folder = st.text_input("Folder path", "data/figlib/sequence1")
    elif mode == "Upload video":
        upload = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    else:
        cam = st.number_input("Camera index", 0, 10, 0)
        max_frames = st.number_input("Frames to process", 10, 5000, 300)

    st.header("Alert settings")
    conf = st.slider("Alert confidence", 0.3, 0.9, 0.5, 0.05)
    persistence = st.slider("Persistence (frames)", 2, 10, 3)
    step = st.number_input("Process every Nth frame", 1, 100, 1)
    sec_per_frame = st.number_input("Seconds between processed frames", 0.01, 600.0,
                                    60.0 if mode.startswith("Image") else 1.0)
    zones_text = st.text_area("Exclusion zones (x1,y1,x2,y2 per line)", "")
    start = st.button("Start", type="primary")

# ---------------- run ----------------
status_box = st.empty()
image_box = st.empty()
col1, col2 = st.columns(2)
timer_box = col1.empty()
chart_box = col2.empty()

if start:
    if not Path(weights).exists():
        st.error(f"Weights not found: {weights}")
        st.stop()
    try:
        zones = parse_zones(zones_text)
    except ValueError:
        st.error("Exclusion zones must look like: 100,50,300,200")
        st.stop()

    if mode.startswith("Image"):
        source = folder
    elif mode == "Upload video":
        if upload is None:
            st.error("Upload a video first.")
            st.stop()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
        tmp.write(upload.read())
        tmp.close()
        source = tmp.name
    else:
        source = str(int(cam))

    cfg = Config(conf_alert=conf, persistence_frames=persistence, exclusion_zones=zones)
    sw = SmokeWatch(load_detector(weights, cfg.conf_watch), cfg,
                    source_name=Path(source).name or source)
    areas = []
    for n, (idx, frame) in enumerate(iter_frames(source, int(step))):
        if mode == "Live camera" and n >= max_frames:
            break
        ts = n * sec_per_frame
        status, dets, annotated = sw.process(frame, idx, timestamp=ts)
        status_box.markdown(badge(status.level, status.reason), unsafe_allow_html=True)
        image_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        smoke_area = max([(d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                          for d in dets if d.cls == "smoke"], default=0)
        areas.append(smoke_area)
        if status.first_detection_ts is not None:
            mins = (ts - status.first_detection_ts) / 60
            timer_box.metric("Minutes since first detection", f"{mins:.1f}")
        else:
            timer_box.metric("Minutes since first detection", "none yet")
        if n % 5 == 0:
            chart_box.line_chart({"Smoke area (px)": areas})
    chart_box.line_chart({"Smoke area (px)": areas})
    st.success(f"Done. Processed {len(areas)} frames.")

# ---------------- alert log with feedback ----------------
st.subheader("Alert log")
alerts = store.list_alerts()
if not alerts:
    st.info("No alerts yet.")
for a in alerts[:20]:
    c1, c2, c3 = st.columns([1, 3, 2])
    if a["snapshot"] and Path(a["snapshot"]).exists():
        c1.image(a["snapshot"], use_container_width=True)
    c2.markdown(f"**{a['level']}** at frame {a['frame_index']} ({a['source']})  \n{a['reason']}")
    c3.write(f"Feedback: {a['feedback'] or 'none'}")
    if c3.button("Confirm", key=f"c{a['id']}"):
        store.set_feedback(a["id"], "confirmed")
        st.rerun()
    if c3.button("False alarm", key=f"f{a['id']}"):
        store.set_feedback(a["id"], "false_alarm")
        st.rerun()
