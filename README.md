# SmokeWatch

Edge AI wildfire smoke early warning that runs on an NVIDIA Jetson.

A YOLOv8n model detects smoke and fire in camera frames. A rule engine turns
those detections into an alert level per camera, and a Streamlit dashboard shows
the result. Running on the Jetson means remote towers with weak internet do not
need to stream video to the cloud.

Course: Special Topics in Data Science (UMBC). Individual project.

## Docker image

`nvcr.io/nvidia/l4t-ml:r36.2.0-py3` (JetPack 6, includes PyTorch 2.1.0 and JupyterLab).
Development is done inside the running container `jetson_labs_dev`, which uses this image.
A `Dockerfile` based on the same image is included.

## Project layout

```
backend/
  rules.py      alert rules (persistence, growth, exclusion zones, uncertainty)
  detector.py   YOLO wrapper, frame quality check, drawing
  pipeline.py   frames in, status + annotated frame out, alerts saved
  store.py      SQLite alert log and user feedback
frontend/
  app.py        Streamlit dashboard
scripts/
  train.py           fine-tune YOLOv8n on D-Fire, export to TensorRT
  convert_weights.py rebuild Colab-trained weights for the Jetson's Ultralytics
  run_sequence.py    run the backend on a video, image folder or camera
tests/              unit tests for the rules and pipeline
data/dfire.yaml     dataset config
requirements.txt
Dockerfile
```

## Decision logic

| Level | When |
|---|---|
| Clear | no confident smoke |
| Watch | smoke with confidence 0.3 to 0.5, logged only |
| Possible Fire | **Rule 1:** smoke with confidence 0.5+ in the same area for 3+ frames |
| Confirmed Fire | **Rule 2:** that smoke grows 20%+ in area over 5 frames, or fire appears inside it. Stays confirmed while the smoke is visible |
| Feed issue | frame is too dark or frozen, so the system does not claim "no smoke" |

**Rule 3:** smoke inside user-defined exclusion zones (chimneys, dusty roads) is
logged but never raises an alert. A smoke track survives up to 2 missed frames.

## Setup on the Jetson

These steps create an isolated Python environment inside the course container, because
the container's own NumPy 2 clashes with its OpenCV and PyTorch.

```bash
cd /home/jetson/Desktop/jetson_labs
git clone https://github.com/Kenil-Nakrani/smokewatch.git
cd smokewatch

# the container only starts when the USB camera (/dev/video0) is plugged in
docker start jetson_labs_dev

# one-time: create the environment inside the container
docker exec -u $(id -u):$(id -g) -w /home/jetson_labs/smokewatch jetson_labs_dev \
  python3 -m venv --system-site-packages --without-pip .venv
docker exec -u $(id -u):$(id -g) -w /home/jetson_labs/smokewatch jetson_labs_dev \
  .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
```

Shortcut for the commands below (run it once per terminal session):

```bash
SW="docker exec -u $(id -u):$(id -g) -e PYTHONUNBUFFERED=1 -e YOLO_CONFIG_DIR=/tmp/Ultralytics -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb -w /home/jetson_labs/smokewatch jetson_labs_dev"
```

## Model

The model is YOLOv8n fine-tuned on D-Fire in Google Colab (T4 GPU, 32 epochs, 2 hours).
Colab uses a much newer PyTorch and Ultralytics than the Jetson, so the trained file is
rebuilt with the Jetson's own versions before use:

```bash
# copy best.pt from Colab into weights/, then:
$SW .venv/bin/python scripts/convert_weights.py weights/best.pt weights/best_jetson.pt
```

Validation results on the D-Fire val split (3,099 images):

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.756 | 0.434 |
| smoke | 0.810 | 0.494 |
| fire | 0.703 | 0.374 |

## Usage

```bash
$SW .venv/bin/python -m pytest                       # run the tests
$SW .venv/bin/python scripts/run_sequence.py data/figlib/20160604_FIRE_smer-tcs3-mobo-c --weights weights/best_jetson.pt
```

Dashboard (the container uses host networking, so no port mapping is needed):

```bash
$SW .venv/bin/python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open `http://<jetson-ip>:8501` in a browser on your laptop.

## Results on a real fire (backend on the Jetson)

HPWREN FIgLib sequence `20160604_FIRE_smer-tcs3-mobo-c`: 81 frames, one per minute,
from 40 minutes before to 40 minutes after ignition.

| Measure | Result |
|---|---|
| False alarms in the 40 minutes before ignition | 0 |
| First Watch (low-confidence smoke) | 1 minute after ignition |
| Possible Fire (Rule 1) | 4 minutes after ignition |
| Confirmed Fire (Rule 2) | 6 minutes after ignition |

Download the sequence into `data/figlib/` with:

```bash
mkdir -p data/figlib/20160604_FIRE_smer-tcs3-mobo-c && cd data/figlib/20160604_FIRE_smer-tcs3-mobo-c
wget -q -r -l1 -np -nd -A jpg https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20160604_FIRE_smer-tcs3-mobo-c/index.html
```

## Known issue

On the Jetson, Python sometimes crashes while shutting down ("corrupted size vs. prev_size"),
after all the work is done. `run_sequence.py` exits immediately when finished to avoid it,
and `PYTHONUNBUFFERED=1` makes sure output is printed even if a crash happens.

## Data

- Training: [D-Fire](https://github.com/wizbeans/FireDataset), about 21,500 labeled smoke and fire images,
  downloaded in Colab from the Kaggle copy `sayedgamal99/smoke-fire-detection-yolo`
  (14,122 train, 3,099 val, 4,306 test images).
- Testing: [HPWREN FIgLib](https://www.hpwren.ucsd.edu/FIgLib/), real camera sequences covering
  40 minutes before and after ignition.

Datasets and model weights are not stored in git (see `.gitignore`).

## Status

- [x] Alert rule engine with unit tests
- [x] Backend pipeline, SQLite alert log, command-line runner
- [x] Streamlit dashboard (first version)
- [x] Fine-tune the model on D-Fire (Colab)
- [x] Run the backend on a real fire sequence on the Jetson
- [ ] Evaluate on more FIgLib sequences
- [ ] TensorRT export and speed test on the Jetson
- [ ] Frontend polish and live camera demo
