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
  train.py          fine-tune YOLOv8n on D-Fire, export to TensorRT
  run_sequence.py   run the backend on a video, image folder or camera
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
| Confirmed Fire | **Rule 2:** that smoke grows 20%+ in area over 5 frames, or fire appears inside it |
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

# one-time: create the environment inside the container
docker exec -u $(id -u):$(id -g) -w /home/jetson_labs/smokewatch jetson_labs_dev \
  python3 -m venv --system-site-packages --without-pip .venv
docker exec -u $(id -u):$(id -g) -w /home/jetson_labs/smokewatch jetson_labs_dev \
  .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
```

Short alias for the commands below:

```bash
alias sw='docker exec -u $(id -u):$(id -g) -e YOLO_CONFIG_DIR=/tmp/Ultralytics -w /home/jetson_labs/smokewatch jetson_labs_dev .venv/bin/python'
```

## Usage

```bash
sw -m pytest                                        # run the tests
sw scripts/train.py --data data/dfire.yaml          # fine-tune (or train on Colab and copy best.pt)
sw scripts/train.py --export weights/best.pt        # TensorRT engine for the Jetson
sw scripts/run_sequence.py data/figlib/seq1 --weights weights/best.pt
```

Dashboard (the container uses host networking, so no port mapping is needed):

```bash
docker exec -u $(id -u):$(id -g) -w /home/jetson_labs/smokewatch jetson_labs_dev \
  .venv/bin/python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://<jetson-ip>:8501` in a browser on your laptop.

## Data

- Training: [D-Fire](https://github.com/wizbeans/FireDataset), about 21,500 labeled smoke and fire images.
- Testing: [HPWREN FIgLib](https://www.hpwren.ucsd.edu/FIgLib/), real camera sequences covering
  40 minutes before and after ignition.

Datasets and model weights are not stored in git (see `.gitignore`).

## Status

- [x] Alert rule engine with unit tests
- [x] Backend pipeline, SQLite alert log, command-line runner
- [x] Streamlit dashboard
- [ ] Fine-tune the model on D-Fire
- [ ] TensorRT export and speed test on the Jetson
- [ ] Evaluate on FIgLib sequences
