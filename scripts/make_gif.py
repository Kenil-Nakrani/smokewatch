"""Make an animated GIF of SmokeWatch running on a FIgLib fire sequence.

Each frame shows the detection boxes plus a bar with the alert level and the
minutes since ignition (read from the FIgLib file names).

Example:
    python scripts/make_gif.py data/figlib/20160604_FIRE_smer-tcs3-mobo-c docs/images/detection.gif --start -10
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detector import LEVEL_COLORS, YoloDetector  # noqa: E402
from backend.pipeline import IMG_EXT, SmokeWatch  # noqa: E402
from backend.rules import Config  # noqa: E402


def offset_seconds(name: str):
    m = re.search(r"_([+-]?\d+)\.\w+$", name)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="FIgLib sequence folder")
    ap.add_argument("out", help="output .gif file")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--start", type=float, default=None,
                    help="first minute to include, e.g. -10 for 10 minutes before ignition")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    files = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"No images found in {args.folder}")

    sw = SmokeWatch(YoloDetector(args.weights, conf=Config().conf_watch), Config(),
                    save_alerts=False)
    frames = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        off = offset_seconds(f.name)
        t = float(off if off is not None else i * 60)
        status, _, annotated = sw.process(img, i, timestamp=t)  # every frame feeds the rules
        if args.start is not None and t / 60 < args.start:
            continue

        h, w = annotated.shape[:2]
        small = cv2.resize(annotated, (args.width, int(h * args.width / w)))
        bar = 34
        cv2.rectangle(small, (0, 0), (small.shape[1], bar), LEVEL_COLORS[status.level], -1)
        text = f"t = {t / 60:+.0f} min    {status.level.label.upper()}"
        cv2.putText(small, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

    if not frames:
        sys.exit("No frames left after --start")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames = [fr.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for fr in frames]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 / args.fps), loop=0)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"saved {args.out}: {len(frames)} frames, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
