"""Run the SmokeWatch backend on a video, a folder of images (e.g. a FIgLib
sequence) or a camera index, and print the alert level as it changes.

Example:
    python scripts/run_sequence.py data/figlib/sequence1 --weights weights/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detector import YoloDetector  # noqa: E402
from backend.pipeline import SmokeWatch, iter_frames  # noqa: E402
from backend.rules import Config  # noqa: E402


def parse_zone(text: str):
    x1, y1, x2, y2 = (float(v) for v in text.split(","))
    return (x1, y1, x2, y2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file, image folder, or camera index like 0")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--sec-per-frame", type=float, default=60.0,
                    help="seconds between processed frames (FIgLib is about 60)")
    ap.add_argument("--conf", type=float, default=0.5, help="alert confidence")
    ap.add_argument("--persistence", type=int, default=3)
    ap.add_argument("--zone", action="append", default=[], type=parse_zone,
                    help="exclusion zone x1,y1,x2,y2 (repeatable)")
    ap.add_argument("--no-save", action="store_true", help="do not write alerts or snapshots")
    args = ap.parse_args()

    cfg = Config(conf_alert=args.conf, persistence_frames=args.persistence,
                 exclusion_zones=args.zone)
    name = Path(args.source).name or args.source
    sw = SmokeWatch(YoloDetector(args.weights, conf=cfg.conf_watch), cfg,
                    source_name=name, save_alerts=not args.no_save)

    last, n = None, 0
    for idx, frame in iter_frames(args.source, args.step):
        ts = (idx / args.step) * args.sec_per_frame
        status, dets, _ = sw.process(frame, idx, timestamp=ts)
        n += 1
        if status.level != last:
            print(f"frame {idx:6d}  t={ts / 60:6.1f} min  {status.level.label:15s} {status.reason}")
            last = status.level
    first = sw.engine.first_detection_ts
    print(f"\nProcessed {n} frames. First alert at "
          f"{'never' if first is None else f'{first / 60:.1f} min'}.")


if __name__ == "__main__":
    main()
