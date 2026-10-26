"""Fine-tune YOLOv8n on D-Fire and optionally export a TensorRT engine.

Example:
    python scripts/train.py --data data/dfire.yaml --epochs 50
    python scripts/train.py --export weights/best.pt        # run on the Jetson
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dfire.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--export", metavar="WEIGHTS",
                    help="skip training, export these weights to a TensorRT engine")
    args = ap.parse_args()

    from ultralytics import YOLO  # lazy import

    if args.export:
        out = YOLO(args.export).export(format="engine", half=True, imgsz=args.imgsz)
        print("Exported:", out)
        return

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project="outputs/train", name="smokewatch")
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/best.pt")
    print("Saved weights/best.pt")


if __name__ == "__main__":
    main()
