"""Turn a folder of camera images (e.g. a FIgLib fire sequence) into an MP4 video.

Example:
    python scripts/make_video.py data/figlib/20160604_FIRE_smer-tcs3-mobo-c data/videos/fire.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of images, played in file-name order")
    ap.add_argument("out", help="output .mp4 file")
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second of video")
    ap.add_argument("--width", type=int, default=1280, help="output width in pixels")
    args = ap.parse_args()

    files = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"No images found in {args.folder}")

    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    size = (args.width, int(h * args.width / w) // 2 * 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    if not writer.isOpened():
        sys.exit("Could not open the video writer")

    n = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(cv2.resize(img, size))
            n += 1
    writer.release()
    print(f"saved {args.out}: {n} frames, {n / args.fps:.0f} seconds at {args.fps:g} fps, {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
