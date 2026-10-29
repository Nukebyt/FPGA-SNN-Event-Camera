"""Crop and zoom phone clips so the person fills the picture the way KTH does (whole body, 4:3).

    python scripts/crop_user_clips.py --stabilize                # data/user_clips/*.mp4 -> *_crop.mp4, originals moved to originals/
    python scripts/crop_user_clips.py --width 640 --stabilize --from-originals

The crop is one fixed window (centre cx, cy and width in source pixels; height = width * 3/4) cut from every frame and scaled to
320x240, so the camera stays 'static' like in KTH and nothing is tracked. The frames are written at 30 fps: the phone clips are
29.94-29.97 fps, i.e. 0.2% off, and re-timing them is simpler and cleaner than blending. Nothing else is changed (no brightness
correction), so the network sees what the phone recorded, only closer. With --stabilize each frame is first shifted back onto
the first frame (phase correlation of the blurred grey picture; the corridor dominates the picture, the person is small): handheld
phone clips drift 15-25 px and jitter 0.4-0.6 px per frame, and a tight crop turns that into events on every wall edge.
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stabilizer():
    """Return f(frame) -> frame shifted onto the first frame it saw (translation only)."""
    state = {}

    def grey(fr):
        return cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype("float32")

    def fix(fr):
        g = grey(fr)
        if "ref" not in state:
            state["ref"], state["win"] = g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
            return fr
        (dx, dy), _ = cv2.phaseCorrelate(state["ref"], g, state["win"])
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(fr, m, (fr.shape[1], fr.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return fix


def crop_clip(src, dst, cx, cy, w, stabilize=False):
    h = w * 3 // 4
    cap = cv2.VideoCapture(src)
    fw, fh = int(cap.get(3)), int(cap.get(4))
    if w > fw or h > fh:
        raise SystemExit("crop window %dx%d is larger than the %dx%d picture" % (w, h, fw, fh))
    x0 = min(max(cx - w // 2, 0), fw - w)  # the window is pushed back inside the picture, as in scripts/crop_sweep.py
    y0 = min(max(cy - h // 2, 0), fh - h)
    fix = stabilizer() if stabilize else (lambda fr: fr)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = fix(fr)
        out.write(cv2.resize(fr[y0:y0 + h, x0:x0 + w], (320, 240), interpolation=cv2.INTER_AREA if w > 320 else cv2.INTER_CUBIC))
        n += 1
    out.release()
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "user_clips"))
    ap.add_argument("--cx", type=int, default=430, help="crop centre x in the source picture")
    ap.add_argument("--cy", type=int, default=235, help="crop centre y")
    ap.add_argument("--width", type=int, default=640, help="crop width; the height is width * 3 / 4")
    ap.add_argument("--stabilize", action="store_true", help="remove hand-held camera shake before cropping")
    ap.add_argument("--from-originals", action="store_true", help="redo the crops from the files in originals/ (after a first run)")
    a = ap.parse_args()
    orig = os.path.join(a.dir, "originals")
    os.makedirs(orig, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(orig if a.from_originals else a.dir, "*.mp4"))):
        stem = os.path.basename(f)[:-4]
        if stem.endswith("_crop"):
            continue
        dst = os.path.join(a.dir, stem + "_crop.mp4")
        n = crop_clip(f, dst, a.cx, a.cy, a.width, a.stabilize)
        if not a.from_originals:
            shutil.move(f, os.path.join(orig, os.path.basename(f)))
        print("%s -> %s: %d frames, crop centre (%d, %d) width %d%s" % (stem, os.path.basename(dst), n, a.cx, a.cy, a.width, ", stabilised" if a.stabilize else ""))


if __name__ == "__main__":
    main()
