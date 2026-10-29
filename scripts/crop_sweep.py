"""How much zoom and what stabilisation helps the phone clips? Model only (bit-exact to the board), first 320 frames of each original.

    python scripts/crop_sweep.py --out docs/results/crop_sweep.json

Variants: crop width in source pixels (854 = the whole picture, no crop) x camera-shake removal off/on. The crop centre is fixed
(430, 235). The answer to look at is the share of 1-second windows that name the true action, and the share of pixels that are events.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.crop_user_clips import stabilizer  # noqa: E402
from scripts.user_clip_test import truth_of  # noqa: E402
from sw.host import snnproto as P  # noqa: E402
from sw.host.snnboard_sim import SimBoard  # noqa: E402


def variant_frames(path, width, stab, cx, cy, nmax):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rs = None if abs(fps - 30.0) < 0.05 else P.Resampler30(fps)
    fix = stabilizer() if stab else (lambda fr: fr)
    k = 0
    while k < nmax:
        ok, img = cap.read()
        if not ok:
            break
        img = fix(img)
        if width < img.shape[1]:
            h = width * 3 // 4
            x0 = min(max(cx - width // 2, 0), img.shape[1] - width)
            y0 = min(max(cy - h // 2, 0), img.shape[0] - h)
            img = img[y0:y0 + h, x0:x0 + width]
        f = P.prepare_frame(img)
        for g in ([f] if rs is None else rs.push(f)):
            if k >= nmax:
                break
            yield g
            k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--widths", default="854,640,520,420,320")
    a = ap.parse_args()
    sim = SimBoard(os.path.join(ROOT, "rtl", "weights", "kth6"))
    files = sorted(glob.glob(os.path.join(ROOT, "data", "user_clips", "originals", "*.mp4")))
    rows = []
    for width in [int(w) for w in a.widths.split(",")]:
        for stab in (False, True):
            tot_w = tot_ok = 0
            clip_ok, evs, per = 0, [], {}
            for f in files:
                name = os.path.basename(f)[:-4]
                truth = truth_of(name)
                sim._reset_state()
                preds, scores, n = [], np.zeros(6, np.int64), 0
                for fr in variant_frames(f, width, stab, 430, 235, a.frames):
                    r = sim.process(fr, n + 1)
                    n += 1
                    evs.append(r.events)
                    if r.windows > len(preds):
                        preds.append(r.pred)
                        scores += np.array(r.win_score, np.int64)
                hist = Counter(P.CLASSES[p] for p in preds)
                ans = P.CLASSES[int(np.argmax(scores))]
                tot_w += len(preds)
                tot_ok += hist.get(truth, 0)
                clip_ok += ans == truth
                per[name] = {"answer": ans, "hist": dict(hist)}
            row = {"width": width, "stabilised": stab, "clips_correct": clip_ok, "clips": len(files), "windows": tot_w,
                   "windows_correct": tot_ok, "mean_events_pct": float(np.mean(evs) / 16384 * 100), "per_clip": per}
            rows.append(row)
            print("width %3d  stabilised %-5s  clips right %d/%d  windows right %3d/%3d (%.0f%%)  events %.1f%%" % (
                width, stab, clip_ok, len(files), tot_ok, tot_w, 100 * tot_ok / max(1, tot_w), row["mean_events_pct"]), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"frames_per_clip": a.frames, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
