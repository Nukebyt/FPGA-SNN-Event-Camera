"""Stress test with footage that is NOT from KTH, through the real board (or the models).

    python scripts/ood_test.py --connect 192.168.137.200 --out docs/results/ood_board.json

Sources (all on this PC, from the author's other projects): thermal pedestrian sequences of CDnet 2014 (park, corridor, lakeSide,
library, diningRoom; static thermal camera, people walking or sitting, different scale and look) and hand-in-a-dark-room clips of the
night-vision project (1080p, ISO 2000, very noisy). They are processed exactly like any clip (grey, squashed to 128x128, 30 fps).
Labels are WEAK: where people simply walk past the camera the expected class is walking; where the footage shows none of the six
actions (a person sitting, a hand moving in the dark) there is no correct answer and the table shows what the network says anyway.
With --check every frame is also run through the bit-accurate model and compared with the board's answer.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sw.host import snnproto as P  # noqa: E402
from sw.host.snnboard_sim import SimBoard  # noqa: E402

CD = r"F:\Projects\FPGA-Thermal-NUC_Motion\data\cdnet\thermal"
NV = r"F:\Projects\FPGA-Adaptive-Night-Vision\data\own_clips"
KTH = os.path.join(ROOT, "data", "kth", "videos")
FIELDS = ["pred", "windows", "wcount", "spk1", "spk2", "events", "cur", "score", "win_score"]

# name, kind, path, first frame, number of frames, weak label (None = none of the six actions), note
CLIPS = [
    ("KTH walking (reference)", "video", os.path.join(KTH, "person22_walking_d1_uncomp.avi"), 0, 600, "walking", "in distribution"),
    ("KTH boxing (reference)", "video", os.path.join(KTH, "person22_boxing_d1_uncomp.avi"), 0, 360, "boxing", "in distribution"),
    ("thermal park", "jpgs", os.path.join(CD, "park", "input"), 0, 600, "walking", "pedestrians on a path, side view, thermal"),
    ("thermal corridor", "jpgs", os.path.join(CD, "corridor", "input"), 400, 900, "walking", "people far away in a corridor, thermal"),
    ("thermal lakeSide", "jpgs", os.path.join(CD, "lakeSide", "input"), 1600, 800, "walking", "small distant people, thermal"),
    ("thermal library", "jpgs", os.path.join(CD, "library", "input"), 450, 900, None, "a person sitting down: none of the six actions"),
    ("thermal diningRoom", "jpgs", os.path.join(CD, "diningRoom", "input"), 600, 800, None, "a person moving about indoors"),
    ("hand, fast (dark room)", "video", os.path.join(NV, "move_fast_hand_iso2000_1-60.mp4"), 0, 429, None, "hand moving, 1080p, ISO 2000 noise"),
    ("hand, slow (dark room)", "video", os.path.join(NV, "move_slow_hand_iso2000_1-60.mp4"), 0, 700, None, "hand moving slowly, ISO 2000 noise"),
    ("empty static scene (dark)", "video", os.path.join(NV, "static_dimroom_iso2000_1-60.mp4"), 0, 400, None, "nothing moves: only sensor noise"),
]


def frames_of(kind, path, start, n):
    if kind == "jpgs":
        files = sorted(glob.glob(os.path.join(path, "*.jpg")))[start:start + n]
        for f in files:
            yield P.prepare_frame(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rs = None if abs(fps - 30.0) < 0.05 else P.Resampler30(fps)
    k = 0
    while k < n:
        ok, img = cap.read()
        if not ok:
            break
        f = P.prepare_frame(img)
        for g in ([f] if rs is None else rs.push(f)):
            if k >= n:
                break
            yield g
            k += 1
    cap.release()


def run(args):
    sim = SimBoard(os.path.join(ROOT, "rtl", "weights", "kth6"))
    client = None
    if args.connect:
        host, _, port = args.connect.partition(":")
        client = P.Client(host, int(port) if port else 5002, timeout=20)
        client.hello()
    rows, mism_total, frames_total = [], 0, 0
    for name, kind, path, start, n, weak, note in CLIPS:
        if not os.path.exists(path):
            print("skip (missing):", name)
            continue
        if client:
            client.reset()
        sim._reset_state()
        preds, scores, mism, nfr, ev = [], np.zeros(6, np.int64), 0, 0, []
        t0 = time.time()
        for f in frames_of(kind, path, start, n):
            ref = sim.process(f, nfr + 1)
            if client:
                client.send_frame(f)
                res = client.recv_result()
                if args.check and any(getattr(res, k) != getattr(ref, k) for k in FIELDS):
                    mism += 1
            else:
                res = ref
            if res.windows > len(preds):
                preds.append(res.pred)
                scores += np.array(res.win_score, np.int64)
            ev.append(res.events)
            nfr += 1
        hist = Counter(P.CLASSES[p] for p in preds)
        clip_class = P.CLASSES[int(np.argmax(scores))] if preds else "-"
        frac = (hist.get(weak, 0) / len(preds)) if (weak and preds) else None
        rows.append({"clip": name, "note": note, "weak_label": weak, "frames": nfr, "windows": len(preds), "clip_answer": clip_class,
                     "window_hist": dict(hist), "weak_label_fraction": frac, "mean_events_pct": float(np.mean(ev) / 16384 * 100) if ev else 0.0,
                     "mismatching_frames": mism, "seconds": round(time.time() - t0, 1)})
        mism_total += mism
        frames_total += nfr
        print("%-26s %4d frames %3d windows -> %-14s hist %s%s" % (name, nfr, len(preds), clip_class, dict(hist),
              ("   board!=model on %d frames" % mism) if mism else ""))
    if client:
        client.close()
    out = {"backend": "board " + args.connect if args.connect else "model", "rows": rows, "frames": frames_total, "mismatching_frames": mism_total}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print("frames %d, board != model on %d" % (frames_total, mism_total))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect")
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--out")
    run(ap.parse_args())
