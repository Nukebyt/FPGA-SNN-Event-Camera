"""Run the phone clips (data/user_clips and its originals/ folder) through the board and the model; print answers per window.

    python scripts/user_clip_test.py --connect 192.168.137.200 --out docs/results/user_clips_board.json

The true action is taken from the file name (hand_waving*, hand_clapping*, boxing*, walking*, jogging*, running*). Every frame is also
run through the bit-accurate model and compared with the board (any difference is counted and shown).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.ood_test import FIELDS, frames_of  # noqa: E402
from sw.host import snnproto as P  # noqa: E402
from sw.host.snnboard_sim import SimBoard  # noqa: E402

TRUTH = {"hand_waving": "hand waving", "hand_clapping": "hand clapping", "boxing": "boxing", "walking": "walking",
         "jogging": "jogging", "running": "running"}


def truth_of(name):
    for k, v in TRUTH.items():
        if name.startswith(k):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = os.path.join(ROOT, "data", "user_clips")
    files = sorted(glob.glob(os.path.join(d, "*.mp4")) + glob.glob(os.path.join(d, "originals", "*.mp4")),
                   key=lambda f: (os.path.basename(f).replace("_crop", ""), "originals" not in f))
    sim = SimBoard(os.path.join(ROOT, "rtl", "weights", "kth6"))
    client = None
    if a.connect:
        client = P.Client(a.connect, 5002, timeout=20)
        client.hello()
    rows, mism_total, frames_total = [], 0, 0
    for f in files:
        name = os.path.basename(f)[:-4]
        truth = truth_of(name)
        if client:
            client.reset()
        sim._reset_state()
        preds, scores, mism, n, ev = [], np.zeros(6, np.int64), 0, 0, []
        for fr in frames_of("video", f, 0, 10 ** 6):
            ref = sim.process(fr, n + 1)
            res = ref
            if client:
                client.send_frame(fr)
                res = client.recv_result()
                if any(getattr(res, k) != getattr(ref, k) for k in FIELDS):
                    mism += 1
            if res.windows > len(preds):
                preds.append(res.pred)
                scores += np.array(res.win_score, np.int64)
            ev.append(res.events)
            n += 1
        hist = Counter(P.CLASSES[p] for p in preds)
        ans = P.CLASSES[int(np.argmax(scores))] if preds else "-"
        rows.append({"clip": name, "cropped": name.endswith("_crop"), "truth": truth, "frames": n, "windows": len(preds), "clip_answer": ans,
                     "correct": ans == truth, "window_hist": dict(hist), "truth_window_share": hist.get(truth, 0) / max(1, len(preds)),
                     "mean_events_pct": float(np.mean(ev) / 16384 * 100), "mismatching_frames": mism})
        mism_total += mism
        frames_total += n
        print("%-24s true %-13s -> %-13s %s  windows %s  events %.1f%%%s" % (
            name, truth, ans, "OK " if ans == truth else "BAD", dict(hist), np.mean(ev) / 16384 * 100,
            ("  board!=model on %d" % mism) if mism else ""))
    if client:
        client.close()
    print("frames %d, board != model on %d" % (frames_total, mism_total))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"backend": "board " + a.connect if a.connect else "model", "rows": rows, "frames": frames_total,
                       "mismatching_frames": mism_total}, fh, indent=1)


if __name__ == "__main__":
    main()
