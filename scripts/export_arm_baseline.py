"""Files for sw/arm_baseline.c and the check of its results against the Python models.

    python scripts/export_arm_baseline.py export --video data/kth/videos/person22_boxing_d1_uncomp.avi --frames 300 --out work/arm
    python scripts/export_arm_baseline.py compare --out work/arm --res work/arm/res_sparse.bin

export writes w.bin (int8 w1, w2, fc in numpy order), lut.bin, frames.u8 and expected.bin (what the bit-accurate models answer per
frame: spk1, spk2, events, 6 class currents, pred, windows, wcount as 12 int32). compare requires the C program's res.bin to be equal.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def export(video, frames, out, weights):
    from model.log_lut import LUT
    from model.snn_int import IntSNN
    from scripts.snn_viewer import frames_30fps
    from sw.host.snnboard_sim import SimBoard
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    net = IntSNN.load(weights)
    (out / "w.bin").write_bytes(b"".join(np.asarray(w, np.int8).tobytes() for w in (net.wc[0], net.wc[1], net.wfc)))
    (out / "lut.bin").write_bytes(np.asarray(LUT, np.uint8).tobytes())
    fr = []
    for f in frames_30fps(video):
        fr.append(f)
        if len(fr) >= frames:
            break
    (out / "frames.u8").write_bytes(np.stack(fr).astype(np.uint8).tobytes())
    board = SimBoard(weights)
    rows = []
    for i, f in enumerate(fr):
        r = board.process(f, i + 1)
        rows.append([r.spk1, r.spk2, r.events] + list(r.cur) + [r.pred, r.windows, r.wcount])
    np.asarray(rows, np.int32).tofile(out / "expected.bin")
    print("exported %d frames, w.bin %d bytes" % (len(fr), (out / "w.bin").stat().st_size))


def compare(out, res):
    exp = np.fromfile(Path(out) / "expected.bin", np.int32).reshape(-1, 12)
    got = np.fromfile(res, np.int32).reshape(-1, 12)
    n = min(len(exp), len(got))
    bad = np.nonzero((exp[:n] != got[:n]).any(axis=1))[0]
    print("%d frames compared, %d differ" % (n, len(bad)))
    if len(bad):
        i = bad[0]
        print("first difference at frame %d\n expected %s\n got      %s" % (i, exp[i].tolist(), got[i].tolist()))
    print("RESULT: PASS" if len(bad) == 0 and n == len(exp) else "RESULT: FAIL")
    return 0 if len(bad) == 0 and n == len(exp) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "compare"])
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="work/arm")
    ap.add_argument("--weights", default="rtl/weights/kth6")
    ap.add_argument("--res")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.video, a.frames, a.out, a.weights)
    else:
        sys.exit(compare(a.out, a.res))
