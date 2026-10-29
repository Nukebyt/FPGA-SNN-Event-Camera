"""Hardware verification: run the same frames through the real board and through the simulated board (the Python golden models)
and require every per-frame answer to be identical.

    python scripts/check_board.py --host 192.168.137.200 --source data/kth/videos/person22_boxing_d1_uncomp.avi --frames 200

Compared for every frame: spikes of both layers, number of emulator events, the six class currents, the running window score, the
window counters, the class and scores of every finished window. Not compared: the cycle counters (they are measurements) and
the status word. Prints RESULT: PASS only if all frames match (checklist 4.2: look at the offset only after a value mismatch;
the first differing frame is printed with both answers).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.snn_viewer import frames_30fps  # noqa: E402
from sw.host import snnproto as P  # noqa: E402
from sw.host.snnboard_sim import SimBoard  # noqa: E402

FIELDS = ["pred", "windows", "wcount", "spk1", "spk2", "events", "cur", "score", "win_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--weights", default="rtl/weights/kth6")
    a = ap.parse_args()

    frames = []
    for f in frames_30fps(a.source):
        frames.append(f)
        if len(frames) >= a.frames:
            break
    print("%d frames from %s" % (len(frames), a.source))

    sim = SimBoard(a.weights)
    sport = sim.start()
    hw, ref = P.Client(a.host, a.port), P.Client("127.0.0.1", sport)
    bad = 0
    lat = []
    try:
        print("board build id 0x%08X" % hw.hello())
        ref.hello()
        hw.reset()
        ref.reset()
        for i, f in enumerate(frames):
            hw.send_frame(f)
            rh = hw.recv_result()
            ref.send_frame(f)
            rr = ref.recv_result()
            lat.append(rh.latency_cycles)
            diffs = [k for k in FIELDS if getattr(rh, k) != getattr(rr, k)]
            if diffs:
                bad += 1
                if bad <= 3:
                    print("MISMATCH frame %d fields %s" % (i, diffs))
                    for k in diffs:
                        print("   %-10s board %s\n   %-10s model %s" % (k, getattr(rh, k), "", getattr(rr, k)))
        print("info: %d frames, %d differ; DMA start to result: min %d max %d clocks (%.2f ms), step %d clocks" % (
            len(frames), bad, min(lat), max(lat), max(lat) / 25000.0, rh.step_cycles))
        print("info: finished windows on the board: %d, last window class %s" % (rh.windows, P.CLASSES[rh.pred] if rh.windows else "-"))
    finally:
        hw.close()
        ref.close()
        sim.stop()
    print("RESULT: PASS" if bad == 0 else "RESULT: FAIL")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
