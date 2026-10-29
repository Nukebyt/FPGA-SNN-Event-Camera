"""Guided photo session for the board's HEX / LED display: plays one KTH clip per action on the real board and tells you what to see.

    python scripts/photo_session.py --host 192.168.137.200                 # asks Enter before each clip
    python scripts/photo_session.py --host 192.168.137.200 --no-pause --seconds 6

What the board shows while a clip streams (rtl/snn_engine.v status conduit, DE10_Standard_SNN.v):
  HEX0        class of the last finished 32-frame window (0 walking, 1 jogging, 2 running, 3 boxing, 4 hand waving, 5 hand clapping)
  HEX3..HEX2  steps done in the current window, hex 00..1F (counts up about once a second)
  HEX5..HEX4  frames processed since the clip started, low byte, hex
  LEDR[1]     engine ready     LEDR[2]  enabled (on while streaming, off after the stream stops)      LEDR[3]  error (must stay OFF)      LEDR[0]  HPS heartbeat (blinks)
The clip is looped and the engine is reset before each action, so the first window ends 32 frames (1.07 s) after the start;
photograph after HEX0 shows the digit this script prints (it prints what the board itself reported, read over the network).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.snn_viewer import frames_30fps  # noqa: E402
from sw.host import snnproto as P  # noqa: E402

CLIPS = [("walking", "person22_walking_d1"), ("jogging", "person22_jogging_d1"), ("running", "person22_running_d1"),
         ("boxing", "person22_boxing_d1"), ("hand waving", "person22_handwaving_d1"), ("hand clapping", "person22_handclapping_d1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--seconds", type=float, default=12.0, help="how long each action plays")
    ap.add_argument("--no-pause", action="store_true")
    a = ap.parse_args()
    print(__doc__.split("What the board shows")[0].strip().splitlines()[0])
    for k, (label, stem) in enumerate(CLIPS):
        path = os.path.join(ROOT, "data", "kth", "videos", stem + "_uncomp.avi")
        frames = list(f for _, f in zip(range(600), frames_30fps(path)))
        if not a.no_pause:
            input("\n[%d/6] next: %s  (true action).  Aim the camera at HEX5..HEX0 and LEDR, then press Enter ... " % (k + 1, label))
        c = P.Client(a.host, timeout=15)
        c.hello()
        c.reset()
        print("\n=== %s: streaming for %.0f s; the board's answers (what HEX0 will show after each window):" % (label, a.seconds))
        t_end = time.time() + a.seconds
        seen, i = 0, 0
        while time.time() < t_end:
            t0 = time.time()
            c.send_frame(frames[i % len(frames)])
            r = c.recv_result()
            i += 1
            if r.windows > seen:
                seen = r.windows
                print("   window %2d finished at frame %3d: HEX0 shows %d (%s)   [frames processed %d = 0x%02X]" % (seen, i, r.pred, P.CLASSES[r.pred], r.done_seq, r.done_seq & 0xFF))
            time.sleep(max(0.0, 1 / 30.0 - (time.time() - t0)))
        c.close()
        print("   (stream stopped; the display keeps the last values until the next connection)")
        if not a.no_pause:
            print("   >>> photograph NOW (the display is frozen at the last result) <<<")


if __name__ == "__main__":
    main()
