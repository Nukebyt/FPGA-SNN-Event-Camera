"""Pixel-exact check of the on-chip demo's VGA frames against the Python models.

Reads the FRAME lines printed by tb/demo_top_tb.v and the dumped frameNN.hex files, rebuilds the expected
picture from model/pattern_source.py -> model/emulator_int.py -> event-view rules, and compares every pixel.

    python scripts/check_demo_vga.py --dir work/demo --log work/demo/sim.log --theta 9 --mode step
Prints "RESULT: PASS" / "RESULT: FAIL".
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.emulator_int import EmuConfig, IntEmulator  # noqa: E402
from model.event_format import unpack  # noqa: E402
from model.pattern_source import pattern_frame  # noqa: E402

GREEN, RED, GREY, BLACK = (0, 255, 0), (255, 0, 0), (28, 28, 28), (0, 0, 0)


def expected_image(first, last, events_per_frame):
    state = np.zeros((128, 128), np.uint8)       # 0 none, 1 ON, 2 OFF
    for k in range(first, last + 1):
        u = unpack(events_per_frame[k])
        state[u["y"], u["x"]] = np.where(u["pol"] == 1, 1, 2)
    panel = np.empty((128, 128, 3), np.uint8)
    panel[:] = GREY
    panel[state == 1] = GREEN
    panel[state == 2] = RED
    img = np.zeros((480, 640, 3), np.uint8)
    img[48:432, 128:512] = np.repeat(np.repeat(panel, 3, axis=0), 3, axis=1)
    return img


def load_hex(path):
    vals = []
    for line in Path(path).read_text().split():
        if line.startswith("//") or line.startswith("0x") or line.startswith("@"):
            continue                              # address comments written by $writememh
        try:
            vals.append(int(line, 16))
        except ValueError:
            vals.append(-1)                      # X/Z in the simulation output
    v = np.array(vals, dtype=np.int64)
    assert len(v) == 307200, len(v)
    img = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=1).astype(np.uint8).reshape(480, 640, 3)
    return img, int((v < 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/demo")
    ap.add_argument("--log", default="work/demo/sim.log")
    ap.add_argument("--theta", type=int, default=9)
    ap.add_argument("--mode", default="step")
    a = ap.parse_args()

    text = Path(a.log).read_text()
    frames = [tuple(int(v) for v in m) for m in re.findall(r"FRAME (\d+) first=(\d+) last=(\d+)", text)]
    if not frames:
        print("FAIL: no FRAME lines in log")
        print("RESULT: FAIL")
        return 1
    max_k = max(l for _, f, l in frames if f <= l) if any(f <= l for _, f, l in frames) else 0
    emu = IntEmulator(EmuConfig(theta=a.theta, mode=a.mode))
    events = [emu.step(pattern_frame(k)) for k in range(max_k + 1)]

    ok = True
    nonempty = 0
    for n, first, last in frames:
        img, nx = load_hex(Path(a.dir) / ("frame%02d.hex" % n))
        exp = expected_image(first, last, events) if first <= last else expected_image(1, 0, events)
        diff = int((img != exp).any(axis=2).sum())
        n_on = int((exp == np.array(GREEN)).all(axis=2).sum()) // 9
        n_off = int((exp == np.array(RED)).all(axis=2).sum()) // 9
        nonempty += (n_on + n_off) > 0
        print("frame %2d  window %d..%d  expected ON %4d OFF %4d  X-pixels %d  mismatching pixels %d"
              % (n, first, last, n_on, n_off, nx, diff))
        if diff or nx:
            ok = False
            try:
                import cv2
                cv2.imwrite(str(Path(a.dir) / ("bad%02d.png" % n)), img[:, :, ::-1])
                cv2.imwrite(str(Path(a.dir) / ("exp%02d.png" % n)), exp[:, :, ::-1])
            except Exception:
                pass
    if nonempty == 0:
        print("FAIL: no frame contained any event, the check would be vacuous")
        ok = False
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
