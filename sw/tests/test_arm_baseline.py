"""The C baseline (sw/arm_baseline.c, built in WSL) must equal the bit-accurate model on every frame, in both implementations."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from model.log_lut import LUT
from model.pattern_source import pattern_frame
from model.snn_int import IntSNN
from sw.host.snnboard_sim import SimBoard

ROOT = Path(__file__).resolve().parents[2]


def _wsl_ok():
    if not shutil.which("wsl"):
        return False
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", "which gcc"], capture_output=True, text=True, timeout=60)
    return r.returncode == 0 and "gcc" in r.stdout


@pytest.mark.skipif(not _wsl_ok(), reason="needs WSL Ubuntu with gcc")
@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_arm_baseline_equals_model(mode):
    work = ROOT / "work" / "test_arm"
    work.mkdir(parents=True, exist_ok=True)
    net = IntSNN.load(str(ROOT / "rtl" / "weights" / "kth6"))
    (work / "w.bin").write_bytes(b"".join(np.asarray(w, np.int8).tobytes() for w in (net.wc[0], net.wc[1], net.wfc)))
    (work / "lut.bin").write_bytes(np.asarray(LUT, np.uint8).tobytes())
    n = 70                                       # more than two windows of 32 frames: the window reset is exercised
    rng = np.random.default_rng(3)
    frames = [np.clip(pattern_frame(f).astype(int) + rng.integers(-6, 7, (128, 128)), 0, 255).astype(np.uint8) for f in range(n)]
    (work / "frames.u8").write_bytes(np.stack(frames).tobytes())
    board = SimBoard(str(ROOT / "rtl" / "weights" / "kth6"))
    rows = []
    for i, f in enumerate(frames):
        r = board.process(f, i + 1)
        rows.append([r.spk1, r.spk2, r.events] + list(r.cur) + [r.pred, r.windows, r.wcount])
    expected = np.asarray(rows, np.int32)
    posix = "/mnt/" + ROOT.drive[0].lower() + str(ROOT.as_posix())[2:]
    cmd = ("cd %s && gcc -std=gnu99 -O2 -Wall -Wextra -o /tmp/arm_baseline_t sw/arm_baseline.c && "
           "/tmp/arm_baseline_t --weights work/test_arm/w.bin --lut work/test_arm/lut.bin --frames work/test_arm/frames.u8 --n %d --mode %s --out work/test_arm/res_%s.bin" % (posix, n, mode, mode))
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "warning" not in r.stderr, r.stdout + r.stderr
    got = np.fromfile(work / ("res_%s.bin" % mode), np.int32).reshape(-1, 12)
    assert got.shape == expected.shape
    assert (got == expected).all(), "first differing frame %d" % int(np.nonzero((got != expected).any(axis=1))[0][0])
    assert expected[:, 2].sum() > 1000 and expected[:, 0].sum() > 100, "test must exercise events and spikes"
