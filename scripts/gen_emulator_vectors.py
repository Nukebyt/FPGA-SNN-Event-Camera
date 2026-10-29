"""Writes the stimulus + expected-event vectors for tb/event_emulator_tb.v.

Per scenario directory (default work/vec/<name>/):
    frames.hex    one pixel byte per line, frames concatenated, raster order
    expected.hex  one 32-bit event word per line (8 hex digits), from model/emulator_int.py
    params.vh     `define macros the testbench includes

Scenarios deliberately vary the timing dimension simple benches skip (checklist rule 4):
source gaps (GAP_MODE) and a stalled consumer (READY_MODE), plus edge data (full-frame bursts,
saturated pixels, theta extremes, timestamp wrap).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import synth_video as sv  # noqa: E402
from model.emulator_int import EmuConfig, emulate  # noqa: E402


def _edge_frames(size=16):
    z = np.zeros((size, size), np.uint8)
    f = [z, z + 255, z + 255, z, z + 255]
    f.append(np.tile(np.arange(size, dtype=np.uint8) * 16, (size, 1)))   # gradient
    return np.stack(f)


# name -> dict(frames, cfg, gap, ready, aw, expect_stall)
# gap  : 0 none, 1 random 25% bubbles on the pixel stream
# ready: 0 always ready, 1 random 50%, 2 starved (ready 1 clock in 16)
SCENARIOS = {
    "bar128":        dict(frames=lambda: sv.moving_bar(12), cfg=EmuConfig(theta=9, mode="step"),
                          gap=0, ready=0, aw=10, stall=False),
    "blob128_gaps":  dict(frames=lambda: sv.moving_blob(8), cfg=EmuConfig(theta=6, mode="reset"),
                          gap=1, ready=1, aw=10, stall=False),
    "flash32_burst": dict(frames=lambda: sv.flash(n_frames=10, size=32, at=4),
                          cfg=EmuConfig(theta=9, mode="step", width=32, height=32),
                          gap=0, ready=2, aw=6, stall=True),
    "noise32_stall": dict(frames=lambda: sv.noise(6, size=32), cfg=EmuConfig(theta=5, mode="reset", width=32, height=32),
                          gap=1, ready=2, aw=5, stall=True),
    "edge16_t255":   dict(frames=_edge_frames, cfg=EmuConfig(theta=255, mode="step", width=16, height=16),
                          gap=1, ready=1, aw=6, stall=False),
    "edge16_t1":     dict(frames=_edge_frames, cfg=EmuConfig(theta=1, mode="step", width=16, height=16),
                          gap=0, ready=1, aw=8, stall=False),
    "tswrap16":      dict(frames=lambda: sv.moving_bar(10, size=16, bar_w=3, step=1),
                          cfg=EmuConfig(theta=9, mode="reset", period_ticks=40000, width=16, height=16),
                          gap=0, ready=0, aw=8, stall=False),
}


def write_scenario(name, out_dir):
    s = SCENARIOS[name]
    frames = np.asarray(s["frames"](), dtype=np.uint8)
    cfg = s["cfg"]
    assert frames.shape[1:] == (cfg.height, cfg.width), (frames.shape, cfg)
    ev = emulate(frames, cfg)
    words = np.concatenate(ev) if ev else np.zeros(0, np.uint32)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames.hex").write_text("".join("%02x\n" % v for v in frames.reshape(-1)))
    (out / "expected.hex").write_text("".join("%08x\n" % int(v) for v in words) or "00000000\n")
    posix = lambda p: (out / p).as_posix()
    (out / "params.vh").write_text(
        "`define W %d\n`define H %d\n`define NFRAMES %d\n`define THETA %d\n`define MODE_RESET %d\n"
        "`define PERIOD %d\n`define NEXP %d\n`define GAP_MODE %d\n`define READY_MODE %d\n"
        "`define FIFO_AW %d\n`define EXPECT_STALL %d\n`define FRAMES_FILE \"%s\"\n`define EXP_FILE \"%s\"\n"
        % (cfg.width, cfg.height, len(frames), cfg.theta, int(cfg.mode == "reset"), cfg.period_ticks,
           len(words), s["gap"], s["ready"], s["aw"], int(s["stall"]), posix("frames.hex"), posix("expected.hex")))
    return len(frames), len(words)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--out", default=str(ROOT / "work" / "vec"))
    a = ap.parse_args()
    for n in (sorted(SCENARIOS) if a.scenario == "all" else [a.scenario]):
        nf, ne = write_scenario(n, Path(a.out) / n)
        print("%-16s frames=%d events=%d" % (n, nf, ne))
