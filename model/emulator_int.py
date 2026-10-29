"""Bit-accurate integer model of the event-emulator RTL (rtl/event_emulator.v).

This is the golden reference for the cocotb/Icarus testbenches. Per pixel, in raster order:

    cur = LUT[pixel]                      (8 bit)
    first frame after reset: ref = cur, no events
    d   = cur - ref                       (9 bit signed)
    d >=  theta  -> ON  event;   d <= -theta -> OFF event      (at most ONE event per pixel per frame)
    update policy on an event:
        "step"  : ref += / -= theta       (residual contrast is kept, big changes spread over frames)
        "reset" : ref  = cur              (residual is discarded)
    no event -> ref unchanged

All events of frame k carry timestamp (k * period_ticks) mod 2**17. Events leave the model in
raster order (y major, x minor), which is the order the RTL emits them.
"""
from dataclasses import dataclass

import numpy as np

from model.event_format import frame_ts, pack
from model.log_lut import LUT


@dataclass(frozen=True)
class EmuConfig:
    theta: int = 9            # LUT LSBs, >= 1 (9 ~ 0.2 ln units)
    mode: str = "step"        # "step" or "reset"
    period_ticks: int = 333   # frame period in 100 us ticks (333 ~ 30 fps)
    width: int = 128
    height: int = 128

    def __post_init__(self):
        assert self.mode in ("step", "reset")
        assert 1 <= self.theta <= 255
        assert 1 <= self.width <= 128 and 1 <= self.height <= 128


class IntEmulator:
    def __init__(self, cfg=EmuConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.ref = None
        self.frame_idx = 0

    def step(self, frame):
        """Feed one uint8 frame (height x width). Returns the uint32 event words of this frame."""
        c = self.cfg
        frame = np.asarray(frame, dtype=np.uint8)
        assert frame.shape == (c.height, c.width), frame.shape
        cur = LUT[frame].astype(np.int16)
        ts = frame_ts(self.frame_idx, c.period_ticks)
        self.frame_idx += 1
        if self.ref is None:
            self.ref = cur.copy()
            return np.zeros(0, dtype=np.uint32)

        d = cur - self.ref
        on = d >= c.theta
        off = d <= -c.theta
        if c.mode == "step":
            self.ref = self.ref + c.theta * on.astype(np.int16) - c.theta * off.astype(np.int16)
        else:
            self.ref = np.where(on | off, cur, self.ref)

        ev = on | off
        ys, xs = np.nonzero(ev)  # np.nonzero is C-order: y major, x minor
        pol = on[ys, xs].astype(np.uint32)
        return pack(pol, ys, xs, np.full(ys.shape, ts, dtype=np.uint32))


def emulate(frames, cfg=EmuConfig()):
    """frames: iterable of uint8 (H, W). Returns a list of per-frame uint32 event arrays."""
    emu = IntEmulator(cfg)
    return [emu.step(f) for f in frames]
