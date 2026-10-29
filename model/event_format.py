"""32-bit event word shared by the emulator RTL, the golden models and the SNN front end.

    bit 31      polarity (1 = ON / brighter, 0 = OFF / darker)
    bits 30:24  y  (0..127)
    bits 23:17  x  (0..127)
    bits 16:0   timestamp, 17 bits, units of TS_TICK_US (wraps every 2**17 ticks)

Frame-based emulation stamps every event of a frame with that frame's timestamp
(docs/architecture_decisions.md, decision 3).
"""
import numpy as np

TS_BITS = 17
TS_MASK = (1 << TS_BITS) - 1
TS_TICK_US = 100  # one timestamp unit = 100 us; 17 bits wrap every 13.1 s
XY_BITS = 7
MAX_DIM = 1 << XY_BITS


def pack(pol, y, x, ts):
    pol = np.asarray(pol, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    x = np.asarray(x, dtype=np.uint32)
    ts = np.asarray(ts, dtype=np.uint32)
    assert np.all(y < MAX_DIM) and np.all(x < MAX_DIM) and np.all(pol <= 1)
    return (pol << 31) | (y << 24) | (x << 17) | (ts & TS_MASK)


def unpack(words):
    w = np.asarray(words, dtype=np.uint32)
    return {
        "pol": (w >> 31) & 1,
        "y": (w >> 24) & 0x7F,
        "x": (w >> 17) & 0x7F,
        "ts": w & TS_MASK,
    }


def frame_ts(frame_idx, period_ticks):
    """Timestamp stamped on all events of frame `frame_idx` (0-based since reset)."""
    return (frame_idx * period_ticks) & TS_MASK
