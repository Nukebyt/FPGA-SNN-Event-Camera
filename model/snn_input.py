"""Events -> binary SNN input maps [steps, 2, 32, 32]. Channel 0 = OFF, channel 1 = ON.

The RTL time-bin accumulator does exactly this: set bit (pol, y >> 2, x >> 2) per event, hand the
bitmap to layer 1 at the bin boundary, clear. A bin boundary is an explicit marker (the emulator's
end of frame, or the HPS replay engine for recorded events), not a timestamp compare.
"""
import numpy as np

from model.event_format import unpack


def bin_real(x, y, p, t_us, bin_us=33333, size=128, down=4, n_bins=None, t0=None):
    """Recorded events (arrays) -> uint8 [n_bins, 2, size/down, size/down].

    Bin k covers [t0 + k*bin_us, t0 + (k+1)*bin_us). t0 defaults to the first event time; pass the
    start of a labelled window to align bins with it. With n_bins, later events are dropped.
    """
    t_us = np.asarray(t_us, dtype=np.int64)
    if t0 is None:
        t0 = t_us.min() if len(t_us) else 0
    b = (t_us - t0) // bin_us
    nb = int(b.max()) + 1 if len(b) else 1
    if n_bins is not None:
        nb = n_bins
        keep = b < nb
    else:
        keep = np.ones(len(b), bool)
    s = size // down
    out = np.zeros((nb, 2, s, s), np.uint8)
    out[b[keep], np.asarray(p)[keep].astype(np.intp), np.asarray(y)[keep] // down, np.asarray(x)[keep] // down] = 1
    return out


def bin_emulated(per_frame_words, size=128, down=4):
    """Emulator output (list of uint32 arrays, one per frame) -> uint8 [frames, 2, s, s]."""
    s = size // down
    out = np.zeros((len(per_frame_words), 2, s, s), np.uint8)
    for i, w in enumerate(per_frame_words):
        if len(w):
            u = unpack(w)
            out[i, u["pol"].astype(np.intp), (u["y"] // down).astype(np.intp), (u["x"] // down).astype(np.intp)] = 1
    return out
