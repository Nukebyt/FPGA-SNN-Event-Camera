"""Reader for the IBM DVS128 Gesture release (AEDAT 3.1, polarity events only).

File = ASCII header lines starting with '#' ending with '#!END-HEADER\\r\\n', then packets:
    28-byte packet header: u16 eventType, u16 eventSource, u32 eventSize, u32 eventTSOffset,
                           u32 eventTSOverflow, u32 eventCapacity, u32 eventNumber, u32 eventValid
    eventNumber events of eventSize bytes: u32 data, u32 timestamp
    data: bit 0 valid, bit 1 polarity, bits 2-14 y, bits 17-29 x  (README of the dataset)
    time (us) = (eventTSOverflow << 31) | timestamp
Only type 1 (polarity) packets are decoded; anything else is skipped by its declared size.
"""
import csv
import struct
from pathlib import Path

import numpy as np

HEADER_END = b"#!END-HEADER\r\n"
PKT = struct.Struct("<HHIIIIII")


def read_aedat(path):
    """Returns dict of arrays x, y, p (uint8), t (int64, microseconds)."""
    buf = Path(path).read_bytes()
    pos = buf.index(HEADER_END) + len(HEADER_END)
    xs, ys, ps, ts = [], [], [], []
    n = len(buf)
    while pos + PKT.size <= n:
        etype, _src, esize, _off, overflow, _cap, number, _valid = PKT.unpack_from(buf, pos)
        pos += PKT.size
        nbytes = esize * number
        if etype == 1 and esize == 8:
            a = np.frombuffer(buf, dtype="<u4", count=2 * number, offset=pos).reshape(number, 2)
            data, ts32 = a[:, 0], a[:, 1]
            ok = (data & 1) == 1
            data, ts32 = data[ok], ts32[ok]
            xs.append(((data >> 17) & 0x1FFF).astype(np.int16))
            ys.append(((data >> 2) & 0x1FFF).astype(np.int16))
            ps.append(((data >> 1) & 1).astype(np.uint8))
            ts.append((np.int64(overflow) << 31) | ts32.astype(np.int64))
        pos += nbytes
    if not xs:
        z = np.zeros(0, np.int64)
        return {"x": z.astype(np.int16), "y": z.astype(np.int16), "p": z.astype(np.uint8), "t": z}
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "p": np.concatenate(ps), "t": np.concatenate(ts)}


def read_labels(path):
    """[(class 1..11, start_us, end_us), ...] in file order."""
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 3 and row[0].strip():
                rows.append((int(row[0]), int(row[1]), int(row[2])))
    return rows
