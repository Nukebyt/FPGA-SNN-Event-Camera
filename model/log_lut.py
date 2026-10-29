"""Log-intensity LUT: 8-bit pixel -> 8-bit fixed-point log value.

    L[p] = floor(255 * log2(1 + p) / 8 + 0.5),  p = 0..255,  so L[0] = 0 and L[255] = 255.

One LSB of L is ln(2) * 8 / 255 = 0.02174 natural-log units, so a DVS-style contrast
threshold of 0.2 ln units is theta ~ 9 LSB. The +1 keeps log finite at p = 0 and gives a
dark-pixel response that is not infinitely sensitive.

`python -m model.log_lut` writes rtl/log_lut.hex (the ROM image the RTL loads).
"""
from pathlib import Path

import numpy as np

LOG_BITS = 8
LN_PER_LSB = np.log(2.0) * 8.0 / 255.0


def build_lut():
    p = np.arange(256, dtype=np.float64)
    return np.floor(255.0 * np.log2(1.0 + p) / 8.0 + 0.5).astype(np.uint8)


LUT = build_lut()


def theta_from_ln(theta_ln):
    """Integer threshold (LUT LSBs) closest to a contrast threshold in natural-log units."""
    return max(1, int(round(theta_ln / LN_PER_LSB)))


def write_hex(path):
    Path(path).write_text("".join("%02x\n" % v for v in LUT))


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "rtl" / "log_lut.hex"
    write_hex(out)
    print("wrote", out, " LSB = %.5f ln units" % LN_PER_LSB)
