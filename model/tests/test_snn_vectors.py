"""The ROM image layouts written by scripts/gen_snn_vectors.py are read back by independent code and must give the original weights
(the RTL benches check the same layouts end to end; this catches a layout slip in seconds)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_snn_vectors import pack_bins, pack_conv, pack_fc, pack_membranes, random_net, run_model, random_bins  # noqa: E402


def signed8(v):
    return v - 256 if v >= 128 else v


def test_conv_layout_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.integers(-128, 128, (32, 16, 3, 3))
    lines = pack_conv(w)
    assert len(lines) == 4 * 16 * 9
    for g in range(4):
        for ci in range(16):
            for ky in range(3):
                for kx in range(3):
                    word = int(lines[g * 16 * 9 + ci * 9 + ky * 3 + kx], 16)
                    for j in range(8):
                        assert signed8((word >> (8 * j)) & 0xFF) == w[g * 8 + j, ci, ky, kx]


def test_fc_layout_roundtrip():
    rng = np.random.default_rng(1)
    w = rng.integers(-128, 128, (6, 32 * 64))
    lines = pack_fc(w, 32)
    assert len(lines) == 6 * 2048
    for k in range(6):
        for a in range(256):          # a = g*64 + y*8 + x (the pooled word address), j = channel inside the word
            g, rem = divmod(a, 64)
            for j in range(8):
                assert signed8(int(lines[k * 2048 + a * 8 + j], 16)) == w[k, (g * 8 + j) * 64 + rem]


def test_bins_layout():
    b = np.zeros((1, 2, 32, 32), np.uint8)
    b[0, 1, 5, 9] = 1                  # ON channel, y 5, x 9 -> word 32 + 5, bit 9
    lines = pack_bins(b)
    assert len(lines) == 64
    assert [i for i, l in enumerate(lines) if int(l, 16)] == [37] and int(lines[37], 16) == 1 << 9


def test_membrane_pack_signed12():
    v = np.zeros((8, 4, 4), np.int32)
    v[3, 1, 2] = -2048
    v[7, 1, 2] = 2047
    word = int(pack_membranes(v)[1 * 4 + 2], 16)
    assert (word >> 36) & 0xFFF == 0x800 and (word >> 84) & 0xFFF == 0x7FF


def test_random_net_fires_and_model_is_deterministic():
    net = random_net(1)
    exp1, s1 = run_model(net, random_bins(1, 4))
    exp2, s2 = run_model(net, random_bins(1, 4))
    assert exp1 == exp2 and (s1 == s2).all()
    assert exp1[2][0] > 100 and exp1[2][1] > 100            # both layers spike, the test is not vacuous
