"""Bit-accurate integer inference model of the SNN (the golden model for the Phase 5 RTL).

Weights are int8 arrays: conv1 [C1, 2, 3, 3], conv2 [C2, C1, 3, 3], (conv3 [C3, C2, 3, 3] if c3 > 0), then
fc [classes, C_last*(H/2^n)^2] (fc input flattened in (channel, y, x) order). Everything is int32 arithmetic; the only
"multiplication" is thr * spike, i.e. a conditional subtract.
"""
from pathlib import Path

import numpy as np

from model.snn_config import SNNConfig


def conv3x3(x, w):
    """x [Cin, H, W] int32, w [Cout, Cin, 3, 3] int32, zero padding 1 -> [Cout, H, W] int32."""
    cin, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], h, wd), np.int32)
    for ky in range(3):
        for kx in range(3):
            out += np.einsum("oi,ihw->ohw", w[:, :, ky, kx], xp[:, ky:ky + h, kx:kx + wd], dtype=np.int32)
    return out


def pool_or(s):
    c, h, w = s.shape
    return s.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def lif(v, current, cfg, thr):
    v = v - (v >> cfg.leak_shift)
    v = np.clip(v + current, cfg.vmin, cfg.vmax)
    spk = (v >= thr).astype(np.int32)
    v = v - thr * spk
    return v, spk


class IntSNN:
    def __init__(self, cfg, *weights):
        """weights: one array per conv layer, then the fc array (so IntSNN(cfg, w1, w2, w3) still works)."""
        self.cfg = cfg
        ws = [np.asarray(w, dtype=np.int32) for w in weights]
        assert len(ws) == cfg.n_conv + 1, "expected %d weight arrays, got %d" % (cfg.n_conv + 1, len(ws))
        for w in ws:
            assert w.min() >= -128 and w.max() <= 127, "weights must fit int8"
        self.wc, self.wfc = ws[:-1], ws[-1]
        self.reset()

    @staticmethod
    def load(directory):
        d = Path(directory)
        cfg = SNNConfig.from_json((d / "snn_config.json").read_text())
        z = np.load(d / "weights_int8.npz")
        return IntSNN(cfg, *[z["conv%d" % (i + 1)] for i in range(cfg.n_conv)], z["fc"])

    def reset(self):
        c = self.cfg
        self.v = [np.zeros((ch, c.in_hw >> i, c.in_hw >> i), np.int32) for i, ch in enumerate(c.channels)]
        self.v_out = np.zeros((c.classes,), np.int32)

    def step(self, x):
        """One time step. x [2, H, H] 0/1. Returns (s1, ..., s_n, out): conv spikes (before pooling) and the
        output: class currents of this step (readout "sum") or output spikes (readout "spike")."""
        c = self.cfg
        cur = np.asarray(x, dtype=np.int32)
        spikes = []
        for i, w in enumerate(self.wc):
            self.v[i], s = lif(self.v[i], conv3x3(cur, w), c, c.thr[i])
            spikes.append(s)
            cur = pool_or(s)
        i_out = self.wfc @ cur.reshape(-1)
        if c.readout == "sum":
            return tuple(spikes) + (i_out.astype(np.int64),)
        self.v_out, s_out = lif(self.v_out, i_out, c, c.thr[-1])
        return tuple(spikes) + (s_out,)

    def run(self, bins, reset=True, record=False):
        """bins [T, 2, H, H]. Returns class scores [classes] (and the per-step step() tuples if record).

        scores = summed class currents (readout "sum") or output spike counts (readout "spike")."""
        if reset:
            self.reset()
        counts = np.zeros(self.cfg.classes, np.int64)
        rec = []
        for t in range(len(bins)):
            out = self.step(bins[t])
            counts += out[-1]
            if record:
                rec.append(out)
        return (counts, rec) if record else counts

    def predict(self, bins):
        counts = self.run(bins)
        return int(np.argmax(counts)), counts   # ties -> lowest class index (the RTL must use the same rule)
