"""Fast GPU evaluation of exported int8 weights: per-class recall and window / sample accuracy.

    python -m train.per_class train/runs/dvsg_32_16_32 train/runs/r2_c3_64 ...

Uses the QAT forward pass (bit-identical to the integer model, see train/tests/test_qat_vs_int.py), so it is
seconds instead of minutes; `train.eval_int` remains the reference for the final numbers.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from model.snn_config import SNNConfig
from train.datasets import WindowDataset, load_cache
from train.snn_qat import QATSNN

NAMES = ["clap", "Rwave", "Lwave", "Rcw", "Rccw", "Lcw", "Lccw", "roll", "drums", "guitar", "other"]


def load_qat(run):
    """QATSNN with the exported int8 weights of a training run directory (bit-identical to the integer model)."""
    d = Path(run)
    cfg = SNNConfig.from_json((d / "snn_config.json").read_text())
    z = np.load(d / "weights_int8.npz")
    net = QATSNN(cfg).cuda()
    keys = ["conv%d" % (i + 1) for i in range(cfg.n_conv)] + ["fc"]
    with torch.no_grad():
        for p, k in zip(list(net.wc) + [net.w_fc], keys):
            p.copy_(torch.from_numpy(z[k].astype(np.float32)) / cfg.wscale)
    return net, cfg


def evaluate(run, T=48, data=None):
    net, cfg = load_qat(run)
    X, Y = load_cache(Path(data or "data/dvsg%d" % cfg.in_hw) / "test.npz")
    ds = WindowDataset(X, Y, T, False)
    preds, labs, sids, scores = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            items = [ds[j] for j in range(i, min(i + 64, len(ds)))]
            out = net(torch.stack([t[0] for t in items]).cuda()).cpu().numpy()
            scores.append(out)
            preds.append(out.argmax(1))
            labs += [t[1] for t in items]
            sids += [t[2] for t in items]
    scores, preds, labs, sids = np.concatenate(scores), np.concatenate(preds), np.array(labs), np.array(sids)
    per_sample = {}
    for s, o, y in zip(sids, scores, labs):
        per_sample.setdefault(s, [0, y])[0] = per_sample.get(s, [0, y])[0] + o
    samp = np.mean([int(np.argmax(v[0]) == v[1]) for v in per_sample.values()])
    recall = [float((preds[labs == c] == c).mean()) for c in range(11)]
    return float((preds == labs).mean()), float(samp), recall


if __name__ == "__main__":
    print("%-26s %-6s %-6s " % ("run", "win", "smp") + " ".join("%-5s" % n for n in NAMES))
    for run in sys.argv[1:]:
        w, s, r = evaluate(run)
        print("%-26s %.3f  %.3f  " % (Path(run).name, w, s) + " ".join("%.2f " % v for v in r))
