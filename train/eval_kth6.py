"""Score the KTH network on one split: window / sample accuracy, per class, per condition, confusion matrix.

    python -m train.eval_kth6 --weights train/runs/kth --split val
    python -m train.eval_kth6 --weights train/runs/kth --split test --int-check 40      # test people: look once

Windows tile each repetition without overlap (T bins, last one zero-padded, as in training evaluation); a sample is
scored by the summed class currents of its windows. --int-check N re-runs N samples through the bit-accurate numpy
model (model/snn_int.py) and fails if one prediction differs from the GPU evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.snn_int import IntSNN
from train.datasets import WindowDataset, load_cache
from train.per_class import load_qat


@torch.no_grad()
def scores(net, X, Y, T, n_cls):
    ds = WindowDataset(X, Y, T, False)
    sc = np.zeros((len(X), n_cls))
    win = []        # (sample, predicted)
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    for x, y, sid in loader:
        out = net(x.cuda()).cpu().numpy()
        for o, s in zip(out, sid.numpy()):
            sc[s] += o
            win.append((int(s), int(o.argmax())))
    return sc, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/kth32")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--int-check", type=int, default=0)
    a = ap.parse_args()
    net, cfg = load_qat(a.weights)
    net.eval()
    ACTIONS = json.loads((Path(a.data) / "prep.json").read_text())["classes"]
    assert len(ACTIONS) == cfg.classes, "cache has %d classes, the network %d" % (len(ACTIONS), cfg.classes)
    X, Y = load_cache(Path(a.data) / ("%s.npz" % a.split))
    meta = json.loads((Path(a.data) / ("meta_%s.json" % a.split)).read_text())
    sc, win = scores(net, X, Y, a.T, len(ACTIONS))
    ws = np.array([s for s, _ in win])
    wp = np.array([p for _, p in win])
    wy = Y[ws]
    sp = sc.argmax(1)
    print("%s: %d samples, %d windows (T=%d)   window accuracy %.4f   sample accuracy %.4f"
          % (a.split, len(X), len(win), a.T, np.mean(wp == wy), np.mean(sp == Y)))
    print("per class (window recall / sample recall):")
    for c, name in enumerate(ACTIONS):
        m = Y == c
        if m.any():
            print("  %-13s %.3f / %.3f  (n=%d)" % (name, np.mean(wp[wy == c] == c), np.mean(sp[m] == c), m.sum()))
    conds = np.array([m["cond"] for m in meta])
    print("per condition (window / sample): " + "  ".join(
        "%s %.3f / %.3f" % (c, np.mean((wp == wy)[conds[ws] == c]), np.mean((sp == Y)[conds == c])) for c in ("d1", "d2", "d3", "d4")))
    conf = np.zeros((len(ACTIONS),) * 2, int)
    np.add.at(conf, (Y, sp), 1)
    print("sample confusion (rows = true, columns = predicted; order %s):\n%s" % (", ".join(ACTIONS), conf))
    if a.int_check:
        inet = IntSNN.load(a.weights)
        from train.eval_int import windows
        bad = 0
        idx = np.linspace(0, len(X) - 1, min(a.int_check, len(X))).astype(int)
        for i in idx:
            tot = np.zeros(len(ACTIONS), np.int64)
            for w in windows(X[i], a.T):
                tot += inet.run(w)
            bad += int(np.argmax(tot)) != int(sp[i])
        print("integer model vs GPU evaluation on %d samples: %d prediction differences" % (len(idx), bad))
        assert bad == 0


if __name__ == "__main__":
    main()
