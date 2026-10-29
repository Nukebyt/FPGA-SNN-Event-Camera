"""KTH videos -> emulator events -> 2x32x32 presence bins, cached per split (the training data of the KTH network).

    python -m train.prep_kth                       # data/kth32/{train,val,test}.npz (+ meta_<split>.json), theta 0.2 ln, step
    python -m train.prep_kth --actions handclapping,handwaving,boxing    # only the actions downloaded so far

Every repetition listed in 00sequences.txt is one sample (about 2400 over 599 videos). Frames are the whole 160x120
picture squashed to 128x128 (so walkers keep their full path), 25 -> 30 fps by blending, then the integer emulator
(the same code the RTL is checked against) and the same binning as the DVS128 pipeline. Labels are the
position of the action in --actions, kept in the paper's order (train.kth.ACTIONS); the list is saved in prep.json. Split by person as in the original paper (SPLITS in train/kth.py).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from model.emulator_int import EmuConfig, emulate
from model.log_lut import theta_from_ln
from model.snn_input import bin_emulated
from train.datasets import save_cache
from train.kth import ACTIONS, load_kth


def _one(args):
    root, action, split, theta_ln, mode, down, fit = args
    xs, metas = [], []
    for frames, meta in load_kth(root, action, split, fit=fit):
        xs.append(bin_emulated(emulate(frames, EmuConfig(theta=theta_from_ln(theta_ln), mode=mode)), down=down))
        metas.append(meta)
    return xs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kth", default="data/kth")
    ap.add_argument("--out", default="data/kth32")
    ap.add_argument("--theta-ln", type=float, default=0.2)
    ap.add_argument("--mode", default="step", choices=["step", "reset"])
    ap.add_argument("--down", type=int, default=4, help="128 / down = SNN input size")
    ap.add_argument("--fit", default="squash", choices=["squash", "crop"])
    ap.add_argument("--actions", default=",".join(ACTIONS))
    a = ap.parse_args()
    actions = [x for x in ACTIONS if x in a.actions.split(",")]      # class label = position in this list (paper order)
    with Pool() as pool:
        for split in ("train", "val", "test"):
            jobs = [(a.kth, act, split, a.theta_ln, a.mode, a.down, a.fit) for act in actions]
            X, Y, M = [], [], []
            for act, (xs, metas) in zip(actions, pool.map(_one, jobs)):
                X += xs
                Y += [actions.index(act)] * len(xs)
                M += metas
            save_cache(Path(a.out) / ("%s.npz" % split), X, Y)
            (Path(a.out) / ("meta_%s.json" % split)).write_text(json.dumps(M))
            print("%-5s %4d samples, bins per sample median %d, active input %.3f, per class %s"
                  % (split, len(X), int(np.median([len(x) for x in X])), float(np.mean([x.mean() for x in X])),
                     np.bincount(Y, minlength=len(actions)).tolist()), flush=True)
    (Path(a.out) / "prep.json").write_text(json.dumps(dict(vars(a), classes=actions)))


if __name__ == "__main__":
    main()
