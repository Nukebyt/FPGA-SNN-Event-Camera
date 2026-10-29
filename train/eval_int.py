"""Evaluate the exported int8 SNN with the bit-accurate numpy model (roadmap 3.9, 3.10).

    python -m train.eval_int --weights train/runs/toy --dataset toy
    python -m train.eval_int --weights train/runs/dvsg --dataset dvsg --data data/dvsg32
    python -m train.eval_int --weights train/runs/dvsg --dataset own --clips data/own_clips   # domain gap (3.10)

Per non-overlapping window of T bins: predicted class = argmax of the class scores (summed output current,
or output spike count for the spike readout; ties -> lowest index). Also prints spikes per inference and the per-layer firing fraction, which sets hardware cost.
"""
import argparse
from pathlib import Path

import numpy as np

from model.snn_int import IntSNN


def windows(sample, T):
    for s in range(0, max(len(sample), 1), T):
        w = sample[s:s + T]
        if len(w) < T:
            w = np.concatenate([w, np.zeros((T - len(w),) + w.shape[1:], w.dtype)])
        yield w


def trim_inactive(bins, min_len=48):
    """Drop leading and trailing bins without motion (the 1 s of standing still before and after a gesture)."""
    act = bins.sum(axis=(1, 2, 3)).astype(float)
    idx = np.nonzero(act > max(5.0, 0.25 * np.percentile(act, 90)))[0]
    if len(idx) < min_len:
        return bins
    return bins[idx[0]:idx[-1] + 1]


def load_own_clips(root, size=128, down=4, trim=True):
    """data/own_clips/<NN>_<name>/*.mov -> emulator events -> bins. Track B / domain-gap data.

    The folder name must start with the DVS128 Gesture class number (0-10), e.g. 05_left_arm_cw, so labels are
    right even when only some classes are recorded. Returns (bins list, labels array, folder names)."""
    import re
    from model.emulator_int import EmuConfig, emulate
    from model.log_lut import theta_from_ln
    from model.snn_input import bin_emulated
    from model.video_io import read_video
    X, Y, names = [], [], []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        m = re.match(r"(\d+)_", d.name)
        if not m or not 0 <= int(m.group(1)) <= 10:
            raise ValueError("class folder %r must start with the DVS128 class number 00-10, e.g. 05_left_arm_cw" % d.name)
        names.append(d.name)
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in (".mp4", ".mov", ".m4v", ".avi", ".mkv"):
                frames, fps = read_video(f, size, 30.0)
                bins = bin_emulated(emulate(frames, EmuConfig(theta=theta_from_ln(0.2), period_ticks=int(1e4 / fps))), down=down)
                X.append(trim_inactive(bins) if trim else bins)
                Y.append(int(m.group(1)))
    return X, np.array(Y), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", choices=["toy", "dvsg", "own"], default="toy")
    ap.add_argument("--data", default=None, help="cache folder, default data/dvsg<in_hw of the weights>")
    ap.add_argument("--clips", default="data/own_clips")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--max-samples", type=int, default=0)
    a = ap.parse_args()

    net = IntSNN.load(a.weights)
    if a.dataset == "toy":
        from model.toy_gestures import make_dataset
        X, Y = make_dataset(20, seed=1, down=net.cfg.down)
    elif a.dataset == "dvsg":
        from train.datasets import load_cache
        X, Y = load_cache(Path(a.data or "data/dvsg%d" % net.cfg.in_hw) / "test.npz")
    else:
        X, Y, names = load_own_clips(a.clips, down=net.cfg.down)
        print("classes:", names)
    if a.max_samples:
        X, Y = X[:a.max_samples], Y[:a.max_samples]

    ok = n = 0
    cfg = net.cfg
    nl = cfg.n_conv + 1                                   # conv layers + output
    spikes = np.zeros(nl)
    neurons = [ch * (cfg.in_hw >> i) ** 2 for i, ch in enumerate(cfg.channels)] + [cfg.classes]
    steps = 0
    conf = np.zeros((cfg.classes, cfg.classes), int)
    samp_ok = 0
    for x, y in zip(X, Y):
        total = np.zeros(cfg.classes, np.int64)
        for w in windows(x, a.T):
            counts, rec = net.run(w, record=True)
            total += counts
            pred = int(np.argmax(counts))
            ok += pred == y
            n += 1
            conf[y, pred] += 1
            for step in rec:
                spikes += [s.sum() for s in step[:-1]] + [step[-1].sum() if cfg.readout == "spike" else 0]
            steps += len(rec)
        samp_ok += int(np.argmax(total)) == y
    names = ["L%d" % (i + 1) for i in range(cfg.n_conv)] + ["out"]
    print("window accuracy (integer model): %.4f  (%d windows)   sample accuracy (scores summed over windows): %.4f  (%d samples)"
          % (ok / n, n, samp_ok / len(X), len(X)))
    print("spikes per inference (window of %d steps): %s   total %.0f   (readout: %s)"
          % (a.T, "  ".join("%s %.0f" % (nm, v) for nm, v in zip(names, spikes / n)), spikes.sum() / n, cfg.readout))
    print("firing fraction per neuron per step: " + "  ".join("%s %.4f" % (nm, v) for nm, v in zip(names, spikes / steps / np.array(neurons))))
    print("confusion (rows = true):\n", conf)


if __name__ == "__main__":
    main()
