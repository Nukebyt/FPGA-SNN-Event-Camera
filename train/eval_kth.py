"""Real-video domain-gap test on KTH hand clapping and hand waving (roadmap 3.10 without self-recorded clips).

    python -m train.eval_kth --weights train/runs/dvsg_32_16_32 --kth data/kth --sweep         # choose emulator settings on validation people
    python -m train.eval_kth --weights train/runs/dvsg_32_16_32 --kth data/kth --split test --theta-ln 0.2 --mode step

KTH video -> 128x128 gray at 30 fps -> integer event emulator -> 2x32x32 presence bins -> the network trained on
real DVS128 Gesture events. Targets: hand clapping = DVS class 0; hand waving = DVS class 1 or 2 (the KTH waving
videos do not say which hand, so either wave class counts as correct). Windows are 48 bins (1.6 s) with stride 24
and lie fully inside one repetition; a repetition is scored by the summed scores of its windows.
Settings are chosen on the validation subjects only; test subjects are touched once for the final number.
"""
import argparse
import numpy as np
import torch

from model.emulator_int import EmuConfig, emulate
from model.log_lut import theta_from_ln
from model.snn_input import bin_emulated
from train.kth import load_kth
from train.per_class import load_qat

TARGETS = {"handclapping": {0}, "handwaving": {1, 2}}
NAMES = ["clap", "Rwave", "Lwave", "Rcw", "Rccw", "Lcw", "Lccw", "roll", "drums", "guitar", "other"]


def window_starts(n, T, stride):
    if n <= T:
        return [0]
    return list(range(0, n - T + 1, stride))


def make_windows(seqs, theta_ln, mode, down, T=48, stride=24):
    """seqs: list of (frames, meta). Returns (windows float32 [W, T, 2, H, H], owner index per window, mean active fraction)."""
    wins, owner, act = [], [], []
    for si, (frames, meta) in enumerate(seqs):
        bins = bin_emulated(emulate(frames, EmuConfig(theta=theta_from_ln(theta_ln), mode=mode)), down=down)
        act.append(bins.mean())
        for s in window_starts(len(bins), T, stride):
            w = bins[s:s + T]
            if len(w) < T:
                w = np.concatenate([w, np.zeros((T - len(w),) + w.shape[1:], w.dtype)])
            wins.append(w)
            owner.append(si)
    return np.stack(wins).astype(np.float32), np.array(owner), float(np.mean(act))


@torch.no_grad()
def score(net, wins, bs=64):
    out = []
    for i in range(0, len(wins), bs):
        out.append(net(torch.from_numpy(wins[i:i + bs]).cuda()).cpu().numpy())
    return np.concatenate(out)


def evaluate(net, seqs, theta_ln, mode, down):
    wins, owner, active = make_windows(seqs, theta_ln, mode, down)
    sc = score(net, wins)
    pred = sc.argmax(1)
    res = {"active": active, "n_win": len(wins), "n_seq": len(seqs), "rows": []}
    seq_scores = np.zeros((len(seqs), sc.shape[1]))
    np.add.at(seq_scores, owner, sc)
    for si, (_, meta) in enumerate(seqs):
        ok_w = float(np.mean([pred[j] in TARGETS[meta["action"]] for j in np.nonzero(owner == si)[0]]))
        res["rows"].append((meta, ok_w, int(seq_scores[si].argmax() in TARGETS[meta["action"]])))
    res["pred"], res["owner"] = pred, owner
    return res


def summarize(res, seqs):
    out = {}
    for action in TARGETS:
        rows = [(m, w, s) for m, w, s in res["rows"] if m["action"] == action]
        if not rows:
            continue
        nw = np.array([np.sum(res["owner"] == i) for i, (_, m) in enumerate(seqs) if m["action"] == action])
        out[action] = {"win": float(np.average([w for _, w, _ in rows], weights=nw)), "seq": float(np.mean([s for _, _, s in rows])),
                       "n_seq": len(rows)}
        for cond in ("d1", "d2", "d3", "d4"):
            r = [(w, n) for (m, w, _), n in zip(rows, nw) if m["cond"] == cond]
            if r:
                out[action][cond] = float(np.average([w for w, _ in r], weights=[n for _, n in r]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--kth", default="data/kth")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--theta-ln", type=float, default=0.2)
    ap.add_argument("--mode", default="step", choices=["step", "reset"])
    ap.add_argument("--sweep", action="store_true", help="grid over emulator settings on the validation subjects, then test")
    ap.add_argument("--max-videos", type=int, default=None, help="per action and split (quick checks)")
    a = ap.parse_args()
    net, cfg = load_qat(a.weights)
    net.eval()

    def load(split):
        seqs = []
        for action in TARGETS:
            seqs += list(load_kth(a.kth, action, None if split == "all" else split, max_videos=a.max_videos))
        return seqs

    def show(tag, res, seqs):
        s = summarize(res, seqs)
        print("%-34s active input %.3f | " % (tag, res["active"]) + " | ".join(
            "%s window %.3f seq %.3f (n=%d seqs)" % (act, v["win"], v["seq"], v["n_seq"]) for act, v in s.items()))
        return s

    if a.sweep:
        val = load("val")
        print("validation: %d repetitions" % len(val), flush=True)
        best = None
        for mode in ("step", "reset"):
            for tl in (0.10, 0.15, 0.20, 0.30, 0.40):
                s = show("val  theta %.2f ln  %s" % (tl, mode), evaluate(net, val, tl, mode, cfg.down), val)
                sc = np.mean([v["win"] for v in s.values()])
                if best is None or sc > best[0]:
                    best = (sc, tl, mode)
        print("best on validation: theta %.2f ln, %s (mean window accuracy %.3f)" % (best[1], best[2], best[0]))
        test = load("test")
        print("test: %d repetitions" % len(test))
        show("test  default 0.20 ln step", evaluate(net, test, 0.20, "step", cfg.down), test)
        res = evaluate(net, test, best[1], best[2], cfg.down)
        s = show("test  chosen %.2f ln %s" % (best[1], best[2]), res, test)
    else:
        seqs = load(a.split)
        res = evaluate(net, seqs, a.theta_ln, a.mode, cfg.down)
        s = show("%s theta %.2f ln %s" % (a.split, a.theta_ln, a.mode), res, seqs)
    for action, v in s.items():
        print("  %-13s per condition (window acc): " % action + "  ".join("%s %.2f" % (c, v[c]) for c in ("d1", "d2", "d3", "d4") if c in v))
    # where do the windows go? predicted-class distribution per action (the target classes are marked with *)
    final = test if a.sweep else seqs
    pred, owner = res["pred"], res["owner"]
    for action in TARGETS:
        idx = [j for j in range(len(pred)) if final[owner[j]][1]["action"] == action]
        dist = np.bincount(pred[idx], minlength=11) / max(len(idx), 1)
        print("  %-13s predicted as: " % action + "  ".join("%s%s %.2f" % (NAMES[c], "*" if c in TARGETS[action] else "", dist[c]) for c in np.argsort(-dist)[:5]))
    print("Track A reference on real DVS events: clap recall 0.80, right wave 0.98, left wave 0.94; active input fraction 0.166 (train)")


if __name__ == "__main__":
    main()
