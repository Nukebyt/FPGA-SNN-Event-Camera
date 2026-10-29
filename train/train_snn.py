"""Train the QAT SNN (or the ANN baseline) and export int8 weights.

    python -m train.train_snn --dataset toy --epochs 30 --out train/runs/toy          # no downloads
    python -m train.train_snn --dataset dvsg --epochs 60 --out train/runs/dvsg        # after prep_dvsgesture
    python -m train.train_snn --dataset kth --T 32 --out train/runs/kth        # after train.prep_kth (6 KTH actions)
    python -m train.train_snn --dataset dvsg --ann --out train/runs/dvsg_ann          # baseline (3.4)
    python -m train.train_snn --dataset dvsg --hw 64 --out train/runs/dvsg64        # 64x64 input (needs prep --down 2); not better, see docs/results

Reports window accuracy (every non-overlapping window of every test sample) and sample accuracy
(spike counts summed over the sample's windows). Hardware decides per window, so window accuracy is
the honest number; sample accuracy is what most papers report.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.snn_config import SNNConfig
from train.datasets import FLIP_MAP_DVS, WindowDataset, load_cache
from train.snn_qat import ANNBaseline, QATSNN, export_int


def get_data(a):
    if a.dataset == "toy":
        from model.toy_gestures import CLASSES, make_dataset
        down = 128 // a.hw
        X, Y = make_dataset(a.toy_per_class, seed=0, down=down)
        Xt, Yt = make_dataset(max(a.toy_per_class // 3, 6), seed=1, down=down)
        return X, Y, Xt, Yt, len(CLASSES)
    if a.dataset == "kth":     # monitor on the validation people; the test people are scored once, by train.eval_kth6
        data = a.data or "data/kth%d" % a.hw
        classes = json.loads((Path(data) / "prep.json").read_text())["classes"]
        X, Y = load_cache(Path(data) / "train.npz")
        Xt, Yt = load_cache(Path(data) / "val.npz")
        return X, Y, Xt, Yt, len(classes)
    data = a.data or "data/dvsg%d" % a.hw
    X, Y = load_cache(Path(data) / "train.npz")
    Xt, Yt = load_cache(Path(data) / "test.npz")
    assert X[0].shape[-1] == a.hw, "cache %s is %dx%d, --hw is %d" % (data, X[0].shape[-1], X[0].shape[-1], a.hw)
    return X, Y, Xt, Yt, 11


@torch.no_grad()
def evaluate(model, loader, n_classes, dev, ann, scale):
    model.eval()
    win_ok = win_n = 0
    per_sample = {}
    labels = {}
    for x, y, sid in loader:
        x = x.to(dev)
        out = model(x)
        pred = out.argmax(1).cpu()
        win_ok += int((pred == y).sum())
        win_n += len(y)
        for o, l, s in zip(out.cpu().numpy(), y.numpy(), sid.numpy()):
            per_sample[s] = per_sample.get(s, 0) + o
            labels[s] = l
    samp = np.mean([int(np.argmax(per_sample[s]) == labels[s]) for s in per_sample])
    return win_ok / win_n, float(samp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["toy", "dvsg", "kth"], default="toy")
    ap.add_argument("--hw", type=int, choices=[32, 64], default=32, help="SNN input size (128 -> hw by presence pooling)")
    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--c3", type=int, default=0, help="channels of an optional third conv layer (0 = none)")
    ap.add_argument("--data", default=None, help="cache folder, default data/dvsg<hw>")
    ap.add_argument("--toy-per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--logit-scale", type=float, default=None, help="default: 0.25 for spike readout, 10 / (T * thr_out) for sum readout")
    ap.add_argument("--readout", choices=["sum", "spike"], default="sum")
    ap.add_argument("--ann", action="store_true")
    ap.add_argument("--init-std", default="auto", help="initial int-weight std per layer 'a,b,c', or 'auto' = calibrate layer 1/2 to ~8%%/10%% firing on real training windows")
    ap.add_argument("--aug-drop", type=float, default=0.0, help="event dropout probability per active input cell")
    ap.add_argument("--aug-flip", action="store_true", help="mirror windows left/right with the DVS128 class swaps (dvsg only)")
    ap.add_argument("--aug-speed", type=float, default=0.0, help="probability of a 2x faster (OR-merged) window")
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay on the latent weights")
    ap.add_argument("--float", action="store_true", help="float weights (no int8 rounding), for the 3.3 float-vs-QAT comparison")
    ap.add_argument("--out", default="train/runs/toy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    X, Y, Xt, Yt, ncls = get_data(a)
    cfg = SNNConfig(in_hw=a.hw, c1=a.c1, c2=a.c2, c3=a.c3, classes=ncls, readout=a.readout)
    print("config:", cfg.to_json().replace(chr(10), " "), flush=True)
    print("sizes:", cfg.sizes(), flush=True)
    dev = torch.device(a.device)
    flip_map = FLIP_MAP_DVS if (a.aug_flip and a.dataset == "dvsg") else None
    tr = DataLoader(WindowDataset(X, Y, a.T, True, seed=a.seed, drop=a.aug_drop, flip_map=flip_map, speed=a.aug_speed),
                    batch_size=a.bs, shuffle=True, drop_last=True)
    te = DataLoader(WindowDataset(Xt, Yt, a.T, False), batch_size=32)
    if a.init_std == "auto":
        from train.snn_qat import calibrate_init
        xb = torch.stack([tr.dataset[i][0] for i in range(0, len(tr.dataset), max(len(tr.dataset) // 12, 1))]).to(dev)
        init_std = calibrate_init(cfg, xb)
        print("calibrated init std (int units, conv layers then fc):", ", ".join("%.2f" % v for v in init_std), flush=True)
    else:
        init_std = tuple(float(v) for v in a.init_std.split(","))
    model = (ANNBaseline(cfg) if a.ann else QATSNN(cfg, init_std_int=init_std, quantize=not a.float)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd) if a.wd > 0 else torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    if a.ann:
        scale = 1.0
    elif a.logit_scale is not None:
        scale = a.logit_scale
    else:
        scale = 0.25 if a.readout == "spike" else 10.0 / (a.T * cfg.thr[-1])
    print("readout %s, logit scale %.3g" % (a.readout, scale), flush=True)
    log = []
    for ep in range(a.epochs):
        model.train()
        tot = ok = n = 0
        for x, y, _ in tr:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = F.cross_entropy(out * scale, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            ok += int((out.argmax(1) == y).sum())
            n += len(y)
        sched.step()
        wa, sa = evaluate(model, te, ncls, dev, a.ann, scale)
        log.append(dict(epoch=ep, loss=tot / n, train_acc=ok / n, test_window_acc=wa, test_sample_acc=sa))
        print("ep %3d  loss %.3f  train %.3f  test(window) %.3f  test(sample) %.3f" % (ep, tot / n, ok / n, wa, sa), flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.json").write_text(json.dumps(log, indent=1))
    if not a.ann and not a.float:
        export_int(model, out)
        print("exported int8 weights + config to", out)


if __name__ == "__main__":
    main()
