"""DVS128 Gesture -> binned 32x32 or 64x64 caches (data/dvsg32 or data/dvsg64 /{train,test}.npz).

    python -m train.prep_dvsgesture                       # raw IBM release, 32x32
    python -m train.prep_dvsgesture --down 2              # 64x64 (2x2 presence per cell)
    python -m train.prep_dvsgesture --source tonic        # tonic's preprocessed copy (downloads from figshare)

Raw source: DVSGesture/DvsGesture/DvsGesture/*.aedat + *_labels.csv + trials_to_{train,test}.txt.
Each labelled gesture window (class, start_us, end_us) becomes one sample, binned into 33,333 us steps
aligned to the window start; 128x128 -> 32x32 by 4x4 presence (OR) per polarity. Class 1..11 -> 0..10
(both arm-roll directions are class 8, as in the dataset). No coordinate flip: right-hand gestures appear
on the left of the image (x ~ 40), the normal view of a person facing the camera.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from model.snn_input import bin_real
from train.datasets import save_cache

CLASS_NAMES = ["hand_clapping", "right_hand_wave", "left_hand_wave", "right_arm_cw", "right_arm_ccw",
               "left_arm_cw", "left_arm_ccw", "arm_roll", "air_drums", "air_guitar", "other"]


def prep_raw(raw_dir, out_dir, bin_us, down=4):
    from train.aedat import read_aedat, read_labels
    raw = Path(raw_dir)
    for split in ("train", "test"):
        names = [l.strip() for l in (raw / ("trials_to_%s.txt" % split)).read_text().split() if l.strip()]
        samples, labels, index, missing = [], [], [], []
        for k, name in enumerate(names):
            f = raw / name
            lab = raw / name.replace(".aedat", "_labels.csv")
            if not f.exists() or not lab.exists():
                missing.append(name)
                continue
            ev = read_aedat(f)
            t = ev["t"]
            for cls, a, b in read_labels(lab):
                i0, i1 = np.searchsorted(t, [a, b])
                n_bins = int(np.ceil((b - a) / bin_us))
                s = bin_real(ev["x"][i0:i1], ev["y"][i0:i1], ev["p"][i0:i1], t[i0:i1], bin_us=bin_us,
                             down=down, n_bins=n_bins, t0=a)
                samples.append(s)
                labels.append(cls - 1)
                index.append({"file": name, "class": cls, "start_us": int(a), "end_us": int(b), "events": int(i1 - i0)})
            print("%s %3d/%d %-34s samples so far %d" % (split, k + 1, len(names), name, len(samples)), flush=True)
        save_cache(Path(out_dir) / (split + ".npz"), samples, labels)
        (Path(out_dir) / ("index_%s.json" % split)).write_text(json.dumps(index, indent=0))
        per_class = np.bincount(labels, minlength=11)
        print("== %s: %d samples from %d trials (%d missing: %s)" % (split, len(samples), len(names) - len(missing), len(missing), missing))
        print("   per class:", {CLASS_NAMES[i]: int(c) for i, c in enumerate(per_class)})
        print("   bins/sample: mean %.1f min %d max %d;  active input fraction %.4f"
              % (np.mean([len(s) for s in samples]), min(len(s) for s in samples), max(len(s) for s in samples),
                 np.mean([s.mean() for s in samples])))


def prep_tonic(root, out_dir, bin_us, down=4):
    import tonic
    for split, train in (("train", True), ("test", False)):
        ds = tonic.datasets.DVSGesture(save_to=root, train=train)
        samples, labels = [], []
        for i in range(len(ds)):
            ev, lab = ds[i]
            samples.append(bin_real(ev["x"], ev["y"], ev["p"].astype(np.uint8), ev["t"], bin_us=bin_us, down=down))
            labels.append(lab)
        save_cache(Path(out_dir) / (split + ".npz"), samples, labels)
        print("wrote", split, len(samples), "samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["raw", "tonic"], default="raw")
    ap.add_argument("--raw-dir", default="DVSGesture/DvsGesture/DvsGesture")
    ap.add_argument("--root", default="data/dvsgesture", help="tonic download folder (source=tonic)")
    ap.add_argument("--down", type=int, choices=[2, 4], default=4, help="4: 128 -> 32x32 (data/dvsg32), 2: 128 -> 64x64 (data/dvsg64)")
    ap.add_argument("--out", default=None, help="default data/dvsg<128/down>")
    ap.add_argument("--bin-us", type=int, default=33333)
    a = ap.parse_args()
    out = a.out or "data/dvsg%d" % (128 // a.down)
    if a.source == "raw":
        prep_raw(a.raw_dir, out, a.bin_us, a.down)
    else:
        prep_tonic(a.root, out, a.bin_us, a.down)


if __name__ == "__main__":
    main()
