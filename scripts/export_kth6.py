"""Freeze a trained KTH network into the repository for the board build and the RTL regression.

    python scripts/export_kth6.py --run train/runs/kth6_32_16_32 --bins data/kth32 --out rtl/weights/kth6

Writes into --out:
  snn_config.json, weights_int8.npz   the integer network (IntSNN.load reads this directory)
  w1.hex, w2.hex, wfc.hex             ROM images for rtl/snn_core.v
  clips.hex, labels.hex               demo clips for rtl/clip_replay.v: the FIRST sample of every class in the test split
                                      (no selection by what the network predicts), 64 steps each
  clips_expect.json                   what the Python model predicts for each clip window (the board must show the same)
  sample_bins.npz                     one real sample (8 steps are used) for the RTL regression case snn_kth6
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_snn_vectors import write_clips, write_weights  # noqa: E402
from model.snn_int import IntSNN  # noqa: E402
from train.datasets import load_cache, save_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bins", default="data/kth32", help="cache folder holding test.npz")
    ap.add_argument("--out", default="rtl/weights/kth6")
    ap.add_argument("--steps", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("snn_config.json", "weights_int8.npz"):
        shutil.copy(Path(a.run) / f, out / f)
    net = IntSNN.load(out)
    write_weights(net, out)
    write_clips(net, str(Path(a.bins) / "test.npz"), out, a.steps, 32)
    X, Y = load_cache(Path(a.bins) / "test.npz")
    save_cache(out / "sample_bins.npz", [X[0][:a.steps]], [int(Y[0])])
    print("exported to", out)


if __name__ == "__main__":
    main()
