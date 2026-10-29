"""Weights and golden vectors for the SNN RTL (rtl/snn_core.v and friends), bit-exact to model/snn_int.py.

    python scripts/gen_snn_vectors.py weights --run train/runs/kth6_32_16_32 --out rtl/weights/kth6      # weight ROM images
    python scripts/gen_snn_vectors.py case --run <run> --bins data/kth32/test.npz --sample 5 --steps 8 --out work/vec/x
    python scripts/gen_snn_vectors.py random --out work/vec/snn_random --seed 1 --steps 8                # random net + random bins
    python scripts/gen_snn_vectors.py chain --run <run or random> --frames 6 --out work/vec/snn_chain      # pix_source -> emulator -> bins

ROM image layouts (what lif_conv.v / snn_core.v expect):
  w1.hex, w2.hex : one 64-bit word per line, address g*CIN*9 + ci*9 + ky*3 + kx, output channel g*8+j in bits [8j +: 8]
  wfc.hex        : one int8 per line, address k*FCIN + a*8 + j, a = g*64 + y*8 + x, input index (g*8+j)*64 + y*8 + x
Case files: bins.hex (64 x 32-bit words per step: word = pol*32 + y, bit x), expect_steps.hex (per step: spk1, spk2, cur[0..NCLS-1],
32-bit two's complement), expect_final.hex (pred, score[0..NCLS-1]), params.vh (C1, C2, NCLS, NSTEP).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.snn_config import SNNConfig  # noqa: E402
from model.snn_int import IntSNN  # noqa: E402


def pack_conv(w):
    cout, cin = w.shape[:2]
    assert cout % 8 == 0
    lines = []
    for g in range(cout // 8):
        for ci in range(cin):
            for ky in range(3):
                for kx in range(3):
                    word = 0
                    for j in range(8):
                        word |= (int(w[g * 8 + j, ci, ky, kx]) & 0xFF) << (8 * j)
                    lines.append("%016x" % word)
    return lines


def pack_fc(wfc, c2):
    ncls = wfc.shape[0]
    lines = []
    for k in range(ncls):
        for a in range(c2 // 8 * 64):
            g, rem = divmod(a, 64)
            for j in range(8):
                lines.append("%02x" % (int(wfc[k, (g * 8 + j) * 64 + rem]) & 0xFF))
    return lines


def write_weights(net, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = net.cfg
    assert cfg.n_conv == 2 and cfg.in_hw == 32 and cfg.readout == "sum" and cfg.leak_shift == 4 and cfg.vbits == 12, \
        "the RTL implements 2 conv layers, 32x32 input, sum readout, leak 4, 12-bit membranes"
    (out / "w1.hex").write_text("\n".join(pack_conv(net.wc[0])) + "\n")
    (out / "w2.hex").write_text("\n".join(pack_conv(net.wc[1])) + "\n")
    (out / "wfc.hex").write_text("\n".join(pack_fc(net.wfc, cfg.c2)) + "\n")


def pack_bins(bins):
    """[T, 2, 32, 32] -> lines of 32-bit words, 64 per step."""
    lines = []
    for b in bins:
        for pol in range(2):
            for y in range(32):
                word = 0
                for x in range(32):
                    word |= int(b[pol, y, x]) << x
                lines.append("%08x" % word)
    return lines


def run_model(net, bins):
    net.reset()
    exp, score = [], np.zeros(net.cfg.classes, np.int64)
    for b in bins:
        s1, s2, cur = net.step(b)
        score += cur
        exp.append([int(s1.sum()), int(s2.sum())] + [int(c) for c in cur])
    return exp, score


def pack_membranes(v):
    """v [C, H, H] int -> lines of 8 x 12-bit words (96 bit), address g*H*H + y*H + x, channel g*8+j in bits [12j +: 12]."""
    c, h, _ = v.shape
    lines = []
    for g in range(c // 8):
        for y in range(h):
            for x in range(h):
                word = 0
                for j in range(8):
                    word |= (int(v[g * 8 + j, y, x]) & 0xFFF) << (12 * j)
                lines.append("%024x" % word)
    return lines


def write_case(net, bins, out):
    out = Path(out)
    write_weights(net, out)
    bins = np.asarray(bins, np.uint8)
    exp, score = run_model(net, bins)            # leaves the final membranes in net.v
    (out / "expect_v1.hex").write_text("\n".join(pack_membranes(net.v[0])) + "\n")
    (out / "expect_v2.hex").write_text("\n".join(pack_membranes(net.v[1])) + "\n")
    (out / "bins.hex").write_text("\n".join(pack_bins(bins)) + "\n")
    (out / "expect_steps.hex").write_text("\n".join("%08x" % (v & 0xFFFFFFFF) for row in exp for v in row) + "\n")
    pred = int(np.argmax(score))
    (out / "expect_final.hex").write_text("\n".join("%08x" % (v & 0xFFFFFFFF) for v in [pred] + [int(s) for s in score]) + "\n")
    c = net.cfg
    (out / "params.vh").write_text("`define C1 %d\n`define C2 %d\n`define NCLS %d\n`define NSTEP %d\n" % (c.c1, c.c2, c.classes, len(bins)))
    return exp, score


def random_net(seed, ncls=6, c1=16, c2=32):
    rng = np.random.default_rng(seed)
    cfg = SNNConfig(in_hw=32, c1=c1, c2=c2, classes=ncls)
    w1 = np.clip(np.rint(rng.normal(0, 40, (c1, 2, 3, 3))), -128, 127).astype(np.int32)
    w2 = np.clip(np.rint(rng.normal(0, 14, (c2, c1, 3, 3))), -128, 127).astype(np.int32)
    wf = np.clip(np.rint(rng.normal(0, 6, (ncls, c2 * 64))), -128, 127).astype(np.int32)
    w1[0, 0, 0, 0] = -128; w1[1, 0, 0, 0] = 127          # extreme int8 values
    return IntSNN(cfg, w1, w2, wf)


def saturating_net(seed, ncls=6):
    """Random net whose first channels are pushed to both membrane rails (checks the +-2^11 clip and the int8 extremes)."""
    net = random_net(seed, ncls)
    net.wc[0][0:4] = 127
    net.wc[0][4:8] = -128
    net.wc[1][0:8, :, :, :] = np.where(np.arange(8)[:, None, None, None] < 4, 90, -100)
    net.wfc[0, :] = 3                                     # classes 0 and 1 identical and dominant: an exact score tie,
    net.wfc[1, :] = 3                                     # the controller must answer class 0 (np.argmax rule)
    return net


def random_bins(seed, steps, density=0.12):
    rng = np.random.default_rng(seed + 1000)
    b = (rng.random((steps, 2, 32, 32)) < density).astype(np.uint8)
    b[0] = 0                                             # an empty first frame, like the emulator's init frame
    return b


def chain_bins(frames, theta=9, mode="step"):
    from model.emulator_int import EmuConfig, emulate
    from model.pattern_source import pattern_frame
    from model.snn_input import bin_emulated
    fr = np.stack([pattern_frame(f) for f in range(frames)])
    return bin_emulated(emulate(fr, EmuConfig(theta=theta, mode=mode)), down=4)


def pack_frames(frames):
    """[N, 128, 128] uint8 -> lines of 64-bit words, little endian (byte 0 = bits [7:0]): the memory image the ARM writes."""
    lines = []
    for f in frames:
        flat = np.asarray(f, np.uint8).reshape(-1)
        for i in range(0, len(flat), 8):
            word = 0
            for j in range(8):
                word |= int(flat[i + j]) << (8 * j)
            lines.append("%016x" % word)
    return lines


def engine_frames(video, frames, start=0):
    """Frames for the engine bench: the pattern source, or a KTH video exactly as train/prep_kth.py prepares it."""
    if not video:
        from model.pattern_source import pattern_frame
        return np.stack([pattern_frame(f) for f in range(frames)])
    from train.kth import read_all_frames, resample_linear
    fr = resample_linear(read_all_frames(video, fit="squash"))
    return fr[start:start + frames]


def write_engine_case(net, video, frames, start, out, theta=9, mode="step"):
    """snn_core/chain expectations for `frames` frames through emulator theta, plus frames.hex (what software writes to DDR3)."""
    from model.emulator_int import EmuConfig, emulate
    from model.snn_input import bin_emulated
    fr = engine_frames(video, frames, start)
    assert len(fr) == frames, "video too short"
    bins = bin_emulated(emulate(fr, EmuConfig(theta=theta, mode=mode)), down=4)
    exp, score = write_case(net, bins, out)
    Path(out, "frames.hex").write_text("\n".join(pack_frames(fr)) + "\n")
    return exp, score, bins


def write_clips(net, bins_file, out, steps, window):
    """Demo clips for rtl/clip_replay.v: the first sample of every class in `bins_file` (no selection by model output),
    `steps` bins each (zero padded), clips.hex + labels.hex + clips_expect.json (the model's prediction per window)."""
    import json
    from train.datasets import load_cache
    X, Y = load_cache(bins_file)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines, labels, expect = [], [], []
    for c in range(net.cfg.classes):
        i = int(np.nonzero(Y == c)[0][0])
        b = X[i][:steps]
        if len(b) < steps:
            b = np.concatenate([b, np.zeros((steps - len(b),) + b.shape[1:], b.dtype)])
        lines += pack_bins(b)
        labels.append("%x" % c)
        wins = []
        for w0 in range(0, steps, window):
            _, sc = run_model(net, b[w0:w0 + window])
            wins.append({"pred": int(np.argmax(sc)), "scores": [int(v) for v in sc]})
        expect.append({"class": c, "sample": i, "length": int(len(X[i])), "windows": wins})
    (out / "clips.hex").write_text("\n".join(lines) + "\n")
    (out / "labels.hex").write_text("\n".join(labels) + "\n")
    (out / "clips_expect.json").write_text(json.dumps(expect, indent=1))
    (out / "clips_pred.hex").write_text(chr(10).join("%x" % w["pred"] for e in expect for w in e["windows"]) + chr(10))
    for e in expect:
        print("clip %d (sample %d, %d bins): model window predictions %s" % (e["class"], e["sample"], e["length"], [w["pred"] for w in e["windows"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weights", "case", "random", "chain", "clips", "engine"])
    ap.add_argument("--run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--video", default="", help="engine: a KTH .avi (default: the on-chip pattern)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sat", action="store_true", help="random: saturating network and dense input")
    a = ap.parse_args()
    if a.cmd == "weights":
        write_weights(IntSNN.load(a.run), a.out)
        print("weights written to", a.out)
    elif a.cmd == "case":
        from train.datasets import load_cache
        net = IntSNN.load(a.run)
        X, Y = load_cache(a.bins)
        bins = X[a.sample][:a.steps]
        exp, score = write_case(net, bins, a.out)
        print("sample %d (label %d), %d steps, model scores %s -> class %d" % (a.sample, Y[a.sample], len(bins), score.tolist(), int(np.argmax(score))))
    elif a.cmd == "clips":
        write_clips(IntSNN.load(a.run), a.bins, a.out, a.steps, a.window)
    elif a.cmd == "engine":
        net = IntSNN.load(a.run) if a.run else random_net(a.seed)
        exp, score, bins = write_engine_case(net, a.video, a.frames, a.start, a.out)
        print("engine case: %d frames (%s), events cells/frame %s, spikes %s, scores %s -> class %d" % (
            a.frames, a.video or "pattern", [int(b.sum()) for b in bins], [(e[0], e[1]) for e in exp], score.tolist(), int(np.argmax(score))))
    elif a.cmd == "random":
        net = saturating_net(a.seed) if a.sat else random_net(a.seed)
        exp, score = write_case(net, random_bins(a.seed, a.steps, 0.7 if a.sat else 0.12), a.out)
        print("random net, %d steps, spikes per step %s, scores %s" % (a.steps, [(e[0], e[1]) for e in exp], score.tolist()))
    else:
        net = IntSNN.load(a.run) if a.run else random_net(a.seed)
        bins = chain_bins(a.frames)
        exp, score = write_case(net, bins, a.out)
        print("chain: %d frames, events/frame %s, scores %s" % (a.frames, [int(b.sum()) for b in bins], score.tolist()))


if __name__ == "__main__":
    main()
