"""Regenerate every figure and GIF used by README.md into docs/media/.

    python scripts/make_readme_media.py                    # everything, answers from the bit-exact Python models ("sim")
    python scripts/make_readme_media.py --only figs        # figures only (no video decoding)
    python scripts/make_readme_media.py --only gifs --connect 192.168.137.200   # GIFs with the REAL board's answers (suffix _board)

The numbers in the accuracy figures come from the logs in docs/results/ and train/runs/ (paths in the code); the GIFs are produced by
running KTH test-person videos through the same code path as scripts/snn_viewer.py. With --connect the per-frame answers (scores,
spike counts, event counts) come from the board over Ethernet instead of the model; the event picture is always drawn from the model
(the board returns counts, not event images).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

MEDIA = ROOT / "docs" / "media"
CLASSES = ["walking", "jogging", "running", "boxing", "hand waving", "hand clapping"]
CCOL = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#F58518", "#B279A2"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans"})


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", MEDIA / name)


# ----------------------------------------------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc="#EAF1FB", ec="#2F5597", fs=8.5, weight="normal", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=tc, linespacing=1.25)


def arrow(ax, p, q, text="", color="#333333", fs=7.5, off=(0, 0.09), style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.3, color=color, connectionstyle="arc3,rad=%g" % rad))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=fs, color=color)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 34)
    ax.set_ylim(-0.4, 15.5)
    ax.axis("off")

    def zone(x, w, title, fc, ec):
        ax.add_patch(FancyBboxPatch((x, 0.3), w, 14.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, 14.05, title, ha="center", fontweight="bold", fontsize=10.5)

    zone(0.2, 4.6, "LAPTOP", "#F7F7F7", "#888")
    zone(5.3, 4.6, "HPS (ARM, Linux)", "#FFF6E5", "#C47F00")
    zone(10.4, 2.6, "DDR3", "#F1ECF8", "#6B4C9A")
    zone(13.5, 20.3, "FPGA fabric (Cyclone V, one 25 MHz clock)", "#EEF7EE", "#2E7D32")

    box(ax, 0.6, 9.4, 3.8, 3.4, "video file / webcam\n\ngrey, squash 128x128,\nresample to 30 fps\n\nscripts/snn_viewer.py", fc="#FFFFFF", ec="#555", fs=8.3)
    box(ax, 0.6, 5.2, 3.8, 3.2, "viewer window\n\nframe | DVS events |\n6 running scores |\nanswer of last window", fc="#FFFFFF", ec="#555", fs=8.3)
    box(ax, 0.6, 1.0, 3.8, 3.4, "scripts/check_board.py\n\nsame frames through the\nboard and the Python\nmodels: answers must\nbe identical", fc="#FFFFFF", ec="#555", fs=8.3)
    box(ax, 5.7, 9.4, 3.8, 3.4, "sw/snn_streamer.c\nTCP port 5002\n\nframe -> ring slot,\nIN_WR_SEQ += 1,\nRESULT <- CSR snapshot", fc="#FFFFFF", ec="#C47F00", fs=8.3)
    box(ax, 5.7, 2.0, 3.8, 4.4, "lightweight bridge\n0xFF210000\nCSR block, 32-bit\n\nF2H SDRAM port 0\n64-bit reads", fc="#FFFFFF", ec="#C47F00", fs=8.3)
    box(ax, 10.8, 7.2, 1.8, 5.6, "input\nring\n\n4 slots\nx 16 KiB\n\n@0x30000000\n(Linux uses\nonly 768 MB)", fc="#FFFFFF", ec="#6B4C9A", fs=7.8)

    ytop, h = 9.4, 3.4
    box(ax, 14.0, ytop, 3.0, h, "dma_rd\n+ byte FIFO\n\n2048 x 64-bit\nwords / frame", fc="#FFFFFF", ec="#2E7D32", fs=8.3)
    box(ax, 17.5, ytop, 5.0, h, "event_emulator\n\nlog LUT (8 bit), ref RAM\n|L - ref| >= theta\n-> ON / OFF event\n<= 1 event / pixel", fc="#E3F2E1", ec="#2E7D32", weight="bold", fs=8.3)
    box(ax, 23.0, ytop, 3.8, h, "bin_builder\n\n2 x 32 x 32 bits\nper frame\n2 banks, frame_sync", fc="#FFFFFF", ec="#2E7D32", fs=8.3)
    box(ax, 27.3, ytop, 6.0, h, "snn_core\n\nconv1 2->16 LIF, OR-pool\nconv2 16->32 LIF, OR-pool\nfc 2048->6 (int8)\n205,832 clocks / step", fc="#E3F2E1", ec="#2E7D32", weight="bold", fs=8.3)
    for x1, x2 in [(17.0, 17.5), (22.5, 23.0), (26.8, 27.3)]:
        arrow(ax, (x1, ytop + h / 2), (x2, ytop + h / 2))
    ax.text(20.0, ytop - 0.55, "32-bit event words", ha="center", fontsize=7.5, color="#333")
    ax.text(24.9, ytop - 0.55, "bits", ha="center", fontsize=7.5, color="#333")
    ax.text(15.5, ytop - 0.55, "bytes", ha="center", fontsize=7.5, color="#333")

    box(ax, 14.0, 4.6, 8.0, 2.6, "snn_engine scheduler\n\nstarts a frame when the previous result is final;\nreads 16,384 pixels; publishes a consistent CSR snapshot", fc="#FFFFFF", ec="#2E7D32", fs=8.3)
    box(ax, 27.3, 4.6, 6.0, 2.6, "snn_ctrl\n\nwindow T = 32 frames: sum of class currents,\nargmax (ties -> lowest class)", fc="#FFFFFF", ec="#2E7D32", fs=8.3)
    box(ax, 14.0, 1.0, 19.3, 2.2, "CSR block: class | 6 window scores | 6 class currents | spike counts | events | cycle counters | status", fc="#FFFFFF", ec="#2E7D32", fs=8.3)

    arrow(ax, (4.4, 11.1), (5.7, 11.1), "FRAME_IN\n16 KiB", fs=7.2, off=(0, 0.1))
    arrow(ax, (9.5, 11.1), (10.8, 10.8), "memcpy", fs=7.2, off=(0, 0.12))
    arrow(ax, (12.6, 10.6), (14.0, 11.1), "DMA read", fs=7.2, off=(0, 0.14))
    arrow(ax, (30.3, ytop), (30.3, 7.2), "class currents", fs=7.2, off=(1.4, -0.15))
    arrow(ax, (18.0, 7.2), (18.0, ytop - 0.9), "start / base", fs=7.2, off=(-1.3, -0.2))
    arrow(ax, (30.3, 4.6), (30.3, 3.2))
    arrow(ax, (18.0, 4.6), (18.0, 3.2))
    arrow(ax, (14.0, 2.1), (9.5, 3.6), "", rad=-0.1)
    arrow(ax, (5.7, 4.5), (4.4, 6.6), "RESULT 112 B\n(TCP)", fs=7.2, off=(-0.3, 0.1))
    arrow(ax, (7.6, 9.4), (7.6, 6.4), "CSR reads", fs=7.2, off=(1.1, -0.1))
    ax.text(17.0, -0.15, "one frame = one SNN step: 0.7 ms to stream the pixels + 8.2 ms network = 8.9 ms in simulation (about 110 frames/s possible); the laptop sends 30 frames/s",
            ha="center", fontsize=8.7, style="italic", color="#444")
    ax.set_title("System architecture: video clips from the laptop to a spiking network on the FPGA", fontsize=13.5, fontweight="bold")
    save(fig, "fig_architecture.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 8)
    ax.axis("off")
    items = [("video frame\n128x128x8 bit\n16 KiB", "#FFFFFF"), ("log LUT\nL = 255*log2(1+p)/8", "#E3F2E1"), ("compare with\nper-pixel reference\n|L - ref| >= theta", "#E3F2E1"),
             ("event word\n{pol, y, x, ts}\n32 bit, <= 16384/frame", "#FFFFFF"), ("presence bins\n2 x 32 x 32 bits\n(4x4 cells, OR)", "#E3F2E1"),
             ("conv1 2->16\nLIF + OR-pool\n16x16x16", "#E3F2E1"), ("conv2 16->32\nLIF + OR-pool\n32x8x8", "#E3F2E1"), ("fc 2048->6\nclass currents\n(sum over 32 steps)", "#E3F2E1"), ("argmax\nclass", "#FFFFFF")]
    w, g = 2.9, 0.35
    for i, (t, c) in enumerate(items):
        x = 0.3 + i * (w + g)
        box(ax, x, 2.6, w, 3.0, t, fc=c, ec="#2E7D32" if c != "#FFFFFF" else "#555", fs=8.3)
        if i:
            arrow(ax, (x - g, 4.1), (x, 4.1))
    ax.text(15, 7.3, "Data path, one time step = one video frame (33.3 ms at 30 fps)", ha="center", fontsize=12, fontweight="bold")
    ax.text(15, 1.3, "green = RTL blocks that are bit-exact to the Python golden models   |   state kept between frames: reference RAM (emulator), membranes (SNN)",
            ha="center", fontsize=8.5, color="#444")
    ax.text(15, 0.5, "typical load (KTH): 80-140 event cells of 2048 per frame -> ~400-1000 layer-1 spikes of 16,384 neurons per step", ha="center", fontsize=8.5, color="#444")
    save(fig, "fig_pipeline.png")


def fig_snn_arch():
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 10)
    ax.axis("off")
    stages = [("input\n2 x 32 x 32\nbits", 2.0), ("conv 3x3\n2 -> 16\nLIF", 3.2), ("OR-pool 2x2\n16 x 16 x 16", 3.0), ("conv 3x3\n16 -> 32\nLIF", 3.2), ("OR-pool 2x2\n32 x 8 x 8\n= 2048 bits", 3.2),
              ("fc\n2048 -> 6\nint8", 2.8), ("sum over\nT = 32 steps", 2.8), ("argmax\nclass", 2.2)]
    x = 0.3
    for i, (t, w) in enumerate(stages):
        fc = "#E3F2E1" if "LIF" in t or "fc" in t else "#FFFFFF"
        box(ax, x, 4.8, w, 3.0, t, fc=fc, ec="#2E7D32", fs=8.5)
        if i:
            arrow(ax, (x - 0.3, 6.3), (x, 6.3))
        x += w + 0.35
    ax.text(14, 9.3, "Spiking network (integer only, 17,184 int8 weights, no biases)", ha="center", fontsize=12, fontweight="bold")
    ax.text(14, 3.5, "LIF neuron, per time step:     v <- v - (v >> 4)     v <- clip(v + sum of active int8 weights, -2048, 2047)     spike = (v >= 64)     v <- v - 64 * spike",
            ha="center", fontsize=10, family="monospace")
    ax.text(14, 2.3, "weights: conv1 288, conv2 4,608, fc 12,288    |    state: 16,384 + 8,192 membranes x 12 bit    |    readout: non-spiking, scores = sum of fc currents (design decision D10)",
            ha="center", fontsize=8.7, color="#444")
    ax.text(14, 1.2, "training: quantisation-aware, straight-through estimator, fast-sigmoid surrogate gradient; the PyTorch forward pass is bit-exact to model/snn_int.py", ha="center", fontsize=8.7, color="#444")
    save(fig, "fig_snn_arch.png")


# ----------------------------------------------------------------------------------------------------------------- results figures
def fig_trackA():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    # Track A model comparison (docs/results/trackA_accuracy.md, 8/16-channel network)
    names = ["ANN baseline\n(float)", "SNN float", "SNN int8\n(QAT)", "SNN int8\ninteger model"]
    win, smp = [81.9, 79.6, 80.9, 80.9], [88.5, 87.8, 88.2, 88.2]
    x = np.arange(4)
    axs[0].bar(x - 0.2, win, 0.4, label="per window", color="#4C78A8")
    axs[0].bar(x + 0.2, smp, 0.4, label="per sample", color="#F58518")
    for i in range(4):
        axs[0].text(x[i] - 0.2, win[i] + 0.3, "%.1f" % win[i], ha="center", fontsize=8)
        axs[0].text(x[i] + 0.2, smp[i] + 0.3, "%.1f" % smp[i], ha="center", fontsize=8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(names, fontsize=8.5)
    axs[0].set_ylim(70, 95)
    axs[0].set_ylabel("test accuracy (%)")
    axs[0].set_title("DVS128 Gesture (real events), 11 classes, 8/16 channels")
    axs[0].legend(loc="lower right")
    # size study
    labels = ["32x32\n8/16", "32x32\n16/32 (s0)", "32x32\n16/32 (s1)", "64x64\n8/16", "64x64\n16/32", "3 conv\n16/32/32"]
    w2 = [80.9, 82.8, 82.6, 78.6, 81.3, 82.9]
    s2 = [88.2, 90.3, 89.2, 86.5, 90.3, 90.3]
    x = np.arange(len(labels))
    axs[1].bar(x - 0.2, w2, 0.4, color="#4C78A8", label="per window")
    axs[1].bar(x + 0.2, s2, 0.4, color="#F58518", label="per sample")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, fontsize=8)
    axs[1].set_ylim(70, 95)
    axs[1].set_title("Network size study: a plateau at ~83% / 90%")
    axs[1].legend(loc="lower right")
    fig.suptitle("Track A: the network on real DVS recordings (reference results)", fontweight="bold")
    save(fig, "fig_trackA.png")


def fig_training():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for run, label, col in [("kth6_32_16_32", "KTH 6 classes", "#E45756"), ("kth3_32_16_32", "KTH 3 classes", "#4C78A8")]:
        L = json.load(open(ROOT / "train" / "runs" / run / "log.json"))
        ep = [r["epoch"] + 1 for r in L]
        axs[0].plot(ep, [r["train_acc"] * 100 for r in L], color=col, ls="--", label=label + " train (windows)")
        axs[0].plot(ep, [r["test_window_acc"] * 100 for r in L], color=col, label=label + " validation people")
        axs[1].plot(ep, [r["loss"] for r in L], color=col, label=label)
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("accuracy (%)")
    axs[0].set_title("Quantisation-aware training on emulated KTH events")
    axs[0].legend(fontsize=8, loc="lower right")
    axs[0].set_ylim(40, 100)
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("cross-entropy loss")
    axs[1].set_yscale("log")
    axs[1].set_title("Training loss")
    axs[1].legend(fontsize=8)
    save(fig, "fig_training.png")


def parse_confusion(path):
    txt = Path(path).read_text()
    rows = re.findall(r"\[\s*\[?\s*([\d\s]+?)\]", txt.split("sample confusion")[1])
    mat = np.array([[int(v) for v in r.split()] for r in rows[:6]])
    return mat


def fig_kth_results():
    conf = parse_confusion(ROOT / "docs" / "results" / "kth6_test_eval.log")
    txt = (ROOT / "docs" / "results" / "kth6_test_eval.log").read_text()
    rec = re.findall(r"^\s+(\w+)\s+([\d.]+) / ([\d.]+)\s+\(n=(\d+)\)", txt, flags=re.M)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    names = [r[0] for r in rec]
    win = [float(r[1]) * 100 for r in rec]
    smp = [float(r[2]) * 100 for r in rec]
    x = np.arange(6)
    axs[0].bar(x - 0.2, win, 0.4, color="#4C78A8", label="per 1 s window")
    axs[0].bar(x + 0.2, smp, 0.4, color="#F58518", label="per repetition")
    for i in range(6):
        axs[0].text(x[i] - 0.2, win[i] + 1, "%.0f" % win[i], ha="center", fontsize=8)
        axs[0].text(x[i] + 0.2, smp[i] + 1, "%.0f" % smp[i], ha="center", fontsize=8)
    axs[0].axhline(72.7, color="#4C78A8", ls=":", lw=1)
    axs[0].axhline(79.9, color="#F58518", ls=":", lw=1)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(names, rotation=20, fontsize=9)
    axs[0].set_ylim(0, 118)
    axs[0].set_ylabel("recall (%)")
    axs[0].set_title("Per-class recall, KTH test people (dotted: overall 72.7% / 79.9%)")
    axs[0].legend(loc="lower left", fontsize=8)
    cn = conf / conf.sum(axis=1, keepdims=True)
    im = axs[1].imshow(cn, cmap="Blues", vmin=0, vmax=1)
    axs[1].grid(False)
    for i in range(6):
        for j in range(6):
            axs[1].text(j, i, str(conf[i, j]), ha="center", va="center", color="white" if cn[i, j] > 0.55 else "black", fontsize=9)
    axs[1].set_xticks(range(6))
    axs[1].set_yticks(range(6))
    axs[1].set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8.5)
    axs[1].set_yticklabels(CLASSES, fontsize=8.5)
    axs[1].set_xlabel("predicted")
    axs[1].set_ylabel("true")
    axs[1].set_title("Confusion matrix by repetition (counts), test people")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
    save(fig, "fig_kth_results.png")


def fig_domain_gap():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    groups = ["hand clapping", "hand waving"]
    # per-window recall: DVS net on DVS (docs/results/trackA_accuracy.md), DVS net on emulated KTH (kth_domain_gap.md), KTH net on KTH test (kth6_accuracy.md)
    dvs_on_dvs = [80, 96]            # clap 0.80, waves 0.98 / 0.94 (mean 0.96)
    dvs_on_kth = [20, 53]            # chosen validation setting 0.30 ln reset: clap 0.203, wave 0.529
    kth_on_kth = [72.8, 81.0]
    x = np.arange(2)
    for k, (v, lab, col) in enumerate([(dvs_on_dvs, "trained on real DVS events,\ntested on real DVS events", "#54A24B"),
                                       (dvs_on_kth, "trained on real DVS events,\ntested on emulated KTH video", "#E45756"),
                                       (kth_on_kth, "trained on emulated KTH video,\ntested on emulated KTH test people", "#4C78A8")]):
        axs[0].bar(x + (k - 1) * 0.27, v, 0.27, label=lab, color=col)
        for i in range(2):
            axs[0].text(x[i] + (k - 1) * 0.27, v[i] + 1.5, "%.0f" % v[i], ha="center", fontsize=9)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(groups)
    axs[0].set_ylim(0, 150)
    axs[0].set_ylabel("per-window recall (%)")
    axs[0].set_title("The domain gap: a DVS-trained network does not transfer")
    axs[0].legend(fontsize=7.5, loc="upper center", ncol=1)
    # emulator threshold sweep on validation people (docs/results/kth_domain_gap_raw.log), step mode and reset mode
    th = [0.10, 0.15, 0.20, 0.30, 0.40]
    clap_step = [17.6, 19.9, 16.2, 10.3, 7.1]
    wave_step = [42.0, 51.6, 57.7, 63.1, 63.7]
    act = [12.4, 8.2, 6.2, 4.1, 3.2]
    axs[1].plot(th, clap_step, "o-", color="#B279A2", label="hand clapping")
    axs[1].plot(th, wave_step, "o-", color="#F58518", label="hand waving")
    axs[1].set_xlabel("emulator threshold theta (ln units)")
    axs[1].set_ylabel("DVS-trained network, window accuracy (%)")
    axs[1].set_ylim(0, 80)
    ax2 = axs[1].twinx()
    ax2.plot(th, act, "s--", color="#888", label="active input cells (%)")
    ax2.set_ylabel("active input cells (%), real DVS training data: 16.6%")
    ax2.set_ylim(0, 20)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    axs[1].set_title("No emulator threshold fixes it (validation people, step mode)")
    h1, l1 = axs[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    save(fig, "fig_domain_gap.png")


def fig_hardware():
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
    # cycles per step
    parts = [("layer 1\n2048 neurons-groups x 21", 43008), ("layer 2\n1024 x 147", 150528), ("classifier\n6 x 2048", 12288)]
    axs[0].barh([p[0] for p in parts][::-1], [p[1] for p in parts][::-1], color=["#B279A2", "#54A24B", "#4C78A8"])
    for i, p in enumerate(parts[::-1]):
        axs[0].text(p[1] + 2000, i, "%s clocks\n%.2f ms" % (format(p[1], ","), p[1] / 25000.0), va="center", fontsize=8)
    axs[0].set_xlim(0, 230000)
    axs[0].set_xlabel("clocks at 25 MHz")
    axs[0].set_title("One network step = 205,832 clocks = 8.2 ms")
    # frame budget
    axs[1].barh(["frame period\n(30 fps)", "engine work\nper frame"], [33.33, 8.9], color=["#CCCCCC", "#4C78A8"])
    axs[1].barh(["engine work\nper frame"], [0.65], color="#F58518")
    axs[1].text(33.5, 0, "33.3 ms", va="center", fontsize=9)
    axs[1].text(9.3, 1, "8.9 ms: 0.65 pixels (orange) + 8.2 SNN", va="center", fontsize=8)
    axs[1].set_xlim(0, 40)
    axs[1].set_xlabel("ms")
    axs[1].set_title("Time budget: the network uses 27% of a frame")
    # resources
    names = ["VGA demo\n(emulator)", "snn_demo\n(JTAG)", "snn_soc\n(+HPS, DMA)"]
    alm = [288, 1694, 5243]
    m10k = [27, 175, 79]
    x = np.arange(3)
    axs[2].bar(x - 0.2, [a / 41910 * 100 for a in alm], 0.4, color="#4C78A8", label="ALMs (of 41,910)")
    axs[2].bar(x + 0.2, [m / 553 * 100 for m in m10k], 0.4, color="#F58518", label="M10K (of 553)")
    for i in range(3):
        axs[2].text(x[i] - 0.2, alm[i] / 41910 * 100 + 0.6, "%s" % format(alm[i], ","), ha="center", fontsize=8)
        axs[2].text(x[i] + 0.2, m10k[i] / 553 * 100 + 0.6, str(m10k[i]), ha="center", fontsize=8)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(names, fontsize=8.5)
    axs[2].set_ylabel("% of the device")
    axs[2].set_ylim(0, 42)
    axs[2].set_title("Cyclone V 5CSXFC6: resources (Quartus 21.1)")
    axs[2].legend(fontsize=8)
    save(fig, "fig_hardware.png")


def fig_board():
    """Numbers measured on the board on 2026-10-29 (docs/results/board_run.md)."""
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    axs[0].bar(["first run\n(two TCP writes\nper reply)", "after the fix\n(one write)"], [48.7, 11.7], color=["#E45756", "#54A24B"])
    for i, v in enumerate([48.7, 11.7]):
        axs[0].text(i, v + 1, "%.1f ms" % v, ha="center")
    axs[0].axhline(33.3, color="#444", ls=":")
    axs[0].text(1.45, 34.3, "30 fps budget", ha="right", fontsize=8)
    axs[0].set_ylabel("frame round trip, laptop -> board -> laptop (ms)")
    axs[0].set_title("Fixing a 40 ms TCP delay (bug 31)")
    axs[1].bar(["first run", "after the fix", "paced at\n30 fps"], [20.5, 84.7, 29.4], color=["#E45756", "#54A24B", "#4C78A8"])
    for i, v in enumerate([20.5, 84.7, 29.4]):
        axs[1].text(i, v + 1.5, "%.1f" % v, ha="center")
    axs[1].set_ylabel("frames per second")
    axs[1].set_title("Throughput on the real board")
    axs[2].bar(["simulation", "board (min)", "board (max)"], [222233, 222236, 222245], color=["#B279A2", "#4C78A8", "#4C78A8"])
    axs[2].set_ylim(0, 260000)
    for i, v in enumerate([222233, 222236, 222245]):
        axs[2].text(i, v + 4000, format(v, ","), ha="center", fontsize=9)
    axs[2].set_ylabel("clocks, DMA start to published result")
    axs[2].set_title("Latency: board = simulation + 3 to 12 clocks (8.89 ms)")
    fig.suptitle("Measured on the DE10-Standard (SNN SoC, 25 MHz); 3,600 frames compared with the model: 0 differ", fontweight="bold")
    save(fig, "fig_board.png")


def fig_arm_power():
    """ARM-only baseline (docs/results/arm_baseline.md) and Quartus power estimates (docs/results/power.md)."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    # time per frame
    labels = ["FPGA engine\n(25 MHz, whole frame:\nemulator + network)", "ARM Cortex-A9\ndense, same algorithm\nas the FPGA", "ARM Cortex-A9\nevent driven\n(sparse)"]
    emu = [0.655, 0.284, 0.286]
    net = [8.234, 6.145, 1.139]
    x = np.arange(3)
    axs[0].bar(x, emu, color="#F58518", label="event emulator (pixels -> events)")
    axs[0].bar(x, net, bottom=emu, color="#4C78A8", label="spiking network")
    for i in range(3):
        axs[0].text(x[i], emu[i] + net[i] + 0.2, "%.2f ms\n%.0f frames/s" % (emu[i] + net[i], 1000 / (emu[i] + net[i])), ha="center", fontsize=8.5)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, fontsize=8)
    axs[0].set_ylim(0, 11.5)
    axs[0].set_ylabel("time per frame (ms)")
    axs[0].set_title("Same function, bit-exact on all three: who is faster?")
    axs[0].legend(fontsize=8, loc="upper right")
    # power estimate
    names = ["snn_demo\n(JTAG, no HPS)", "snn_soc\n(HPS + DMA + engine)"]
    stat = [417.07, 416.44]
    dyn = [35.85, 65.51]
    io = [27.91, 48.95]
    x = np.arange(2)
    axs[1].bar(x, stat, color="#BBBBBB", label="device static")
    axs[1].bar(x, dyn, bottom=stat, color="#E45756", label="core dynamic")
    axs[1].bar(x, io, bottom=np.add(stat, dyn), color="#54A24B", label="I/O")
    for i in range(2):
        axs[1].text(x[i], stat[i] + dyn[i] + io[i] + 8, "%.0f mW" % (stat[i] + dyn[i] + io[i]), ha="center")
        axs[1].text(x[i], stat[i] / 2, "%.0f" % stat[i], ha="center", fontsize=8)
        axs[1].text(x[i], stat[i] + dyn[i] / 2, "%.0f" % dyn[i], ha="center", fontsize=8, color="white")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names)
    axs[1].set_ylim(0, 640)
    axs[1].set_ylabel("estimated power (mW)")
    axs[1].set_title("Quartus power estimate (vectorless, low confidence;\nthe ARM cores are NOT included)")
    axs[1].legend(fontsize=8, loc="upper left")
    # why the FPGA engine is slow: levers
    axs[2].axis("off")
    axs[2].text(0.0, 0.97, "Reading the comparison", fontsize=11, fontweight="bold", va="top")
    axs[2].text(0.0, 0.88,
                "- The ARM is FASTER than this FPGA engine:\n"
                "  6x with an event-driven program, 1.4x even running\n"
                "  the dense algorithm the FPGA uses.\n\n"
                "- The engine is small on purpose: 8 channels at a\n"
                "  time, 1 weight per clock, 25 MHz, 13% of the chip\n"
                "  (HPS included). The 25 MHz clock closes with 10.8 ns\n"
                "  of slack: about 34 MHz without pipelining work.\n"
                "  More parallel channels or an event-driven engine\n"
                "  are the levers for a large speed-up; neither\n"
                "  was built.\n\n"
                "- What the FPGA gives today: a fixed, repeatable\n"
                "  latency (222,236 to 222,245 clocks, 9 clocks of\n"
                "  jitter over 8,000 frames) and a CPU that is free.\n\n"
                "- Energy per frame cannot be compared: the ARM's\n"
                "  power was not measured, and the FPGA number is\n"
                "  an estimate. A USB power meter is needed.", fontsize=9, va="top", family="DejaVu Sans")
    save(fig, "fig_arm_power.png")


def fig_ood():
    import json
    d = json.load(open(ROOT / "docs" / "results" / "ood_board.json"))
    rows = d["rows"][::-1]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [2.4, 1]}, constrained_layout=True)
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for k, c in enumerate(CLASSES):
        w = np.array([r["window_hist"].get(c, 0) / max(r["windows"], 1) * 100 for r in rows])
        axs[0].barh(y, w, left=left, color=CCOL[k], label=c)
        for i, v in enumerate(w):
            if v >= 9:
                axs[0].text(left[i] + v / 2, y[i], "%.0f" % v, ha="center", va="center", fontsize=7.5, color="white")
        left += w
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(["%s%s" % (r["clip"], ("  [expect: %s]" % r["weak_label"]) if r["weak_label"] else "  [no correct answer]") for r in rows], fontsize=8.5)
    axs[0].set_xlabel("share of 1-second window answers (%)")
    axs[0].set_title("What the board answers on footage that is not KTH (6,280 frames, board = model on every frame)")
    axs[0].legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    axs[1].barh(y, [r["mean_events_pct"] for r in rows], color="#888")
    for i, r in enumerate(rows):
        axs[1].text(r["mean_events_pct"] + 0.3, y[i], "%.1f%%" % r["mean_events_pct"], va="center", fontsize=8)
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([])
    axs[1].set_xlabel("events per frame / pixels (%)")
    axs[1].set_title("Event load (KTH is 1.4-4.7%)")
    save(fig, "fig_ood.png")


def fig_emulator_lut():
    from model.log_lut import LUT
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
    p = np.arange(256)
    axs[0].plot(p, LUT, color="#2E7D32", lw=2)
    axs[0].set_xlabel("pixel value p")
    axs[0].set_ylabel("log intensity L[p] = round(255 * log2(1 + p) / 8)")
    axs[0].set_title("Log LUT: equal brightness ratios give equal steps")
    for th, col in [(9, "#E45756")]:
        for p0 in (20, 80, 160):
            lo = LUT[p0]
            hi = np.searchsorted(LUT, lo + th)
            axs[0].annotate("", xy=(hi, lo + th), xytext=(p0, lo), arrowprops=dict(arrowstyle="->", color=col))
            axs[0].text(hi + 3, lo + th - 10, "+%d%%" % round((hi + 1) / (p0 + 1) * 100 - 100), color=col, fontsize=8)
    axs[0].text(125, 25, "theta = 9 LUT steps ~ 0.2 ln units:\nan event needs a ~22% brightness change\nat any brightness", color="#E45756", fontsize=9)
    # event bit-exactness illustration: reference update modes
    t = np.arange(0, 30)
    sig = 90 + 70 * np.tanh((t - 10) / 2.5)
    ref_step, ref_reset, ev_step, ev_reset = [sig[0]], [sig[0]], [], []
    th = 12
    for k in range(1, len(t)):
        d = sig[k] - ref_step[-1]
        r = ref_step[-1]
        if abs(d) >= th:
            r += th * np.sign(d)
            ev_step.append(k)
        ref_step.append(r)
        d = sig[k] - ref_reset[-1]
        r = ref_reset[-1]
        if abs(d) >= th:
            r = sig[k]
            ev_reset.append(k)
        ref_reset.append(r)
    axs[1].plot(t, sig, "k-", lw=2, label="log intensity of one pixel")
    axs[1].step(t, ref_step, where="post", color="#4C78A8", label="reference, 'step' mode (default): moves by theta")
    axs[1].step(t, ref_reset, where="post", color="#F58518", ls="--", label="reference, 'reset' mode: jumps to the value")
    axs[1].plot(ev_step, [-5] * len(ev_step), "|", color="#4C78A8", ms=14, mew=2)
    axs[1].plot(ev_reset, [-12] * len(ev_reset), "|", color="#F58518", ms=14, mew=2)
    axs[1].set_ylim(-20, 190)
    axs[1].set_xlabel("frame")
    axs[1].set_title("Frame-based event emulation: a big change spreads over frames in 'step' mode")
    axs[1].legend(fontsize=7.5, loc="lower right")
    save(fig, "fig_emulator.png")


# ----------------------------------------------------------------------------------------------------------------- GIFs
def kth_clip(action, person=22, cond="d1", rep=0, max_frames=130, skip=0):
    from train.kth import parse_sequences, read_all_frames, resample_linear
    seqs = parse_sequences(ROOT / "data" / "kth" / "00sequences.txt")
    key = "person%02d_%s_%s" % (person, action, cond)
    _, _, _, ranges = seqs[key]
    a, b = ranges[rep]
    path = ROOT / "data" / "kth" / "videos" / (key + "_uncomp.avi")
    frames = read_all_frames(path, fit="squash")[a - 1:b]
    out = resample_linear(frames)
    return out[skip:skip + max_frames]


class Runner:
    """Per-frame answers either from the Python models (sim) or from the real board."""

    def __init__(self, connect=None):
        from sw.host.snnboard_sim import SimBoard
        from sw.host import snnproto as P
        self.P = P
        self.sim = SimBoard(str(ROOT / "rtl" / "weights" / "kth6"))
        self.client = None
        if connect:
            host, _, port = connect.partition(":")
            self.client = P.Client(host, int(port) if port else 5002)
            self.client.hello()
        self.tag = "board" if connect else "sim"

    def run(self, frames):
        from model.emulator_int import EmuConfig, IntEmulator
        emu = IntEmulator(EmuConfig(theta=9, mode="step"))
        self.sim._reset_state()
        if self.client:
            self.client.reset()
        out = []
        for i, f in enumerate(frames):
            words = emu.step(f)
            if self.client:
                self.client.send_frame(f)
                res = self.client.recv_result()
            else:
                res = self.sim.process(f, i + 1)
            out.append((words, res))
        return out


def events_rgb(words, size=128):
    from model.event_format import unpack
    img = np.full((size, size, 3), 28, np.uint8)
    if len(words):
        u = unpack(words)
        on = u["pol"] == 1
        img[u["y"][on], u["x"][on]] = (60, 230, 90)
        img[u["y"][~on], u["x"][~on]] = (235, 60, 60)
    return img


def compose(frame, words, res, name, title_tag):
    import cv2
    S = 224
    H, W = 292, 3 * S + 250
    c = np.full((H, W, 3), 255, np.uint8)
    g = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    c[40:40 + S, 8:8 + S] = cv2.resize(g, (S, S), interpolation=cv2.INTER_NEAREST)
    c[40:40 + S, 16 + S:16 + 2 * S] = cv2.resize(events_rgb(words), (S, S), interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(c, "input (128x128)", (10, 32), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(c, "DVS events: green ON, red OFF", (16 + S + 2, 32), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    x0 = 24 + 2 * S
    cv2.putText(c, "running window score", (x0, 32), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    sc = np.array(res.score, np.float64)
    span = max(np.abs(sc).max(), 400.0)
    lead = int(np.argmax(sc))
    cols = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in CCOL]
    for k in range(6):
        y = 48 + 34 * k
        cv2.putText(c, CLASSES[k], (x0, y + 14), font, 0.42, (0, 0, 0) if k == lead else (120, 120, 120), 1, cv2.LINE_AA)
        mid = x0 + 175
        w = int(80 * sc[k] / span)
        cv2.rectangle(c, (mid, y), (mid + w, y + 16), cols[k] if k == lead else (190, 190, 190), -1)
    cv2.line(c, (x0 + 175, 44), (x0 + 175, 48 + 34 * 6), (80, 80, 80), 1)
    ans = ("window answer: %s" % CLASSES[res.pred]) if res.windows else "window answer: (collecting)"
    cv2.putText(c, ans, (8, 40 + S + 22), font, 0.55, (20, 110, 20) if res.windows else (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(c, "step %2d/32   events %4d   spikes L1 %4d  L2 %4d" % (res.wcount if res.wcount else (32 if res.windows else 0), res.events, res.spk1, res.spk2),
                (8 + 330, 40 + S + 22), font, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(c, "true action: %s     [%s]" % (name, title_tag), (8, 14), font, 0.5, (150, 30, 30), 1, cv2.LINE_AA)
    return c


def write_gif(imgs, path, fps=15):
    from PIL import Image
    pil = [Image.fromarray(i).convert("P", palette=Image.ADAPTIVE, colors=64) for i in imgs]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    print("wrote", path, "%.0f KB" % (Path(path).stat().st_size / 1024))


def gifs(connect=None, frames_per_gif=100):
    MEDIA.mkdir(parents=True, exist_ok=True)
    runner = Runner(connect)
    series = {}
    for k, action in enumerate(["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]):
        clip = kth_clip(action, rep=1, max_frames=frames_per_gif)
        res = runner.run(clip)
        series[action] = (clip, res)
        title = CLASSES[k]
        imgs = [compose(f, w, r, title, runner.tag) for f, (w, r) in list(zip(clip, res))[::2]]       # 15 fps from 30 fps
        write_gif(imgs, MEDIA / ("demo_%s_%s.gif" % (action, runner.tag)), fps=15)
    return series, runner


def fig_clip_scores(series, tag):
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 7.2), sharey=False, constrained_layout=True)
    for ax, (k, (action, (clip, res))) in zip(axs.ravel(), enumerate(series.items())):
        sc = np.array([r.score for _, r in res])
        for c in range(6):
            ax.plot(np.arange(1, len(sc) + 1), sc[:, c], color=CCOL[c], lw=2.4 if c == k else 1.2, label=CLASSES[c])
        for w in range(32, len(sc) + 1, 32):
            ax.axvline(w, color="#999", ls=":")
        ax.set_title("true action: %s" % CLASSES[k], fontsize=10, color=CCOL[k], fontweight="bold")
        ax.set_xlabel("frame (running score resets every 32 frames)")
        ax.set_ylabel("summed class current")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Running window scores on one test-person clip per action (answers from the %s)" % ("real board" if tag == "board" else "bit-exact Python model; the RTL is verified identical"),
                 fontweight="bold")
    save(fig, "fig_clip_scores_%s.png" % tag)


def fig_sparsity(series, tag):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.4))
    evs = {a: np.array([r.events for _, r in res]) for a, (_, res) in series.items()}
    s1 = {a: np.array([r.spk1 for _, r in res]) for a, (_, res) in series.items()}
    s2 = {a: np.array([r.spk2 for _, r in res]) for a, (_, res) in series.items()}
    names = list(series)
    x = np.arange(6)
    m = [evs[a][1:].mean() / 16384 * 100 for a in names]
    axs[0].bar(x, m, color=CCOL)
    for i in range(6):
        axs[0].text(x[i], m[i] + 0.1, "%.1f%%\n(%d of 16,384)" % (m[i], evs[names[i]][1:].mean()), ha="center", fontsize=8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(CLASSES, rotation=20, fontsize=8.5)
    axs[0].set_ylabel("events per frame / pixels (%)")
    axs[0].set_ylim(0, max(m) * 1.45)
    axs[0].set_title("Event sparsity: the emulator output is a few percent of the pixels")
    m1 = [s1[a][1:].mean() / 16384 * 100 for a in names]
    m2 = [s2[a][1:].mean() / 8192 * 100 for a in names]
    axs[1].bar(x - 0.2, m1, 0.4, color="#B279A2", label="layer 1 (of 16,384 neurons)")
    axs[1].bar(x + 0.2, m2, 0.4, color="#54A24B", label="layer 2 (of 8,192 neurons)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(CLASSES, rotation=20, fontsize=8.5)
    axs[1].set_ylabel("neurons spiking per step (%)")
    axs[1].set_title("Network activity per time step")
    axs[1].legend(fontsize=8)
    save(fig, "fig_sparsity_%s.png" % tag)


def gif_theta():
    """Same frames, four emulator thresholds side by side."""
    import cv2
    from model.emulator_int import EmuConfig, IntEmulator
    clip = kth_clip("boxing", rep=1, max_frames=80)
    ths = [3, 9, 17, 33]
    emus = [IntEmulator(EmuConfig(theta=t, mode="step")) for t in ths]
    imgs = []
    for f in clip:
        tiles = []
        for t, e in zip(ths, emus):
            w = e.step(f)
            tile = cv2.resize(events_rgb(w), (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, "theta %d  (%d events)" % (t, len(w)), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        imgs.append(np.concatenate(tiles, axis=1))
    write_gif(imgs[::2], MEDIA / "demo_threshold.gif", fps=15)


def gif_pattern():
    """The on-chip pattern source through the emulator: what the first VGA demo shows on the monitor."""
    import cv2
    from model.emulator_int import EmuConfig, IntEmulator
    from model.pattern_source import pattern_frame
    emu = IntEmulator(EmuConfig(theta=9, mode="step"))
    imgs = []
    for f in range(60):
        fr = pattern_frame(f)
        w = emu.step(fr)
        a = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB), (256, 256), interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(events_rgb(w), (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.putText(a, "pattern source (RTL pix_source)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(b, "events, theta 9 (VGA event view)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        imgs.append(np.concatenate([a, b], axis=1))
    write_gif(imgs, MEDIA / "demo_pattern.gif", fps=15)


def gif_montage(tag):
    from PIL import Image
    names = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
    srcs = [Image.open(MEDIA / ("demo_%s_%s.gif" % (n, tag))) for n in names]
    n = min(s.n_frames for s in srcs)
    frames = []
    for i in range(n):
        tiles = []
        for s in srcs:
            s.seek(i)
            im = s.convert("RGB")
            tiles.append(im.resize((im.width * 2 // 3, im.height * 2 // 3)))
        w, h = tiles[0].size
        canvas = Image.new("RGB", (2 * w, 3 * h), "white")
        for k, t in enumerate(tiles):
            canvas.paste(t, ((k % 2) * w, (k // 2) * h))
        frames.append(np.array(canvas))
    write_gif(frames, MEDIA / ("demo_all_actions_%s.gif" % tag), fps=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["figs", "gifs", "all"], default="all")
    ap.add_argument("--connect", help="HOST[:PORT] of the board: GIFs and score plots use the board's answers")
    a = ap.parse_args()
    if a.only in ("figs", "all"):
        fig_architecture()
        fig_pipeline()
        fig_snn_arch()
        fig_trackA()
        fig_training()
        fig_kth_results()
        fig_domain_gap()
        fig_hardware()
        fig_emulator_lut()
        fig_board()
        fig_arm_power()
        fig_ood()
    if a.only in ("gifs", "all"):
        series, runner = gifs(a.connect)
        fig_clip_scores(series, runner.tag)
        fig_sparsity(series, runner.tag)
        gif_montage(runner.tag)
        if not a.connect:
            gif_theta()
            gif_pattern()


if __name__ == "__main__":
    main()
