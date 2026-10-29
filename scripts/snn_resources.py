"""Rough on-chip memory and work estimate for SNN sizes (roadmap 3.2 / 3.12).

    python scripts/snn_resources.py [HW:C1,C2[,C3]]...      e.g.  python scripts/snn_resources.py 32:16,32 32:16,32,64 64:16,32

Memory is a lower bound (bits / 10,240 per M10K); real block counts are higher because of width/depth
aspect ratios and per-channel banking. Device: 5CSXFC6D6F31C6 has 553 M10K (5,662,720 bits).
Emulator probe uses 21 M10K (reference RAM, event FIFO, log LUT).
Work per time step = synaptic additions if every active input spike updates its 3x3 x C_out neighbourhood
(event-driven conv). Rates are per-neuron-per-step firing (table MEASURED, or --rates: input density, then one
per conv layer, the last value reused for deeper layers); pooled spikes are estimated as 2x the per-neuron rate
(OR of 4 neurons, capped at 1), which is rough. Measure with `train.eval_int` for a real number.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.snn_config import SNNConfig  # noqa: E402

M10K_BITS, M10K_TOTAL, EMU_M10K = 10240, 553, 21


# Measured with train.eval_int on the DVS128 test set (48-step windows): input density, then firing per neuron
# per step of each conv layer. Unknown configurations fall back to the 32x32 16/32 values (input density by hw).
MEASURED = {
    "32:8,16": (0.166, 0.0862, 0.0191),
    "32:16,32": (0.166, 0.0558, 0.0088),
    "64:16,32": (0.070, 0.0600, 0.0021),
}


def report(hw, chans, rates=None, spec=None):
    rates = rates or MEASURED.get(spec) or ((0.166 if hw == 32 else 0.070),) + MEASURED["32:16,32"][1:]
    c = SNNConfig(in_hw=hw, c1=chans[0], c2=chans[1], c3=chans[2] if len(chans) > 2 else 0)
    z = c.sizes()
    wbits, mbits = sum(z["weight_bits"].values()), sum(z["membrane_bits"].values())
    blocks = math.ceil((wbits + mbits) / M10K_BITS) + EMU_M10K
    in_act = rates[0] * 2 * hw * hw                       # active input cells per step
    ops = in_act * 9 * c.c1
    prev_neurons = c.c1 * hw * hw
    for i in range(1, c.n_conv):
        r_prev = rates[min(i, len(rates) - 1)]
        pooled_active = min(1.0, 2 * r_prev) * prev_neurons / 4
        ops += pooled_active * 9 * c.channels[i]
        prev_neurons = c.channels[i] * (hw >> i) ** 2
    r_last = rates[min(c.n_conv, len(rates) - 1)]
    ops += min(1.0, 2 * r_last) * prev_neurons / 4 * c.classes     # FC
    print("%2dx%-2d  channels %-10s | weights %6d x int8 = %5.0f kbit | membrane %6d x %d bit = %5.0f kbit | >= %3d M10K (%2.0f%%) incl. emulator | ~%7.0f synaptic adds/step"
          % (hw, hw, "/".join(map(str, chans)), sum(z["weights_int8"].values()), wbits / 1e3, sum(z["membrane_words"].values()),
             c.vbits, mbits / 1e3, blocks, 100 * blocks / M10K_TOTAL, ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", nargs="*", help="HW:C1,C2[,C3]")
    ap.add_argument("--rates", default=None, help="override: input density then per-layer firing per neuron per step")
    a = ap.parse_args()
    rates = [float(v) for v in a.rates.split(",")] if a.rates else None
    for spec in a.cfg or ["32:8,16", "32:16,32", "32:16,32,64", "32:16,32,32", "64:16,32"]:
        hw, ch = spec.split(":")
        report(int(hw), [int(v) for v in ch.split(",")], rates, spec)


if __name__ == "__main__":
    main()
