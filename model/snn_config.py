"""SNN architecture + neuron constants shared by training, the integer model and (later) the RTL.

    input  [2, H, H] binary spikes per time step (OFF/ON x downsampled event presence), H = in_hw (32 or 64)
    conv1  2->C1, 3x3, pad 1, LIF            -> [C1, H, H]
    pool1  2x2 spike OR                      -> [C1, H/2, H/2]
    conv2  C1->C2, 3x3, pad 1, LIF           -> [C2, H/2, H/2]
    pool2  2x2 spike OR                      -> [C2, H/4, H/4]
    [conv3 C2->C3, 3x3, pad 1, LIF -> [C3, H/4, H/4]; pool3 2x2 spike OR -> [C3, H/8, H/8]]   only if c3 > 0
    fc     C_last*(H/2^n)^2 -> classes       -> per-class score over the window (readout, see below)

Readout:
    "sum"   (default) non-spiking output: score[c] = sum over the window of the FC input current of class c;
            11 integer accumulators, no threshold, no reset. Trains reliably (no dead or saturated output spikes).
    "spike" output LIF neurons, score = spike count (kept for comparison; collapses on dense DVS128 input)

Neuron (integer only, no multipliers):
    v <- v - (v >> leak_shift)           arithmetic shift (floor)
    v <- clip(v + I, vmin, vmax)         I = sum of int8 weights of active inputs
    s  = v >= thr
    v <- v - thr * s                     reset by subtraction
"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class SNNConfig:
    in_hw: int = 32
    in_ch: int = 2
    c1: int = 16
    c2: int = 32
    c3: int = 0                                               # 0 = two conv layers, > 0 adds a third conv + pool
    classes: int = 11
    leak_shift: int = 4
    thr: list = field(default_factory=lambda: [64, 64, 64])   # one per conv layer, then the output layer
    vbits: int = 12                                           # membrane width, signed
    wscale: int = 16                                          # latent float weight * wscale = int weight
    bin_us: int = 33333                                       # one time step = one 30 fps frame
    readout: str = "sum"                                       # "sum" or "spike"

    def __post_init__(self):
        if self.c3 and len(self.thr) == 3:            # 2-conv style list given: conv3 reuses the third value
            self.thr = [self.thr[0], self.thr[1], self.thr[2], self.thr[2]]
        assert len(self.thr) == self.n_conv + 1, "thr needs one entry per conv layer plus one for the output"

    @property
    def channels(self):
        return [self.c1, self.c2] + ([self.c3] if self.c3 else [])

    @property
    def n_conv(self):
        return 3 if self.c3 else 2

    @property
    def down(self):
        """Spatial downsampling factor from the 128x128 emulator/sensor grid (4 -> 32x32, 2 -> 64x64)."""
        return 128 // self.in_hw

    @property
    def vmin(self):
        return -(1 << (self.vbits - 1))

    @property
    def vmax(self):
        return (1 << (self.vbits - 1)) - 1

    @property
    def fc_in(self):
        return self.channels[-1] * (self.in_hw >> self.n_conv) ** 2

    def sizes(self):
        """Parameter and state sizes, for the M10K budget (roadmap 3.2)."""
        chans = [self.in_ch] + self.channels
        params = {"conv%d" % (i + 1): chans[i + 1] * chans[i] * 9 for i in range(self.n_conv)}
        params["fc"] = self.classes * self.fc_in
        state = {"conv%d" % (i + 1): chans[i + 1] * (self.in_hw >> i) ** 2 for i in range(self.n_conv)}
        state["fc"] = self.classes
        return {"weights_int8": params, "membrane_words": state,
                "membrane_bits": {k: v * self.vbits for k, v in state.items()},
                "weight_bits": {k: v * 8 for k, v in params.items()}}

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        known = {f for f in SNNConfig.__dataclass_fields__}
        return SNNConfig(**{k: v for k, v in d.items() if k in known})      # tolerate keys of older exports
