# SNN architecture and RTL (roadmap 3.12 / Phase 5)

Frozen on 2026-10-28 for the KTH network. Golden model: `model/snn_int.py`. RTL: `rtl/snn_core.v`, `rtl/lif_conv.v`, `rtl/bin_builder.v`, `rtl/snn_ctrl.v`, `rtl/frame_sync.v`. Why these choices: `docs/architecture_decisions.md` D5, D6, D10, D11, D13.

## Network

| Stage | Shape | Operation |
|---|---|---|
| input | 2 x 32 x 32 bits per 33.3 ms step | channel 0 = OFF, 1 = ON; a bit is set when any emulator event of that polarity falls in the 4x4 pixel cell during the frame |
| conv1 | 2 -> 16, 3x3, zero padding | LIF, int8 weights, no bias, threshold 64 |
| pool1 | 16 x 16 x 16 | 2x2 OR of the spikes |
| conv2 | 16 -> 32, 3x3, zero padding | LIF, threshold 64 |
| pool2 | 32 x 8 x 8 = 2048 bits | 2x2 OR |
| fc | 2048 -> 6, int8, no bias | class currents, **not** spiking |
| window | T = 32 steps (1.07 s) | scores = sum of the class currents over the window; class = argmax, ties to the lowest index |

Neuron: `v <- v - (v >>> 4)` (arithmetic), `v <- clip(v + I, -2048, 2047)` (12-bit), spike if `v >= 64`, `v <- v - 64 * spike`. No multipliers: `I` is a sum of int8 weights of the active inputs. All states are cleared at the start of every window.

Classes (index): 0 walking, 1 jogging, 2 running, 3 boxing, 4 hand waving, 5 hand clapping (KTH, `train/kth.py` ACTIONS).

Sizes (`SNNConfig.sizes()`): weights conv1 288 + conv2 4,608 + fc 12,288 = 17,184 int8; membranes conv1 16,384 x 12 bit, conv2 8,192 x 12 bit, 6 output accumulators.

## Hardware

- One clock, 25 MHz (the PLL output of the demo), no second clock domain, 0 DSP blocks.
- `lif_conv`: output-stationary, 8 output channels per pass ("group"). For each position the CIN*9 input bits are fetched one per clock (zero padding is a masked fetch) and added into 8 18-bit accumulators; then leak, 12-bit saturation, threshold and reset-by-subtraction are applied and the 96-bit membrane word (8 x 12 bit) is written back; the 8 spikes are OR-ed into the pooled word (8 bits). No pooled-map clearing is needed: the first neuron of each 2x2 block writes the word, the other three OR into it.
- Weight ROMs are `$readmemh` images (`rtl/weights/kth6/w1.hex`, `w2.hex`, `wfc.hex`; layouts in `scripts/gen_snn_vectors.py`); `snn_core` runs layer 1, layer 2, then the classifier (one weight per clock).
- `bin_builder`: two banks of 2048 bits. Events set bits in the write bank; at the end of a frame (`frame_sync` waits until the emulator and its FIFO are idle) the banks swap and the other one is cleared (2048 clocks, with back-pressure on the event FIFO, nothing is dropped). The finished bank stays stable while the core reads it.
- `snn_ctrl`: starts a step per frame, counts T steps, latches the winning class and the 6 summed scores, clears the core. A frame that arrives while the previous step is still running sets a sticky `overrun` flag (LEDR[3]).

## Cost (Quartus 21.1, `quartus/snn_demo`, whole board design including the emulator and the VGA event view)

See `docs/results/snn_rtl.md` for the build numbers. Cycles per step, exact: layer 1 2,048 x 21 = 43,008; layer 2 1,024 x 147 = 150,528; classifier 12,288; total **205,832 clocks = 8.2 ms** at 25 MHz, a quarter of the 33.3 ms frame (measured in simulation: every step of every test takes exactly 205,832 clocks).
Memory: membranes 2,048 x 96 bit + 1,024 x 96 bit, pooled maps 512 x 8 + 256 x 8, weights 36 x 64 + 576 x 64 bit and 12,288 x 8 bit, input banks 2 x 2048 bit.

## Verification

| What | How | Result |
|---|---|---|
| core, builder, controller vs `snn_int.py` | `tb/snn_core_tb.v`: per step the layer-1 and layer-2 spike counts and the 6 class currents, after the last step the full contents of both membrane RAMs, then the window class and summed scores | random network, saturating network (both membrane rails, int8 extremes, an exact score tie), and the real KTH network on a real sample: 0 mismatches |
| emulator -> frame_sync -> builder -> core | `tb/snn_chain_tb.v`: the on-chip pattern source through the RTL emulator against `pattern_frame -> emulate -> bin_emulated -> IntSNN` | 0 mismatches, no emulator error flag, no skipped step |
| the board top level | `tb/snn_demo_top_tb.v`: keys, switches, 7-segment and LED outputs of `snn_demo_top` in REPLAY (clips chosen with KEY[3]) and LIVE | the digits and LEDs show the class the model predicts |
| Does the test see faults? | `scripts/snn_regression.py --selftest`: 15 deliberately broken copies of the RTL | see `docs/results/snn_rtl.md` |
| QAT forward pass vs integer model | `train/tests/test_qat_vs_int.py`, and `train.eval_kth6 --int-check` | identical predictions |

Layout of the ROM images is additionally round-trip tested in Python (`model/tests/test_snn_vectors.py`).

## Not covered

The board itself (nothing here has run on hardware yet); timing of the VGA outputs against the DAC (unchanged from the first demo); a different network size or more than two conv layers (the RTL has them fixed: `scripts/gen_snn_vectors.py` refuses other configurations).
