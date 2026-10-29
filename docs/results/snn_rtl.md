# SNN RTL: verification and build results (2026-10-28)

Design: `rtl/snn_demo_top.v` (emulator + bin builder + SNN + clip replay + VGA event view), network and RTL description in `docs/snn_arch.md`. Network: KTH 6-class, 32x32, conv 16/32, int8 (`docs/results/kth6_accuracy.md`).

## Bit-exactness against `model/snn_int.py` (Icarus Verilog 14, `scripts/snn_regression.py`)

| Case | Bench | What is compared | Result |
|---|---|---|---|
| `snn_rand`: random int8 network, 12% dense bins, 6 steps | `tb/snn_core_tb.v` | per step: layer-1 and layer-2 spike counts, the 6 class currents; after the last step: all 2,048 + 1,024 membrane words; then window class and the 6 summed scores | 0 mismatches |
| `snn_sat`: network that drives both membrane rails (-2048 and 2047), int8 extremes (-128, 127), 70% dense bins, 8 steps, two identical dominant output classes (an exact score tie) | `tb/snn_core_tb.v` | as above | 0 mismatches; the tie must give class 0 |
| `snn_chain`: pattern source -> RTL emulator -> frame_sync -> bin builder -> core, random network, 5 frames | `tb/snn_chain_tb.v` | as above against `pattern_frame -> emulate -> bin_emulated -> IntSNN`; no emulator error flag; no skipped step | 0 mismatches |
| `snn_kth6`: the frozen KTH network on a real test repetition, 8 steps | `tb/snn_core_tb.v` | as above | 0 mismatches |
| 32-frame LIVE window with the KTH network (one full decision window) | `tb/snn_chain_tb.v` | as above, 30,674 emulator events | 0 mismatches, class 2 |
| board top level, REPLAY clips 0 and 5 and LIVE (4-frame window) | `tb/snn_demo_top_tb.v` | HEX0 digit, HEX1/2/3, class LEDs, error LED, controller class vs the model | clip 0 -> 1, clip 5 -> 5, LIVE -> 0, all equal to the model, 0 errors |

Every step takes exactly 205,832 clocks in simulation (layer 1 43,008 + layer 2 150,528 + classifier 12,288 + 8): 8.2 ms at 25 MHz.

Mutation self-test (`python scripts/snn_regression.py --selftest`): 15 deliberately broken copies of the RTL (the last three on the DDR3-facing engine: wrong ring slot, wrong CSR snapshot vector, DONE_SEQ reporting the read counter; the first twelve: logical instead of arithmetic leak shift, pooling by AND, membrane clip off by one at each rail, wrong right-edge padding, no reset by subtraction, classifier weight not sign-extended, wrong pooled-word address, argmax `>=` instead of `>`, event x/y swapped in the bin address, bank not swapped, frame closed early): **14 caught, 1 survived** (`frame_sync` condition `pend && idle_dd` instead of `... && emu_idle && idle_d && idle_dd`: the two extra idle clocks are margin that this simulation cannot tell apart, no hard requirement; it stays in the design). The first run missed three (both clip mutations and the tie-break): the bench had not been looking at the membranes and had no tie; both were added (bug log 28) and the three are caught now.

Not done: hundreds of real windows in simulation (a window takes about one minute in Icarus); a Questa or post-fit gate-level simulation. Quartus turned each `$readmemh` image into a memory initialisation file; `scripts/check_mif.py` checks that these hold exactly the words of `w1.hex`, `w2.hex`, `wfc.hex` and `clips.hex` (all four match).

## Quartus 21.1 build (`quartus/snn_demo`, `bash scripts/build_quartus.sh quartus/snn_demo snn_demo divclk`)

| | |
|---|---|
| ALMs | 1,694 / 41,910 (4%) |
| Registers | 1,468 |
| Block memory | 1,440,768 bits in 175 / 553 M10K (32%): clip ROM 786 kbit, membranes 295 kbit, classifier weights 98 kbit, emulator reference RAM and event FIFO |
| DSP | 2 (the 3x upscaler of the VGA event view; the SNN uses none) |
| PLLs | 1 of 15, pins 96 of 499 |
| Timing (25 MHz PLL clock), all four corners | setup slack +15.0 ns at 85 C slow (worst), hold slack +0.12 ns worst, recovery/removal and pulse width positive; `STA: PASS` including the check that the PLL clock actually appears in the tables |
| Critical warnings | 1: the layer-1 weight ROM has 36 words in a 64-word memory (remaining 28 addresses read 0 and are never addressed) |
| Bitstream | `releases/snn_demo_build1/` (`.sof`, `.rbf`, sources fingerprint) |

For comparison the first demo (emulator + VGA only) used 288 ALMs and 27 M10K, so the network adds about 1,400 ALMs and 150 M10K; the design uses about a third of the block memory, mostly the demo clip ROM (the network itself is 17 k weights).
