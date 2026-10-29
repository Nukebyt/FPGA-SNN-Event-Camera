# SNN SoC (video clips from the laptop): verification and build results, 2026-10-29

**Update:** the image was run on the board the same day: bit-exact against the models on 3,600 frames, see `board_run.md`.

Design: `quartus/snn_soc` (HPS + DDR3 + bridges from the night-vision SoC, which ran on this board; new engine `rtl/snn_engine.v`). How to use it: `docs/snn_soc.md`. **Not yet run on the board.**

## Simulation (Icarus Verilog 14)

`tb/snn_engine_tb.v` drives the engine like the ARM daemon does: frames written into a four-slot DDR3 model (64-bit reads, 4 clocks latency, random wait states, one access in four stalled), enable / ring base / `IN_WR_SEQ` / `DONE_SEQ` through the CSR block, up to three frames queued, results read back from the CSR snapshots and compared with the Python chain model (emulator -> bins -> `snn_int.py`).

| Case | Frames | Result |
|---|---|---|
| real KTH boxing video `person22_boxing_d1` (frames 10 to 15, squashed to 128x128 exactly as in training), frozen KTH network | 6 | per-step spike counts and 6 class currents, window class (boxing) and window scores identical; DMA read exactly 6 x 2,048 words; no error flags |
| on-chip pattern frames, random network (`scripts/snn_regression.py`, case `snn_engine`) | 5 | identical |

CSR measurements from the bench: network step 205,831 clocks (8.2 ms at 25 MHz), DMA start to published result 222,233 clocks (8.9 ms), so the engine could take about 110 frames/s; the laptop sends 30.
Mutation self-test additions: wrong ring slot address, wrong vector in a CSR snapshot, `DONE_SEQ` reporting the read counter (`scripts/snn_regression.py --selftest`).
Laptop side (`sw/tests/test_snn_stream.py`, in `pytest`): frame preparation equals the training squash; the 30 fps resampler equals `train.kth.resample_linear` at 25, 24, 50 and 29.97 fps; RESULT payload round trip; the simulated board agrees with the integer model and its window logic; the C daemon, compiled with `-Wall -Wextra` in WSL against a fake engine, answers HELLO / FRAME_IN / RESULT / RESET / CSR messages correctly, ignores client writes to `IN_WR_SEQ` and refuses a wrong-size HELLO.
`scripts/check_board.py` was checked against the simulated board: PASS when the answers agree, FAIL (39 of 40 frames differ) when the simulated board uses another threshold.

## Quartus 21.1 build (`bash scripts/build_snn_soc.sh --generate`, about 25 minutes including Qsys generation)

| | |
|---|---|
| ALMs | 5,243 / 41,910 (13%), the HPS and its DDR3 controller included |
| Registers | 7,602 |
| Block memory | 603,904 bits in 79 / 553 M10K (11%) |
| DSP | 0 |
| Pins | 338 / 499 (the HPS DDR3 and peripheral pins) |
| Bitstream | `DE10_Standard_SNN.rbf` 2,422,232 bytes (compressed; the night-vision image was 3,297,112), `.sof` 7,425,855 |
| STA | `STA: PASS` at all four corners; the 25 MHz PLL clock, `CLOCK_50`, the JTAG clock and the HPS DDR3 clocks are all in the slack tables. Worst setup slack: +1.73 ns (HPS SDRAM clock domain, slow 0 C), +10.3 ns JTAG, +10.8 ns for the 25 MHz clock (slow 85 C), +12.5 ns CLOCK_50. Worst hold slack +0.077 ns (fast 0 C, HPS domain) |
| Critical warnings | 3, all expected: no user location for 72 HPS DDR3/peripheral pins and for the RZQ pin (the HPS pins are fixed by the device), and the 36-word layer-1 weight ROM in a 64-word memory |
| Memory images | `scripts/check_mif.py`: the extracted images of `w1.hex`, `w2.hex` and `wfc.hex` equal the hex files word for word (the clip ROM of the standalone demo is not part of this design) |

Sources and their SHA-256 are archived with the bitstream in `releases/snn_soc_build1/`.

## What was not known at build time, and is now (`board_run.md`)
The image loads from the SD card, the bridge answers at `0xFF210000` with `0x534E0001`, the engine's DMA reads are correct, and the Ethernet path carries 84.7 frames/s of 16 KiB frames (29.4 at the real-time pace) once the daemon sends each reply in a single write.
Still not measured: power, an ARM-only baseline, photographs of the display.
