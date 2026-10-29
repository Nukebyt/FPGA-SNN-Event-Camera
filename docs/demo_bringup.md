# On-chip demo: first thing to run on the board (roadmap 1.11 / 4.11)

Bitstream: `quartus/demo/output_files/demo.sof` (build: `bash scripts/build_quartus.sh quartus/demo demo divclk`).
Built result (Quartus 21.1): 288 ALMs, 442 registers, 214,016 block-memory bits in 27 M10K, 2 DSP (3x upscaler), 1 PLL, 96 pins. STA: all corners positive; clocks in the tables: `CLOCK_50`, PLL `vcoph[0]`, PLL `divclk` (25 MHz, setup slack +31.6 ns at 85 C slow).
Simulation evidence: 6 + 5 complete 640x480 VGA frames pixel-exact against `model/pattern_source.py` -> `model/emulator_int.py` -> event-view rules (`scripts/run_regression.py`).

## Steps
1. Connect power, the USB-Blaster II cable and a VGA monitor. Run `jtagconfig` and read the chain. On this SoC part the chain has two devices (HPS DAP and the FPGA). Note the FPGA's device index in `docs/board_notes.md`; never program with the bare default (checklist 2.4a).
2. Program the SRAM image: `quartus_pgm -m jtag -o "p;quartus/demo/output_files/demo.sof@<N>"` with `<N>` from step 1 (nothing is written to flash; power cycle clears it).
3. Expected within a second: `LEDR[1]` (PLL locked) and `LEDR[2]` (out of reset) on, `LEDR[0]` blinking about every 1.3 s, HEX3..0 counting emulator frames (about 30 per second, hex), HEX4 = 0 (err_flags), HEX5 = 0 (overflow count).
4. Monitor: a dark grey square (the event view, 384x384) centred on black. A 24x24 object (frame source) moves down-right at 3 and 2 pixels per frame: a **green** block (brightness increased, ON) with a **red** trail (OFF) behind it, wrapping around the edges. The static gradient background produces no events.

## Controls
| Control | Effect |
|---|---|
| `SW[4:0]` | threshold theta = 2*SW + 1 LUT steps (SW = 4 -> 9 ~ 0.2 ln units). Larger = fewer events. The square (value 200, LUT 244) differs from the background (values 40-71, LUT 171-197) by 47-73 LUT steps, so from SW = 24 (theta 49) up the part of its edge that lies over the brighter background stops firing, while the part over the darkest background still fires at the maximum theta of 63. The static gradient never produces events. |
| `SW[5]` | 0 = reference moves by theta ("step"), 1 = reference reset to current value ("reset") |
| `SW[7:6]` | event view accumulates 1, 2, 4 or 8 source frames (longer trail) |
| `SW[8]` | pause the frame source |
| `KEY[0]` | reset |
| `KEY[1]` | forget the reference: the next frame is an init frame with no events, timestamps restart |
| `KEY[2]` | clear error flags and counters |
| `LEDR[3]` | emulator idle, `LEDR[4]` any error flag, `LEDR[5]` source frame in progress |

## What to record if something differs
Anything that does not match the list above goes into `docs/bug_log.md` in the format Symptom / Root cause / Fix / Lesson, with exact numbers (checklist 7.3). Likely suspects, in order: VGA output timing against the ADV7123 (the SDC does not constrain it; see `docs/board_notes.md`), pin assignment mismatch (pins are copied from the sister project, where `CLOCK_50`, LEDs and HEX were proven on hardware), SW voltage standard.
