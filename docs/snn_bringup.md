# SNN demo: what to do on the board (first hardware run of the neural network)

Bitstream: `quartus/snn_demo/output_files/snn_demo.sof` (copy in `releases/snn_demo_build1/`). Build and resource numbers: `docs/results/snn_rtl.md`.
**First run (2026-10-28):** programmed over JTAG as in section 1 (configuration succeeded) and reported working by the user; the per-clip results and LIVE mode were not itemised. Before that, everything below was checked in simulation, including the board top level with its keys, switches, 7-segment displays and LEDs (`tb/snn_demo_top_tb.v`), so a difference on the board points at the board side (pins, programming, power) before it points at the logic.

The design is the first demo (`docs/demo_bringup.md`) plus the neural network: pattern source -> event emulator -> 2x32x32 event bins -> integer SNN -> class of the last 1.07 s window. No HPS and no Linux are needed, only JTAG.

## 1. Program it
1. Power the board, connect the USB-Blaster II cable and a VGA monitor (the monitor is only needed for LIVE mode).
2. `C:\intelFPGA_lite\21.1\quartus\bin64\jtagconfig` and read the chain. On this SoC part there are two devices (the HPS DAP and the FPGA); note the FPGA index N in `docs/board_notes.md`. Never program with the bare default index (checklist 2.4a).
3. `C:\intelFPGA_lite\21.1\quartus\bin64\quartus_pgm -m jtag -o "p;quartus/snn_demo/output_files/snn_demo.sof@N"` from the project folder. Nothing is written to flash; a power cycle clears it.
4. Switches at start: **all `SW` down**, except `SW[2]` up (`SW[4:0] = 00100`, which is the emulator threshold 9 the network was trained with; it is only used in LIVE mode). Press KEY[0] once (reset).

## 2. REPLAY mode (SW[9] = 0, the default): the network on real data

Six clips from the KTH test people (the first test repetition of each action, chosen without looking at what the network predicts), already turned into event bins by the Python emulator model, are played into the SNN at 30 steps per second. KEY[3] switches to the next clip (0 to 5, then back to 0) and restarts the window.

What you should see right after programming:
- `LEDR[0]` blinks (about every 1.3 s), `LEDR[1]` on (PLL locked), `LEDR[2]` flickers (the network is busy about a quarter of every frame), `LEDR[3]` **off**.
- `HEX5 HEX4` count the steps of the current window in hex, `00` to `1F`, about 30 per second, so about once per second.
- `HEX3` counts finished windows (low digit).
- `HEX1` = the clip number, `HEX2` = the true class of that clip.
- `HEX0` = `F` until the first window is finished (about 1 s after reset or after KEY[3]), then the class the network chose. `LEDR[9:4]` light the same class as one LED (LEDR[4] = class 0 ... LEDR[9] = class 5).

Class numbers: 0 walking, 1 jogging, 2 running, 3 boxing, 4 hand waving, 5 hand clapping.

**What the display must show.** The "Model predicts" column is what the Python integer model gives for that clip (both windows of a clip give the same answer); the hardware is bit-exact to that model, so `HEX0` must equal this column. The network is not perfect: for clips 0 and 2 the model itself picks jogging. A mismatch with the true class is a network error; a mismatch with the "model predicts" column would be a hardware fault.

| Clip (KEY[3] presses) | True class (HEX2) | Model predicts (HEX0 must show) | Correct? |
|---|---|---|---|
| 0 | 0 walking | **1** jogging | no (model error) |
| 1 | 1 jogging | **1** | yes |
| 2 | 2 running | **1** jogging | no (model error) |
| 3 | 3 boxing | **3** | yes |
| 4 | 4 hand waving | **4** | yes |
| 5 | 5 hand clapping | **5** | yes |

(Both 32-step windows of every 64-step clip give the same class. 4 of 6 right is not a measurement of accuracy; the measured accuracy is in `docs/results/kth6_accuracy.md`: 72.7% per window and 79.9% per repetition on the test people.)

Other checks in REPLAY mode:
- Hold the clip with `SW[8]` up: the step counter stops, `HEX0` keeps the last answer. Down again: it continues.
- `KEY[2]` clears the error flags. `LEDR[3]` must stay off during the whole test.
- The monitor shows the dark grey event view without any events in REPLAY mode (the emulator is idle); that is normal.

## 3. LIVE mode (SW[9] = 1): the whole chain on the board

The on-chip moving square goes through the RTL event emulator, the bin builder and the SNN, and the VGA event view shows the events as in the first demo. The square is not an action, so the class is arbitrary, but it is deterministic: the model predicts it.

1. Put `SW[9]` up **before** pressing KEY[0] (reset), with `SW[4:0] = 00100`, `SW[5] = 0`, `SW[7:6] = 00`, `SW[8] = 0`. (Reset with SW[9] up so the pattern starts at its frame 0, which is what the expectation assumes.)
2. The monitor shows the green block with a red trail moving down-right, as in `docs/demo_bringup.md`. `HEX5 HEX4` count steps, `HEX3` counts windows.
3. `HEX0` must show, for the first four windows: **2, 2, 2, 1** (running, running, running, jogging). `HEX1` and `HEX2` show `F` in LIVE mode. (Simulated: window 1 of the RTL chain equals the model with the real network, 32 frames, 0 mismatches; windows 2 to 4 are from the model.)
4. Changing the threshold switches or SW[5] changes the events and therefore the answer; those cases were not computed.
5. If `LEDR[3]` lights: either an emulator error flag (see `docs/demo_bringup.md`) or a step did not finish within one frame (it needs 8.2 ms of the 33.3 ms). Neither should happen.

## 4. If something differs from this page
Add it to `docs/bug_log.md` with the exact symptom (which switch settings, what HEX/LED showed). Likely causes in order: a pin or polarity mismatch (pins are copied from the sister project, where CLOCK_50, LEDs and HEX were proven; the VGA path is unchanged from the first demo), an old bitstream still loaded (power cycle and reprogram), the wrong switch state at reset (LIVE expectations assume `SW[9]` was already up at reset), then the logic itself. For a logic suspicion, rerun the simulation that matches: `.venv\Scripts\python.exe scripts\snn_regression.py --top` (about 20 minutes; replays clips 0 and 5 and the live chain through the real top level).

## 5. What comes next
Real video into the emulator on the board needs a host-written frame store instead of the pattern source. The mechanism already proven on this board in the sister project `F:\Projects\FPGA-Thermal-NUC_Motion` is the JTAG-to-Avalon host bus (`host/viewer.py`, `host/feed_clip.py`, frame store `rtl/scene_buf.v`, `rtl/host_if.v`); its HPS TCP feeder (`sw/frame_feeder.c`) is written but has not run on the board either. The KTH videos in `data/kth/videos` can then be streamed at 25/30 fps (the network was trained on frames resampled to 30 fps).
